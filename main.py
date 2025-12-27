import os
import json
import time
import math
import shutil
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import random
import albumentations as A

from PIL import Image, ImageTk, ImageEnhance

# Optional but recommended for camera capture
try:
    import cv2
except Exception:
    cv2 = None

# YOLOE (Ultralytics)
try:
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
except Exception:
    YOLOE = None
    YOLOEVPSegPredictor = None

# RFDETR for training
try:
    from training import rfdetr_train
except Exception:
    rfdetr_train = None

# RFDETR for inference
try:
    from rfdetr import RFDETRBase
except Exception:
    RFDETRBase = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MODEL_PATH = "yoloe-11s-seg.pt"  # change to your local path if needed
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


@dataclass
class Box:
    # Stored in IMAGE pixel coords: (x1,y1,x2,y2)
    x1: float
    y1: float
    x2: float
    y2: float
    label: int
    kind: str = "manual"  # "manual" | "auto" | "prompt"

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.as_xyxy()
        return (x1 <= x <= x2) and (y1 <= y <= y2)


@dataclass
class ImageItem:
    path: str
    boxes: List[Box] = field(default_factory=list)


def list_images_in_folder(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                out.append(os.path.join(root, f))
    out.sort()
    return out


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def yolo_xyxy_to_yolo_txt(x1, y1, x2, y2, w, h):
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return cx, cy, bw, bh


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def rotate_box_90_cw_xyxy(box: Box, w: int, h: int) -> Box:
    # Rotate image 90 CW: (x,y)->(h-1-y, x) in pixel coords
    x1, y1, x2, y2 = box.as_xyxy()
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    rot = np.zeros_like(pts)
    rot[:, 0] = (h - 1) - pts[:, 1]
    rot[:, 1] = pts[:, 0]
    nx1, ny1 = rot.min(axis=0)
    nx2, ny2 = rot.max(axis=0)
    return Box(nx1, ny1, nx2, ny2, box.label, kind=box.kind)


def rotate_box_180_xyxy(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box.as_xyxy()
    nx1 = (w - 1) - x2
    nx2 = (w - 1) - x1
    ny1 = (h - 1) - y2
    ny2 = (h - 1) - y1
    return Box(nx1, ny1, nx2, ny2, box.label, kind=box.kind)


def rotate_box_270_cw_xyxy(box: Box, w: int, h: int) -> Box:
    # 270 CW == 90 CCW: (x,y)->(y, w-1-x)
    x1, y1, x2, y2 = box.as_xyxy()
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    rot = np.zeros_like(pts)
    rot[:, 0] = pts[:, 1]
    rot[:, 1] = (w - 1) - pts[:, 0]
    nx1, ny1 = rot.min(axis=0)
    nx2, ny2 = rot.max(axis=0)
    return Box(nx1, ny1, nx2, ny2, box.label, kind=box.kind)


def flip_h_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box.as_xyxy()
    nx1 = (w - 1) - x2
    nx2 = (w - 1) - x1
    return Box(nx1, y1, nx2, y2, box.label, kind=box.kind)


def flip_v_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box.as_xyxy()
    ny1 = (h - 1) - y2
    ny2 = (h - 1) - y1
    return Box(x1, ny1, x2, ny2, box.label, kind=box.kind)


class ZoomCanvas(ttk.Frame):
    """Image viewer with scalable redraw + bbox overlay in image coords."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.canvas = tk.Canvas(self, background="#1e1e1e", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._img_pil: Optional[Image.Image] = None
        self._img_tk: Optional[ImageTk.PhotoImage] = None
        self._img_id = None

        # view state
        self.scale = 1.0
        self._fit_scale = 1.0
        self._pad = 20

        # overlay
        self.boxes: List[Box] = []
        self.temp_box: Optional[Tuple[float, float, float, float, str]] = None  # (x1,y1,x2,y2, kind)
        self.hover_idx: Optional[int] = None

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self._on_wheel)       # Windows/macOS
        self.canvas.bind("<Button-4>", self._on_wheel_linux)   # Linux
        self.canvas.bind("<Button-5>", self._on_wheel_linux)

    def set_image(self, pil_img: Optional[Image.Image]):
        self._img_pil = pil_img
        self.scale = 1.0
        self._fit_scale = 1.0
        self._img_tk = None
        self._img_id = None
        self.canvas.delete("all")
        self.redraw()

    def set_boxes(self, boxes: List[Box]):
        self.boxes = boxes
        self.redraw()

    def set_temp_box(self, xyxy_kind: Optional[Tuple[float, float, float, float, str]]):
        self.temp_box = xyxy_kind
        self.redraw()

    def set_hover_idx(self, idx: Optional[int]):
        self.hover_idx = idx
        self.redraw()

    def _image_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        # image->canvas coords with padding
        return x * self.scale + self._pad, y * self.scale + self._pad

    def _canvas_to_image(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self._pad) / self.scale, (y - self._pad) / self.scale

    def _compute_fit_scale(self):
        if not self._img_pil:
            self._fit_scale = 1.0
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self._img_pil.size
        sw = (cw - 2 * self._pad) / iw
        sh = (ch - 2 * self._pad) / ih
        self._fit_scale = max(0.05, min(sw, sh))

    def fit_to_window(self):
        self._compute_fit_scale()
        self.scale = self._fit_scale
        self.redraw()

    def _on_wheel(self, event):
        # zoom at cursor
        if not self._img_pil:
            return
        delta = event.delta
        factor = 1.1 if delta > 0 else 1 / 1.1
        self._zoom(factor, event.x, event.y)

    def _on_wheel_linux(self, event):
        if not self._img_pil:
            return
        factor = 1.1 if event.num == 4 else 1 / 1.1
        self._zoom(factor, event.x, event.y)

    def _zoom(self, factor: float, cx: float, cy: float):
        old_scale = self.scale
        new_scale = clamp(old_scale * factor, 0.05, 20.0)
        if abs(new_scale - old_scale) < 1e-6:
            return

        # Keep point under cursor stable (in canvas scroll coords)
        x0 = self.canvas.canvasx(cx)
        y0 = self.canvas.canvasy(cy)
        ix, iy = self._canvas_to_image(x0, y0)

        self.scale = new_scale
        self.redraw()

        nx, ny = self._image_to_canvas(ix, iy)
        self.canvas.xview_moveto(max(0, (nx - cx) / max(1, self.canvas.bbox("all")[2])))
        self.canvas.yview_moveto(max(0, (ny - cy) / max(1, self.canvas.bbox("all")[3])))

    def redraw(self):
        self.canvas.delete("all")
        if not self._img_pil:
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            return

        # If first draw, fit
        if self._img_tk is None:
            self._compute_fit_scale()
            self.scale = self._fit_scale

        iw, ih = self._img_pil.size
        dw, dh = int(iw * self.scale), int(ih * self.scale)
        img_resized = self._img_pil.resize((max(1, dw), max(1, dh)), Image.Resampling.BILINEAR)
        self._img_tk = ImageTk.PhotoImage(img_resized)

        self._img_id = self.canvas.create_image(self._pad, self._pad, anchor="nw", image=self._img_tk)

        # Draw boxes
        for i, b in enumerate(self.boxes):
            x1, y1, x2, y2 = b.as_xyxy()
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)

            if b.kind == "prompt":
                color = "#ffcc00"
            elif b.kind == "auto":
                color = "#4fc3f7"
            else:
                color = "#66bb6a"

            width = 3 if (self.hover_idx == i) else 2
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width)
            self.canvas.create_text(
                cx1 + 2, cy1 - 10, anchor="nw",
                fill=color, text=f"{b.label} ({b.kind})", font=("Segoe UI", 10, "bold")
            )

        # Draw temp box
        if self.temp_box is not None:
            x1, y1, x2, y2, kind = self.temp_box
            cx1, cy1 = self._image_to_canvas(min(x1, x2), min(y1, y2))
            cx2, cy2 = self._image_to_canvas(max(x1, x2), max(y1, y2))
            color = "#ffcc00" if kind == "prompt" else "#ffffff"
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=2, dash=(5, 3))
            self.canvas.create_text(cx1 + 2, cy1 - 10, anchor="nw", fill=color,
                                    text=f"pending {kind}", font=("Segoe UI", 10, "bold"))

        # Scroll region
        self.canvas.configure(scrollregion=(0, 0, dw + 2 * self._pad, dh + 2 * self._pad))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLOE Dataset Labeler (Tkinter)")
        self.geometry("1280x800")
        self.minsize(980, 600)

        self.images: List[ImageItem] = []
        self.cur_index: int = -1

        self._pil_cache: Dict[str, Image.Image] = {}
        self._pending_rect: Optional[Tuple[float, float, float, float, str]] = None  # image coords + kind
        self._mode = tk.StringVar(value="draw")  # draw|prompt|delete|pan
        self._status = tk.StringVar(value="Ready")

        # camera
        self._cam_running = False
        self._cam_thread = None

        # model
        self._yoloe_model = None
        self._autolabel_thread = None
        self._stop_autolabel = False

        # notebook screens
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)

        self.screen_import = ttk.Frame(self.nb)
        self.screen_label = ttk.Frame(self.nb)
        self.screen_output = ttk.Frame(self.nb)
        self.screen_training = ttk.Frame(self.nb)
        self.screen_validation = ttk.Frame(self.nb)

        self.nb.add(self.screen_import, text="1) Import")
        self.nb.add(self.screen_label, text="2) Labeling")
        self.nb.add(self.screen_output, text="3) Output")
        self.nb.add(self.screen_training, text="4) Training")
        self.nb.add(self.screen_validation, text="5) Validation")

        self._build_import()
        self._build_labeling()
        self._build_output()
        self._build_training()
        self._build_validation()

        # status bar
        sb = ttk.Frame(self)
        sb.pack(fill="x", side="bottom")
        ttk.Label(sb, textvariable=self._status).pack(side="left", padx=8, pady=4)

        self.bind_all("<KeyPress>", self._on_keypress)

    # ----------------------------
    # Screen 1: Import
    # ----------------------------
    def _build_import(self):
        frm = self.screen_import
        frm.columnconfigure(0, weight=1)

        top = ttk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Import images", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")

        btns = ttk.Frame(top)
        btns.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(btns, text="Add folder…", command=self._import_folder).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Add files…", command=self._import_files).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(btns, text="Clear list", command=self._clear_images).grid(row=0, column=2)

        cam = ttk.Labelframe(frm, text="Video frame extraction (optional)")
        cam.grid(row=1, column=0, sticky="ew", padx=12, pady=12)
        for i in range(6):
            cam.columnconfigure(i, weight=(1 if i == 5 else 0))

        ttk.Label(cam, text="Save to folder:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.cam_out_var = tk.StringVar(value=os.path.abspath("./captured"))
        ttk.Entry(cam, textvariable=self.cam_out_var).grid(row=0, column=1, columnspan=4, sticky="ew", padx=8, pady=8)
        ttk.Button(cam, text="Browse…", command=self._pick_cam_folder).grid(row=0, column=5, sticky="e", padx=8, pady=8)

        ttk.Label(cam, text="Interval (sec):").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.cam_interval_var = tk.DoubleVar(value=2.0)
        ttk.Entry(cam, textvariable=self.cam_interval_var, width=10).grid(row=1, column=1, sticky="w", padx=8, pady=8)

        ttk.Button(cam, text="Upload video…", command=self._upload_video_for_extraction).grid(row=1, column=2, padx=8, pady=8, sticky="e")
        ttk.Button(cam, text="Extract frames", command=self._extract_video_frames).grid(row=1, column=3, padx=8, pady=8, sticky="e")
        
        self.video_path_var = tk.StringVar(value="")
        ttk.Label(cam, text="Video:").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        ttk.Label(cam, textvariable=self.video_path_var, foreground="gray").grid(row=2, column=1, columnspan=4, sticky="w", padx=8, pady=8)

        # imported list preview
        mid = ttk.Frame(frm)
        mid.grid(row=2, column=0, sticky="nsew", padx=12, pady=12)
        frm.rowconfigure(2, weight=1)
        mid.columnconfigure(0, weight=1)
        mid.rowconfigure(0, weight=1)

        self.import_list = tk.Listbox(mid)
        ysb = ttk.Scrollbar(mid, orient="vertical", command=self.import_list.yview)
        self.import_list.configure(yscrollcommand=ysb.set)
        self.import_list.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")

        bottom = ttk.Frame(frm)
        bottom.grid(row=3, column=0, sticky="ew", padx=12, pady=12)
        bottom.columnconfigure(0, weight=1)
        ttk.Button(bottom, text="Go to Labeling →", command=lambda: self.nb.select(self.screen_label)).pack(side="right")

    def _pick_cam_folder(self):
        d = filedialog.askdirectory(title="Select capture output folder")
        if d:
            self.cam_out_var.set(d)

    def _upload_video_for_extraction(self):
        if cv2 is None:
            messagebox.showerror("OpenCV missing", "Install opencv-python to use video extraction.")
            return
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
        )
        if path:
            self.video_path_var.set(path)

    def _extract_video_frames(self):
        if cv2 is None:
            messagebox.showerror("OpenCV missing", "Install opencv-python to use video extraction.")
            return
        video_path = self.video_path_var.get().strip()
        if not video_path or not os.path.exists(video_path):
            messagebox.showwarning("No video", "Please upload a video file first.")
            return
        
        out_dir = self.cam_out_var.get().strip()
        ensure_dir(out_dir)
        interval = float(self.cam_interval_var.get())
        
        if interval <= 0:
            messagebox.showerror("Invalid interval", "Interval must be greater than 0.")
            return
        
        def extract_worker():
            try:
                self._status.set("Extracting frames from video...")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.after(0, lambda: messagebox.showerror("Error", "Failed to open video file."))
                    return
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps * interval)
                frame_count = 0
                saved_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        fn = os.path.join(out_dir, f"frame_{ts}_{saved_count:06d}.jpg")
                        cv2.imwrite(fn, frame)
                        self.after(0, lambda p=fn: self._add_images([p]))
                        saved_count += 1
                    
                    frame_count += 1
                
                cap.release()
                self.after(0, lambda: self._status.set(f"Extracted {saved_count} frames from video."))
                self.after(0, lambda: messagebox.showinfo("Complete", f"Extracted {saved_count} frames to:\n{out_dir}"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Failed to extract frames: {str(e)}"))
                self.after(0, lambda: self._status.set("Frame extraction failed."))
        
        thread = threading.Thread(target=extract_worker, daemon=True)
        thread.start()

    def _import_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        paths = list_images_in_folder(folder)
        self._add_images(paths)

    def _import_files(self):
        paths = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]
        )
        if not paths:
            return
        self._add_images(list(paths))

    def _clear_images(self):
        self.images.clear()
        self.cur_index = -1
        self._pil_cache.clear()
        self._refresh_lists()
        self.viewer.set_image(None)
        self._status.set("Cleared.")


    # ----------------------------
    # Screen 2: Labeling
    # ----------------------------
    def _build_labeling(self):
        frm = self.screen_label
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        # Paned layout
        pan = ttk.Panedwindow(frm, orient="horizontal")
        pan.grid(row=0, column=0, sticky="nsew")

        left = ttk.Frame(pan, width=260)
        right = ttk.Frame(pan)
        pan.add(left, weight=0)
        pan.add(right, weight=1)

        # Left: image list + tools
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Images", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))

        nav = ttk.Frame(left)
        nav.grid(row=1, column=0, sticky="ew", padx=8)
        ttk.Button(nav, text="Prev", command=self._prev_image).pack(side="left")
        ttk.Button(nav, text="Next", command=self._next_image).pack(side="left", padx=6)
        ttk.Button(nav, text="Fit", command=lambda: self.viewer.fit_to_window()).pack(side="right")

        self.img_list = tk.Listbox(left, exportselection=False)
        self.img_list.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        lsb = ttk.Scrollbar(left, orient="vertical", command=self.img_list.yview)
        self.img_list.configure(yscrollcommand=lsb.set)
        lsb.grid(row=2, column=1, sticky="ns", pady=8)
        self.img_list.bind("<<ListboxSelect>>", self._on_list_select)

        # Right: toolbar + viewer
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tb = ttk.Frame(right)
        tb.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for i in range(10):
            tb.columnconfigure(i, weight=0)
        tb.columnconfigure(9, weight=1)

        ttk.Label(tb, text="Mode:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(tb, text="Draw", value="draw", variable=self._mode).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Radiobutton(tb, text="Prompt", value="prompt", variable=self._mode).grid(row=0, column=2, sticky="w", padx=6)
        ttk.Radiobutton(tb, text="Delete", value="delete", variable=self._mode).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Radiobutton(tb, text="Pan", value="pan", variable=self._mode).grid(row=0, column=4, sticky="w", padx=6)

        ttk.Label(tb, text="Conf:").grid(row=0, column=5, sticky="e")
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        ttk.Entry(tb, textvariable=self.conf_var, width=6).grid(row=0, column=6, sticky="w", padx=(4, 10))

        ttk.Label(tb, text="IoU:").grid(row=0, column=7, sticky="e")
        self.iou_var = tk.DoubleVar(value=DEFAULT_IOU)
        ttk.Entry(tb, textvariable=self.iou_var, width=6).grid(row=0, column=8, sticky="w", padx=(4, 10))

        ttk.Button(tb, text="Auto label all", command=self._auto_label_all).grid(row=0, column=9, sticky="e")

        hint = ttk.Label(
            right,
            text=("Draw: drag box, then press a number key (0-9) to commit.\n"
                  "Prompt: drag box around one example object, then press a number key (0-9) to commit as prompt.\n"
                  "Delete: hover box (it highlights), then left-click to delete."),
        )
        hint.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.viewer = ZoomCanvas(right)
        self.viewer.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        # Viewer interactions
        self._drag_start = None
        self._pan_start = None

        self.viewer.canvas.bind("<Button-1>", self._on_mouse_down)
        self.viewer.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.viewer.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.viewer.canvas.bind("<Motion>", self._on_hover_move)

    def _on_mouse_down(self, event):
        if self.cur_index < 0 or self.cur_index >= len(self.images):
            return

        mode = self._mode.get()
        cx = self.viewer.canvas.canvasx(event.x)
        cy = self.viewer.canvas.canvasy(event.y)

        if mode == "pan":
            self._pan_start = (cx, cy, self.viewer.canvas.xview(), self.viewer.canvas.yview())
            return

        if mode in ("draw", "prompt"):
            ix, iy = self.viewer._canvas_to_image(cx, cy)
            self._drag_start = (ix, iy)
            self._pending_rect = (ix, iy, ix, iy, mode)
            self.viewer.set_temp_box(self._pending_rect)

        elif mode == "delete":
            self._delete_hovered_box()

    def _on_mouse_move(self, event):
        if self.cur_index < 0 or self.cur_index >= len(self.images):
            return

        mode = self._mode.get()
        cx = self.viewer.canvas.canvasx(event.x)
        cy = self.viewer.canvas.canvasy(event.y)

        if mode == "pan" and self._pan_start is not None:
            sx, sy, xv, yv = self._pan_start
            dx = sx - cx
            dy = sy - cy
            self.viewer.canvas.xview_moveto(clamp(xv[0] + dx / max(1, self.viewer.canvas.bbox("all")[2]), 0, 1))
            self.viewer.canvas.yview_moveto(clamp(yv[0] + dy / max(1, self.viewer.canvas.bbox("all")[3]), 0, 1))
            return

        if mode in ("draw", "prompt") and self._drag_start is not None:
            ix, iy = self.viewer._canvas_to_image(cx, cy)
            x0, y0 = self._drag_start
            self._pending_rect = (x0, y0, ix, iy, mode)
            self.viewer.set_temp_box(self._pending_rect)

    def _on_mouse_up(self, event):
        if self._mode.get() == "pan":
            self._pan_start = None
            return

        if self._drag_start is not None:
            self._drag_start = None
            # Keep pending rect until number key commits (per request)

    def _on_hover_move(self, event):
        if self.cur_index < 0:
            return
        if self._mode.get() != "delete":
            self.viewer.set_hover_idx(None)
            return
        cx = self.viewer.canvas.canvasx(event.x)
        cy = self.viewer.canvas.canvasy(event.y)
        ix, iy = self.viewer._canvas_to_image(cx, cy)

        boxes = self.images[self.cur_index].boxes
        hovered = None
        for i in reversed(range(len(boxes))):
            if boxes[i].contains(ix, iy):
                hovered = i
                break
        self.viewer.set_hover_idx(hovered)

    def _delete_hovered_box(self):
        if self.cur_index < 0:
            return
        idx = self.viewer.hover_idx
        if idx is None:
            return
        del self.images[self.cur_index].boxes[idx]
        self.viewer.set_hover_idx(None)
        self._refresh_viewer()

    def _on_keypress(self, event):
        if self.nb.index("current") != 1:
            return
        if self.cur_index < 0:
            return

        # Commit by pressing a single digit key (0-9) after drawing
        if event.char.isdigit() and self._pending_rect is not None:
            label = int(event.char)
            x1, y1, x2, y2, mode = self._pending_rect

            img = self._get_current_pil()
            if img is None:
                return
            iw, ih = img.size
            # clamp in image
            x1 = clamp(x1, 0, iw - 1)
            x2 = clamp(x2, 0, iw - 1)
            y1 = clamp(y1, 0, ih - 1)
            y2 = clamp(y2, 0, ih - 1)

            if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
                self._status.set("Box too small; ignored.")
                self._pending_rect = None
                self.viewer.set_temp_box(None)
                return

            kind = "prompt" if mode == "prompt" else "manual"
            self.images[self.cur_index].boxes.append(Box(x1, y1, x2, y2, label=label, kind=kind))

            self._pending_rect = None
            self.viewer.set_temp_box(None)
            self._refresh_viewer()
            self._status.set(f"Added {kind} box label={label}.")

        elif event.keysym == "Escape":
            self._pending_rect = None
            self.viewer.set_temp_box(None)
            self._status.set("Canceled pending box.")

        elif event.keysym in ("Left", "Prior"):
            self._prev_image()
        elif event.keysym in ("Right", "Next"):
            self._next_image()

        elif event.keysym == "Delete" and self._mode.get() == "delete":
            self._delete_hovered_box()

    def _prev_image(self):
        if not self.images:
            return
        self.cur_index = max(0, self.cur_index - 1)
        self._select_index(self.cur_index)

    def _next_image(self):
        if not self.images:
            return
        self.cur_index = min(len(self.images) - 1, self.cur_index + 1)
        self._select_index(self.cur_index)

    def _on_list_select(self, _evt):
        sel = self.img_list.curselection()
        if not sel:
            return
        self.cur_index = int(sel[0])
        self._load_current_image()

    def _select_index(self, idx: int):
        self.img_list.selection_clear(0, "end")
        self.img_list.selection_set(idx)
        self.img_list.activate(idx)
        self.img_list.see(idx)
        self.cur_index = idx
        self._load_current_image()

    def _get_current_pil(self) -> Optional[Image.Image]:
        if self.cur_index < 0 or self.cur_index >= len(self.images):
            return None
        p = self.images[self.cur_index].path
        if p not in self._pil_cache:
            self._pil_cache[p] = Image.open(p).convert("RGB")
        return self._pil_cache[p]

    def _load_current_image(self):
        img = self._get_current_pil()
        self.viewer.set_image(img)
        self._refresh_viewer()
        self._status.set(f"Viewing {os.path.basename(self.images[self.cur_index].path)}")

    def _refresh_viewer(self):
        if self.cur_index < 0:
            self.viewer.set_boxes([])
            return
        self.viewer.set_boxes(self.images[self.cur_index].boxes)

    def _auto_label_all(self):
        if YOLOE is None or YOLOEVPSegPredictor is None:
            messagebox.showerror("Ultralytics missing", "Install ultralytics and ensure YOLOE is available.")
            return
        if not self.images:
            messagebox.showwarning("No images", "Import images first.")
            return

        # Find prompt boxes on CURRENT image only (simple workflow)
        if self.cur_index < 0:
            messagebox.showwarning("No current image", "Select an image and draw a prompt box.")
            return

        prompt_img_item = self.images[self.cur_index]
        prompt_boxes = [b for b in prompt_img_item.boxes if b.kind == "prompt"]
        if not prompt_boxes:
            messagebox.showwarning("No prompt box", "Use Prompt mode, draw a prompt box, then press a number key.")
            return

        # Remap user numeric labels -> sequential class ids 0..K-1 as required by YOLOE visual prompts
        labels = sorted({b.label for b in prompt_boxes})
        label_to_seq = {lab: i for i, lab in enumerate(labels)}
        seq_to_label = {i: lab for lab, i in label_to_seq.items()}

        bboxes = []
        cls = []
        for b in prompt_boxes:
            x1, y1, x2, y2 = b.as_xyxy()
            bboxes.append([x1, y1, x2, y2])
            cls.append(label_to_seq[b.label])

        visual_prompts = dict(
            bboxes=np.array(bboxes, dtype=np.float32),
            cls=np.array(cls, dtype=np.int64),
        )

        refer_image = prompt_img_item.path
        conf = float(self.conf_var.get())
        iou = float(self.iou_var.get())

        if self._autolabel_thread and self._autolabel_thread.is_alive():
            messagebox.showinfo("Busy", "Auto-label is already running.")
            return

        self._stop_autolabel = False

        def worker():
            try:
                # Determine device (GPU if available)
                device = None
                device_name = "CPU"
                gpu_info = ""
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                        device_name = f"GPU ({gpu_name}, {gpu_memory:.1f}GB)"
                        gpu_info = f"\nGPU: {gpu_name}\nGPU Memory: {gpu_memory:.1f}GB\nCUDA Version: {torch.version.cuda}"
                    else:
                        device = "cpu"
                        device_name = "CPU"
                        # Check why CUDA is not available
                        if not torch.cuda.is_available():
                            gpu_info = "\nCUDA is not available. Possible reasons:\n"
                            gpu_info += "- PyTorch was installed without CUDA support\n"
                            gpu_info += "- CUDA drivers are not installed\n"
                            gpu_info += "- GPU is not detected by PyTorch\n"
                            gpu_info += f"PyTorch version: {torch.__version__}"
                except ImportError:
                    device = "cpu"
                    device_name = "CPU"
                    gpu_info = "\nPyTorch not found. Install torch to enable GPU support."
                
                self._status.set(f"Loading YOLOE model on {device_name}...")
                if self._yoloe_model is None:
                    # Try to initialize YOLOE with device parameter
                    try:
                        # Ultralytics YOLO models typically accept device during initialization
                        self._yoloe_model = YOLOE(MODEL_PATH, device=device)
                    except TypeError:
                        # If device parameter not supported, initialize normally and try to set device
                        self._yoloe_model = YOLOE(MODEL_PATH)
                        # Try to set device after initialization
                        if device == "cuda":
                            try:
                                if hasattr(self._yoloe_model, 'to'):
                                    self._yoloe_model.to(device)
                                elif hasattr(self._yoloe_model, 'model'):
                                    # Some models have a nested model attribute
                                    if hasattr(self._yoloe_model.model, 'to'):
                                        self._yoloe_model.model.to(device)
                            except Exception as e:
                                # If setting device fails, log but continue
                                self.after(0, lambda: self._status.set(f"Warning: Could not set device to GPU: {str(e)}"))
                
                # Log which device is being used
                self.after(0, lambda: self._status.set(f"Using {device_name} for autolabeling..."))

                # Get the set of labels we're currently processing
                current_prompt_labels = set(labels)

                # Remove only auto boxes that match the current prompt labels (not all auto boxes)
                # This preserves manual boxes and auto boxes from other label sets
                for item in self.images:
                    item.boxes = [b for b in item.boxes if not (b.kind == "auto" and b.label in current_prompt_labels)]

                # Batch processing for speed
                batch_size = 8  # Process 8 images at a time
                total_images = len(self.images)
                
                # Process images in batches
                for batch_start in range(0, total_images, batch_size):
                    if self._stop_autolabel:
                        break
                    
                    batch_end = min(batch_start + batch_size, total_images)
                    batch_items = self.images[batch_start:batch_end]
                    batch_paths = [item.path for item in batch_items]
                    
                    num_batches = (total_images + batch_size - 1) // batch_size
                    current_batch = batch_start // batch_size + 1
                    self._status.set(f"Auto-labeling batch {current_batch}/{num_batches} ({batch_start+1}-{batch_end}/{total_images})...")
                    
                    # Process batch - YOLOE can handle lists of images
                    # Device is already set during model initialization, so we don't need to pass it to predict
                    try:
                        results = self._yoloe_model.predict(
                            batch_paths,
                            refer_image=refer_image,
                            visual_prompts=visual_prompts,
                            predictor=YOLOEVPSegPredictor,
                            conf=conf,
                            iou=iou,
                            verbose=False,
                        )
                    except Exception as e:
                        # Fallback to individual processing if batch fails
                        self.after(0, lambda: self._status.set(f"Batch processing failed, using individual processing..."))
                        results = []
                        for path in batch_paths:
                            result = self._yoloe_model.predict(
                                path,
                                refer_image=refer_image,
                                visual_prompts=visual_prompts,
                                predictor=YOLOEVPSegPredictor,
                                conf=conf,
                                iou=iou,
                                verbose=False,
                            )
                            results.extend(result)

                    # Process results for each image in the batch
                    for i, (item, result) in enumerate(zip(batch_items, results)):
                        if not hasattr(result, "boxes") or result.boxes is None:
                            continue

                        # Get boxes from result
                        if hasattr(result.boxes, 'xyxy'):
                            xyxy = result.boxes.xyxy
                            if hasattr(xyxy, 'cpu'):
                                xyxy = xyxy.cpu().numpy()
                            else:
                                xyxy = np.array(xyxy)
                        else:
                            continue

                        if hasattr(result.boxes, 'cls'):
                            c = result.boxes.cls
                            if hasattr(c, 'cpu'):
                                c = c.cpu().numpy().astype(int)
                            else:
                                c = np.array(c).astype(int)
                        else:
                            continue

                        # Add new auto boxes
                        for (x1, y1, x2, y2), cid in zip(xyxy, c):
                            lab = seq_to_label.get(int(cid), int(cid))
                            item.boxes.append(Box(float(x1), float(y1), float(x2), float(y2), label=lab, kind="auto"))

                        # Refresh current display if this is the current image
                        idx = batch_start + i
                        if self.cur_index == idx:
                            self.after(0, self._refresh_viewer)

                # Show completion alert
                def show_completion():
                    self._status.set("Auto-label done.")
                    # Bring window to front
                    self.lift()
                    self.attributes('-topmost', True)
                    self.after_idle(lambda: self.attributes('-topmost', False))
                    # Show messagebox with device info
                    msg = f"Autolabeling finished successfully!\n\n"
                    msg += f"Processed {total_images} images using {device_name}."
                    if device == "cpu" and gpu_info:
                        msg += f"\n\n{gpu_info}"
                    messagebox.showinfo("Autolabeling Complete", msg)
                
                self.after(0, show_completion)
            except Exception as e:
                def show_error():
                    self._status.set("Auto-label failed.")
                    # Bring window to front
                    self.lift()
                    self.attributes('-topmost', True)
                    self.after_idle(lambda: self.attributes('-topmost', False))
                    messagebox.showerror("Auto-label error", str(e))
                self.after(0, show_error)

        self._autolabel_thread = threading.Thread(target=worker, daemon=True)
        self._autolabel_thread.start()

    # ----------------------------
    # Screen 3: Output
    # ----------------------------
    def _build_output(self):
        frm = self.screen_output
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(2, weight=1)

        top = ttk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Output dataset", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")

        out = ttk.Labelframe(frm, text="Export settings")
        out.grid(row=1, column=0, sticky="ew", padx=12, pady=12)
        for i in range(6):
            out.columnconfigure(i, weight=(1 if i == 3 else 0))

        ttk.Label(out, text="Output folder:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.out_dir_var = tk.StringVar(value=os.path.abspath("./dataset_out"))
        ttk.Entry(out, textvariable=self.out_dir_var).grid(row=0, column=1, columnspan=3, sticky="ew", padx=8, pady=8)
        ttk.Button(out, text="Browse…", command=self._pick_out_folder).grid(row=0, column=4, padx=8, pady=8)
        ttk.Button(out, text="Export", command=self._export_dataset).grid(row=0, column=5, padx=8, pady=8)

        ttk.Label(out, text="Format:").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.format_var = tk.StringVar(value="yolo")
        ttk.Radiobutton(out, text="YOLO .txt", value="yolo", variable=self.format_var).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Radiobutton(out, text="COCO .json", value="coco", variable=self.format_var).grid(row=1, column=2, sticky="w", padx=8)
        ttk.Radiobutton(out, text="Pascal VOC .xml", value="voc", variable=self.format_var).grid(row=1, column=3, sticky="w", padx=8)

        aug = ttk.Labelframe(frm, text="Albumentations augmentations")
        aug.grid(row=2, column=0, sticky="nsew", padx=12, pady=12)
        aug.columnconfigure(0, weight=1)

        row = ttk.Frame(aug)
        row.pack(fill="x", padx=8, pady=8)

        self.albu_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Enable Albumentations", variable=self.albu_enable).grid(row=0, column=0, sticky="w")

        ttk.Label(row, text="Num augmented copies per image:").grid(row=0, column=1, sticky="e", padx=(12, 6))
        self.albu_copies = tk.IntVar(value=2)
        ttk.Entry(row, textvariable=self.albu_copies, width=6).grid(row=0, column=2, sticky="w")

        # Common transforms
        grid = ttk.Frame(aug)
        grid.pack(fill="x", padx=8, pady=8)

        self.albu_geom = tk.BooleanVar(value=True)
        self.albu_color = tk.BooleanVar(value=True)
        self.albu_noise_blur = tk.BooleanVar(value=True)
        self.albu_dropout = tk.BooleanVar(value=False)
        self.albu_safe_crop = tk.BooleanVar(value=True)

        ttk.Checkbutton(grid, text="Geometric (flip/rotate/affine/perspective)", variable=self.albu_geom).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Color (HSV/brightness/contrast/gamma/CLAHE)", variable=self.albu_color).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Noise/blur/compression", variable=self.albu_noise_blur).grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Dropout (coarse dropout/cutout-like)", variable=self.albu_dropout).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Safe crop (bbox aware)", variable=self.albu_safe_crop).grid(row=2, column=0, sticky="w", padx=6, pady=4)

        # Mosaic
        mrow = ttk.Frame(aug)
        mrow.pack(fill="x", padx=8, pady=(8, 4))

        self.albu_mosaic = tk.BooleanVar(value=False)
        ttk.Checkbutton(mrow, text="Enable Mosaic", variable=self.albu_mosaic).pack(side="left")

        ttk.Label(mrow, text="Mosaic prob:").pack(side="left", padx=(12, 6))
        self.albu_mosaic_p = tk.DoubleVar(value=0.3)
        ttk.Entry(mrow, textvariable=self.albu_mosaic_p, width=6).pack(side="left")

        ttk.Label(mrow, text="Grid (y,x):").pack(side="left", padx=(12, 6))
        self.albu_mosaic_gy = tk.IntVar(value=2)
        self.albu_mosaic_gx = tk.IntVar(value=2)
        ttk.Entry(mrow, textvariable=self.albu_mosaic_gy, width=4).pack(side="left")
        ttk.Entry(mrow, textvariable=self.albu_mosaic_gx, width=4).pack(side="left", padx=(4, 0))

        ttk.Label(mrow, text="Target size:").pack(side="left", padx=(12, 6))
        self.albu_mosaic_ts = tk.IntVar(value=1024)
        ttk.Entry(mrow, textvariable=self.albu_mosaic_ts, width=6).pack(side="left")

    # ----------------------------
    # Screen 4: Training
    # ----------------------------
    def _build_training(self):
        frm = self.screen_training
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(1, weight=1)

        top = ttk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="RF-DETR Model Training", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")

        settings = ttk.Labelframe(frm, text="Training Settings")
        settings.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        settings.columnconfigure(1, weight=1)

        # Dataset path
        ttk.Label(settings, text="Dataset path:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.train_dataset_var = tk.StringVar(value="")
        ttk.Entry(settings, textvariable=self.train_dataset_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(settings, text="Browse…", command=self._pick_train_dataset).grid(row=0, column=2, padx=8, pady=8)

        # Output directory
        ttk.Label(settings, text="Output directory:").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.train_output_var = tk.StringVar(value=os.path.abspath("./models"))
        ttk.Entry(settings, textvariable=self.train_output_var).grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(settings, text="Browse…", command=self._pick_train_output).grid(row=1, column=2, padx=8, pady=8)

        # Engine output filename
        ttk.Label(settings, text="Engine filename:").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        self.train_engine_var = tk.StringVar(value="trained_rfdetr.engine")
        ttk.Entry(settings, textvariable=self.train_engine_var).grid(row=2, column=1, sticky="ew", padx=8, pady=8)

        # Training parameters
        params = ttk.Labelframe(settings, text="Training Parameters")
        params.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        params.columnconfigure(1, weight=1)
        params.columnconfigure(3, weight=1)

        ttk.Label(params, text="Epochs:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.train_epochs_var = tk.IntVar(value=100)
        ttk.Entry(params, textvariable=self.train_epochs_var, width=10).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(params, text="Batch size:").grid(row=0, column=2, sticky="w", padx=8, pady=4)
        self.train_batch_var = tk.IntVar(value=4)
        ttk.Entry(params, textvariable=self.train_batch_var, width=10).grid(row=0, column=3, sticky="w", padx=8, pady=4)

        ttk.Label(params, text="Learning rate:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.train_lr_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(params, textvariable=self.train_lr_var, width=10).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(params, text="Grad accum steps:").grid(row=1, column=2, sticky="w", padx=8, pady=4)
        self.train_grad_accum_var = tk.IntVar(value=4)
        ttk.Entry(params, textvariable=self.train_grad_accum_var, width=10).grid(row=1, column=3, sticky="w", padx=8, pady=4)

        ttk.Label(params, text="Workspace size (MB):").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.train_workspace_var = tk.IntVar(value=4096)
        ttk.Entry(params, textvariable=self.train_workspace_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=4)

        # Training status
        status_frame = ttk.Frame(settings)
        status_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        status_frame.columnconfigure(0, weight=1)

        self.train_status_var = tk.StringVar(value="Ready to train")
        ttk.Label(status_frame, textvariable=self.train_status_var, foreground="gray").grid(row=0, column=0, sticky="w")

        # Progress bar
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.train_progress = ttk.Progressbar(status_frame, variable=self.train_progress_var, maximum=100.0, length=300)
        self.train_progress.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.train_progress.grid_remove()  # Hide initially

        # Start training button
        ttk.Button(settings, text="Start Training", command=self._start_training).grid(row=5, column=0, columnspan=3, padx=8, pady=12)

        self._training_thread = None
        self._stop_training = False

    def _pick_train_dataset(self):
        d = filedialog.askdirectory(title="Select dataset folder (COCO format)")
        if d:
            self.train_dataset_var.set(d)

    def _pick_train_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.train_output_var.set(d)

    def _start_training(self):
        if rfdetr_train is None:
            messagebox.showerror("Training module missing", "Could not import training module.")
            return
        
        dataset_path = self.train_dataset_var.get().strip()
        if not dataset_path or not os.path.exists(dataset_path):
            messagebox.showwarning("Invalid dataset", "Please select a valid dataset folder.")
            return

        output_dir = self.train_output_var.get().strip()
        engine_output = self.train_engine_var.get().strip()
        if not engine_output:
            messagebox.showwarning("Invalid engine name", "Please specify an engine filename.")
            return

        epochs = int(self.train_epochs_var.get())
        batch_size = int(self.train_batch_var.get())
        lr = float(self.train_lr_var.get())
        grad_accum = int(self.train_grad_accum_var.get())
        workspace_size = int(self.train_workspace_var.get())

        if self._training_thread and self._training_thread.is_alive():
            messagebox.showinfo("Training in progress", "Training is already running.")
            return

        self._stop_training = False

        def training_worker():
            try:
                # Show progress bar and reset
                self.after(0, lambda: self.train_progress.grid())
                self.after(0, lambda: self.train_progress_var.set(0.0))
                self.after(0, lambda: self.train_progress.configure(mode='indeterminate'))
                self.after(0, lambda: self.train_progress.start(10))  # Start indeterminate progress
                self.after(0, lambda: self.train_status_var.set("Training started..."))
                self.after(0, lambda: self._status.set("Training RF-DETR model..."))
                
                engine_path = rfdetr_train(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    engine_output=engine_output,
                    epochs=epochs,
                    batch_size=batch_size,
                    grad_accum_steps=grad_accum,
                    lr=lr,
                    workspace_size=workspace_size
                )
                
                # Stop progress bar and hide it
                self.after(0, lambda: self.train_progress.stop())
                self.after(0, lambda: self.train_progress.grid_remove())
                self.after(0, lambda: self.train_progress_var.set(100.0))
                self.after(0, lambda: self.train_status_var.set(f"Training complete! Engine saved to: {engine_path}"))
                self.after(0, lambda: self._status.set("Training completed successfully."))
                self.after(0, lambda: messagebox.showinfo("Training Complete", f"Model trained and exported to:\n{engine_path}"))
            except Exception as e:
                # Stop progress bar on error
                error_msg = str(e)  # Capture error message for lambda
                self.after(0, lambda: self.train_progress.stop())
                self.after(0, lambda: self.train_progress.grid_remove())
                self.after(0, lambda msg=error_msg: self.train_status_var.set(f"Training failed: {msg}"))
                self.after(0, lambda: self._status.set("Training failed."))
                self.after(0, lambda msg=error_msg: messagebox.showerror("Training Error", f"An error occurred during training:\n{msg}"))

        self._training_thread = threading.Thread(target=training_worker, daemon=True)
        self._training_thread.start()

    # ----------------------------
    # Screen 5: Validation
    # ----------------------------
    def _build_validation(self):
        frm = self.screen_validation
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(1, weight=1)

        top = ttk.Frame(frm)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Model Validation", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")

        settings = ttk.Labelframe(frm, text="Validation Settings")
        settings.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        settings.columnconfigure(1, weight=1)

        # Model engine path
        ttk.Label(settings, text="Model engine file:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.val_model_var = tk.StringVar(value="")
        ttk.Entry(settings, textvariable=self.val_model_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(settings, text="Browse…", command=self._pick_val_model).grid(row=0, column=2, padx=8, pady=8)

        # Video path
        ttk.Label(settings, text="Test video:").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.val_video_var = tk.StringVar(value="")
        ttk.Entry(settings, textvariable=self.val_video_var).grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(settings, text="Browse…", command=self._pick_val_video).grid(row=1, column=2, padx=8, pady=8)

        # Output video path
        ttk.Label(settings, text="Output video:").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        self.val_output_var = tk.StringVar(value=os.path.abspath("./validation_output.mp4"))
        ttk.Entry(settings, textvariable=self.val_output_var).grid(row=2, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(settings, text="Browse…", command=self._pick_val_output).grid(row=2, column=2, padx=8, pady=8)

        # Confidence threshold
        ttk.Label(settings, text="Confidence threshold:").grid(row=3, column=0, sticky="w", padx=8, pady=8)
        self.val_conf_var = tk.DoubleVar(value=0.25)
        ttk.Entry(settings, textvariable=self.val_conf_var, width=10).grid(row=3, column=1, sticky="w", padx=8, pady=8)

        # Validation status
        status_frame = ttk.Frame(settings)
        status_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        status_frame.columnconfigure(0, weight=1)

        self.val_status_var = tk.StringVar(value="Ready to validate")
        ttk.Label(status_frame, textvariable=self.val_status_var, foreground="gray").grid(row=0, column=0, sticky="w")

        # Start validation button
        ttk.Button(settings, text="Process Video", command=self._start_validation).grid(row=5, column=0, columnspan=3, padx=8, pady=12)

        self._validation_thread = None
        self._validation_model = None

    def _pick_val_model(self):
        path = filedialog.askopenfilename(
            title="Select model engine file",
            filetypes=[("Engine files", "*.engine"), ("All files", "*.*")]
        )
        if path:
            self.val_model_var.set(path)

    def _pick_val_video(self):
        if cv2 is None:
            messagebox.showerror("OpenCV missing", "Install opencv-python to use video validation.")
            return
        path = filedialog.askopenfilename(
            title="Select test video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"), ("All files", "*.*")]
        )
        if path:
            self.val_video_var.set(path)

    def _pick_val_output(self):
        path = filedialog.asksaveasfilename(
            title="Save output video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        if path:
            self.val_output_var.set(path)

    def _start_validation(self):
        if RFDETRBase is None:
            messagebox.showerror("RFDETR missing", "Could not import RFDETR. Please ensure rfdetr is installed.")
            return
        if cv2 is None:
            messagebox.showerror("OpenCV missing", "Install opencv-python to use video validation.")
            return

        model_path = self.val_model_var.get().strip()
        video_path = self.val_video_var.get().strip()
        output_path = self.val_output_var.get().strip()

        if not model_path or not os.path.exists(model_path):
            messagebox.showwarning("Invalid model", "Please select a valid model engine file.")
            return
        if not video_path or not os.path.exists(video_path):
            messagebox.showwarning("Invalid video", "Please select a valid test video file.")
            return
        if not output_path:
            messagebox.showwarning("Invalid output", "Please specify an output video path.")
            return

        conf_threshold = float(self.val_conf_var.get())

        if self._validation_thread and self._validation_thread.is_alive():
            messagebox.showinfo("Processing in progress", "Validation is already running.")
            return

        def validation_worker():
            try:
                self.after(0, lambda: self.val_status_var.set("Loading model..."))
                self.after(0, lambda: self._status.set("Loading RF-DETR model..."))

                # Load model (assuming RFDETRBase can load from engine file)
                # Note: This may need adjustment based on actual RFDETR API
                model = RFDETRBase()
                if hasattr(model, 'load_engine'):
                    model.load_engine(model_path)
                elif hasattr(model, 'load'):
                    model.load(model_path)
                else:
                    # Try to initialize with engine path
                    model = RFDETRBase(model_path=model_path)

                self.after(0, lambda: self.val_status_var.set("Processing video..."))
                self.after(0, lambda: self._status.set("Processing video with detections..."))

                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise Exception("Failed to open video file")

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                frame_num = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_num += 1
                    self.after(0, lambda n=frame_num, t=total_frames: self.val_status_var.set(f"Processing frame {n}/{t}..."))

                    # Run inference
                    # Convert BGR to RGB for model
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run prediction (adjust based on actual RFDETR API)
                    try:
                        if hasattr(model, 'predict'):
                            results = model.predict(frame_rgb, conf=conf_threshold)
                        elif hasattr(model, 'inference'):
                            results = model.inference(frame_rgb, conf=conf_threshold)
                        else:
                            # Fallback: assume model returns detections in standard format
                            results = model(frame_rgb)
                        
                        # Draw detections on frame
                        # Adjust based on actual result format
                        if hasattr(results, 'boxes'):
                            boxes = results.boxes
                            if hasattr(boxes, 'xyxy'):
                                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                                conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                                cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                                
                                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                                    if conf[i] >= conf_threshold:
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                        label_text = f"Class {int(cls[i])}: {conf[i]:.2f}"
                                        cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        elif isinstance(results, (list, tuple)) and len(results) > 0:
                            # Handle list/tuple format
                            for det in results:
                                if len(det) >= 6:  # x1, y1, x2, y2, conf, cls
                                    x1, y1, x2, y2, conf, cls = det[:6]
                                    if conf >= conf_threshold:
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                        label_text = f"Class {int(cls)}: {conf:.2f}"
                                        cv2.putText(frame, label_text, (int(x1), int(y1) - 10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        # If inference fails, just write the original frame
                        pass

                    out.write(frame)

                cap.release()
                out.release()

                self.after(0, lambda: self.val_status_var.set(f"Validation complete! Output saved to: {output_path}"))
                self.after(0, lambda: self._status.set("Validation completed successfully."))
                self.after(0, lambda: messagebox.showinfo("Validation Complete", f"Processed video saved to:\n{output_path}"))
            except Exception as e:
                self.after(0, lambda: self.val_status_var.set(f"Validation failed: {str(e)}"))
                self.after(0, lambda: self._status.set("Validation failed."))
                self.after(0, lambda: messagebox.showerror("Validation Error", f"An error occurred during validation:\n{str(e)}"))

        self._validation_thread = threading.Thread(target=validation_worker, daemon=True)
        self._validation_thread.start()

    def _build_albu_transform(self, target_size: int) -> A.Compose:
        # Internal boxes are stored as pascal_voc absolute pixels already.
        # Albumentations requires bbox_params with format + label_fields to keep labels aligned. [web:29]
        t = []

        if self.albu_color.get():
            t += [
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.RandomGamma(p=1.0),
                    A.CLAHE(p=1.0),
                    A.RGBShift(p=1.0),
                ], p=0.8),
            ]

        if self.albu_noise_blur.get():
            t += [
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.ISONoise(p=1.0),
                    A.MotionBlur(p=1.0),
                    A.GaussianBlur(p=1.0),
                    A.ImageCompression(quality_range=(35, 95), p=1.0),
                ], p=0.6),
            ]

        if self.albu_geom.get():
            t += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=20, border_mode=0, p=0.7),
                A.Perspective(scale=(0.02, 0.08), p=0.25),
            ]

        # Safer bbox-aware crops (avoid RandomCrop dropping everything). [web:29]
        if self.albu_safe_crop.get():
            t += [
                A.RandomSizedBBoxSafeCrop(
                    height=target_size,
                    width=target_size,
                    erosion_rate=0.1,
                    p=0.35,
                )
            ]
        else:
            t += [A.LongestMaxSize(max_size=target_size, p=1.0), A.PadIfNeeded(target_size, target_size, p=1.0)]

        if self.albu_dropout.get():
            t += [
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(0.05, 0.25),
                    hole_width_range=(0.05, 0.25),
                    fill=0,
                    p=0.3,
                )
            ]

        # IMPORTANT: Mosaic needs extra images passed in via metadata at call time. [web:24]
        if self.albu_mosaic.get():
            gy = int(self.albu_mosaic_gy.get())
            gx = int(self.albu_mosaic_gx.get())
            p = float(self.albu_mosaic_p.get())
            t = [
                A.Mosaic(
                    grid_yx=(gy, gx),
                    target_size=(target_size, target_size),
                    fit_mode="contain",
                    metadata_key="mosaic_metadata",
                    p=p,
                )
            ] + t

        return A.Compose(
            t,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["bbox_labels"],
                min_visibility=0.0,
            ),
            p=1.0,
        )


    def _apply_albu(self, img_pil: Image.Image, boxes: List[Box], transform: A.Compose) -> Tuple[Image.Image, List[Box]]:
        # Convert PIL->numpy (RGB)
        img = np.array(img_pil)

        bboxes = []
        labels = []
        for b in boxes:
            x1, y1, x2, y2 = b.as_xyxy()
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(int(b.label))

        # For Mosaic: pass additional examples through "mosaic_metadata". [web:24]
        data = {"image": img, "bboxes": bboxes, "bbox_labels": labels}

        if self.albu_mosaic.get():
            # Need (gy*gx - 1) additional samples (approx); Albumentations example passes a list in metadata. [web:24]
            gy = int(self.albu_mosaic_gy.get())
            gx = int(self.albu_mosaic_gx.get())
            need = max(0, gy * gx - 1)

            # Sample labeled images (can include unlabeled too; labeled preferred for meaningful mosaic)
            candidates = [it for it in self.images if it.boxes]
            if len(candidates) >= need:
                picked = random.sample(candidates, need)
            else:
                picked = random.choices(candidates if candidates else self.images, k=need)

            meta = []
            for it in picked:
                im = Image.open(it.path).convert("RGB")
                arr = np.array(im)
                bb = []
                ll = []
                for bx in [b for b in it.boxes if b.kind != "prompt"]:
                    x1, y1, x2, y2 = bx.as_xyxy()
                    bb.append([float(x1), float(y1), float(x2), float(y2)])
                    ll.append(int(bx.label))
                meta.append({"image": arr, "bboxes": bb, "bbox_labels": ll})

            data["mosaic_metadata"] = meta

        out = transform(**data)

        out_img = Image.fromarray(out["image"])
        out_boxes = []
        for (x1, y1, x2, y2), lab in zip(out["bboxes"], out["bbox_labels"]):
            out_boxes.append(Box(float(x1), float(y1), float(x2), float(y2), int(lab), kind="auto"))

        return out_img, out_boxes

    def _pick_out_folder(self):
        d = filedialog.askdirectory(title="Select export folder")
        if d:
            self.out_dir_var.set(d)

    def _export_dataset(self):
        if not self.images:
            messagebox.showwarning("No images", "Nothing to export.")
            return

        out_dir = self.out_dir_var.get().strip()
        ensure_dir(out_dir)

        fmt = self.format_var.get()
        try:
            if fmt == "yolo":
                # new path: albumentations handled inside _export_yolo
                self._export_yolo(out_dir, aug_ops=None)
            elif fmt == "coco":
                self._export_coco(out_dir, aug_ops=None)
            elif fmt == "voc":
                self._export_voc(out_dir, aug_ops=None)
            else:
                messagebox.showerror("Format", f"Unknown format: {fmt}")
                return

            self._status.set(f"Exported dataset to: {out_dir}")
            messagebox.showinfo("Export", f"Export complete:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))


    def _iter_augmented_samples(self, item: ImageItem, base_img: Image.Image, boxes: List[Box], aug_ops):
        """
        Yields tuples of (suffix, img, boxes_transformed)
        Always yields original as suffix="" first.
        """
        yield "", base_img, boxes

        # If aug_ops is None or empty, only yield the original
        if aug_ops is None:
            return

        w, h = base_img.size

        for op in aug_ops:
            kind = op[0]
            if kind == "hflip":
                img2 = base_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                b2 = [flip_h_box(b, w, h) for b in boxes]
                yield "_hflip", img2, b2

            elif kind == "vflip":
                img2 = base_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                b2 = [flip_v_box(b, w, h) for b in boxes]
                yield "_vflip", img2, b2

            elif kind == "rot90":
                img2 = base_img.transpose(Image.Transpose.ROTATE_270)  # PIL rotate 90 CW == ROTATE_270
                b2 = [rotate_box_90_cw_xyxy(b, w, h) for b in boxes]
                yield "_rot90", img2, b2

            elif kind == "rot180":
                img2 = base_img.transpose(Image.Transpose.ROTATE_180)
                b2 = [rotate_box_180_xyxy(b, w, h) for b in boxes]
                yield "_rot180", img2, b2

            elif kind == "rot270":
                img2 = base_img.transpose(Image.Transpose.ROTATE_90)  # 270 CW == ROTATE_90
                b2 = [rotate_box_270_cw_xyxy(b, w, h) for b in boxes]
                yield "_rot270", img2, b2

            elif kind == "bright":
                factor = float(op[1])
                img2 = ImageEnhance.Brightness(base_img).enhance(factor)
                yield f"_bright{factor:g}", img2, boxes

    def _export_yolo(self, out_dir: str, _unused_aug_ops):
        images_dir = os.path.join(out_dir, "images")
        labels_dir = os.path.join(out_dir, "labels")
        ensure_dir(images_dir)
        ensure_dir(labels_dir)

        used_labels = sorted({b.label for it in self.images for b in it.boxes if b.kind != "prompt"})
        with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
            for lab in used_labels:
                f.write(f"{lab}\n")

        target_size = int(self.albu_mosaic_ts.get())
        copies = max(0, int(self.albu_copies.get()))
        use_albu = bool(self.albu_enable.get())

        albu_t = self._build_albu_transform(target_size) if use_albu else None

        for item in self.images:
            base = Image.open(item.path).convert("RGB")
            base_boxes = [b for b in item.boxes if b.kind != "prompt"]

            stem = os.path.splitext(os.path.basename(item.path))[0]

            # Always export original once
            self._write_yolo_sample(images_dir, labels_dir, stem, "", base, base_boxes)

            # Albumentations augmented copies
            if use_albu and albu_t is not None:
                for k in range(copies):
                    img2, b2 = self._apply_albu(base, base_boxes, albu_t)
                    self._write_yolo_sample(images_dir, labels_dir, stem, f"_albu{k:03d}", img2, b2)


    def _write_yolo_sample(self, images_dir, labels_dir, stem, suffix, img_pil, boxes):
        out_img = os.path.join(images_dir, f"{stem}{suffix}.jpg")
        img_pil.save(out_img, quality=95)

        w, h = img_pil.size
        out_lbl = os.path.join(labels_dir, f"{stem}{suffix}.txt")
        with open(out_lbl, "w", encoding="utf-8") as f:
            for b in boxes:
                x1, y1, x2, y2 = b.as_xyxy()
                cx, cy, bw, bh = yolo_xyxy_to_yolo_txt(x1, y1, x2, y2, w, h)
                f.write(f"{b.label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


    def _export_coco(self, out_dir: str, aug_ops):
        # RFDETR expects train, valid, and test subdirectories
        train_dir = os.path.join(out_dir, "train")
        valid_dir = os.path.join(out_dir, "valid")
        test_dir = os.path.join(out_dir, "test")
        
        train_images_dir = os.path.join(train_dir, "images")
        valid_images_dir = os.path.join(valid_dir, "images")
        test_images_dir = os.path.join(test_dir, "images")
        
        ensure_dir(train_images_dir)
        ensure_dir(valid_images_dir)
        ensure_dir(test_images_dir)

        used_labels = sorted({b.label for it in self.images for b in it.boxes if b.kind != "prompt"})
        label_to_catid = {lab: i + 1 for i, lab in enumerate(used_labels)}

        # Create base category structure
        categories = [
            {
                "id": label_to_catid[lab],
                "name": str(lab),
                "supercategory": "none"  # COCO format requires supercategory field
            }
            for lab in used_labels
        ]

        # Split dataset: 70% train, 20% valid, 10% test
        all_items = list(self.images)
        random.shuffle(all_items)
        total = len(all_items)
        train_end = int(total * 0.7)
        valid_end = int(total * 0.9)
        
        train_items = all_items[:train_end]
        valid_items = all_items[train_end:valid_end]
        test_items = all_items[valid_end:]

        # Helper function to create COCO structure for a split
        def create_coco_split(items, split_name):
            coco = {
                "info": {"description": f"Exported by YOLOE Dataset Labeler - {split_name}"},
                "licenses": [],
                "categories": categories,
                "images": [],
                "annotations": [],
            }
            
            ann_id = 1
            img_id = 1
            
            for item in items:
                base = Image.open(item.path).convert("RGB")
                boxes = [b for b in item.boxes if b.kind != "prompt"]

                stem = os.path.splitext(os.path.basename(item.path))[0]
                for suffix, img2, b2 in self._iter_augmented_samples(item, base, boxes, aug_ops):
                    out_name = f"{stem}{suffix}.jpg"
                    
                    # Determine which images directory to use
                    if split_name == "train":
                        images_dir = train_images_dir
                    elif split_name == "valid":
                        images_dir = valid_images_dir
                    else:  # test
                        images_dir = test_images_dir
                    
                    out_img = os.path.join(images_dir, out_name)
                    img2.save(out_img, quality=95)
                    w, h = img2.size

                    # COCO format: file_name should be relative to the root directory
                    # If root is train/, then file_name should be "images/filename.jpg"
                    coco["images"].append({
                        "id": img_id,
                        "file_name": os.path.join("images", out_name),
                        "width": w,
                        "height": h,
                    })

                    for b in b2:
                        x1, y1, x2, y2 = b.as_xyxy()
                        bw = x2 - x1
                        bh = y2 - y1
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": label_to_catid[b.label],
                            "bbox": [float(x1), float(y1), float(bw), float(bh)],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                        })
                        ann_id += 1
                    img_id += 1
            
            return coco

        # Create splits
        train_coco = create_coco_split(train_items, "train")
        valid_coco = create_coco_split(valid_items, "valid")
        test_coco = create_coco_split(test_items, "test")

        # Save annotation files
        train_annotations_path = os.path.join(train_dir, "_annotations.coco.json")
        valid_annotations_path = os.path.join(valid_dir, "_annotations.coco.json")
        test_annotations_path = os.path.join(test_dir, "_annotations.coco.json")
        
        with open(train_annotations_path, "w", encoding="utf-8") as f:
            json.dump(train_coco, f, indent=2)
        with open(valid_annotations_path, "w", encoding="utf-8") as f:
            json.dump(valid_coco, f, indent=2)
        with open(test_annotations_path, "w", encoding="utf-8") as f:
            json.dump(test_coco, f, indent=2)

    def _export_voc(self, out_dir: str, aug_ops):
        import xml.etree.ElementTree as ET

        images_dir = os.path.join(out_dir, "JPEGImages")
        ann_dir = os.path.join(out_dir, "Annotations")
        ensure_dir(images_dir)
        ensure_dir(ann_dir)

        for item in self.images:
            base = Image.open(item.path).convert("RGB")
            boxes = [b for b in item.boxes if b.kind != "prompt"]

            stem = os.path.splitext(os.path.basename(item.path))[0]
            for suffix, img2, b2 in self._iter_augmented_samples(item, base, boxes, aug_ops):
                out_name = f"{stem}{suffix}.jpg"
                out_img = os.path.join(images_dir, out_name)
                img2.save(out_img, quality=95)
                w, h = img2.size

                ann = ET.Element("annotation")
                ET.SubElement(ann, "filename").text = out_name

                size = ET.SubElement(ann, "size")
                ET.SubElement(size, "width").text = str(w)
                ET.SubElement(size, "height").text = str(h)
                ET.SubElement(size, "depth").text = "3"

                for b in b2:
                    x1, y1, x2, y2 = b.as_xyxy()
                    obj = ET.SubElement(ann, "object")
                    ET.SubElement(obj, "name").text = str(b.label)
                    bb = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bb, "xmin").text = str(int(round(x1)))
                    ET.SubElement(bb, "ymin").text = str(int(round(y1)))
                    ET.SubElement(bb, "xmax").text = str(int(round(x2)))
                    ET.SubElement(bb, "ymax").text = str(int(round(y2)))

                tree = ET.ElementTree(ann)
                tree.write(os.path.join(ann_dir, f"{stem}{suffix}.xml"), encoding="utf-8", xml_declaration=True)

    # ----------------------------
    # Shared
    # ----------------------------
    def _add_images(self, paths: List[str]):
        # Dedup by absolute path
        existing = {os.path.abspath(it.path) for it in self.images}
        added = 0
        for p in paths:
            ap = os.path.abspath(p)
            if ap in existing:
                continue
            if not os.path.isfile(ap):
                continue
            ext = os.path.splitext(ap)[1].lower()
            if ext not in IMG_EXTS:
                continue
            self.images.append(ImageItem(path=ap))
            existing.add(ap)
            added += 1

        self._refresh_lists()
        if added > 0 and self.cur_index < 0:
            self.cur_index = 0
            self._select_index(0)

        self._status.set(f"Added {added} images (total={len(self.images)}).")

    def _refresh_lists(self):
        # import screen list
        self.import_list.delete(0, "end")
        for it in self.images:
            self.import_list.insert("end", it.path)

        # labeling screen list
        self.img_list.delete(0, "end")
        for it in self.images:
            self.img_list.insert("end", os.path.basename(it.path))

    def on_close(self):
        self._cam_running = False
        self._stop_autolabel = True
        self._stop_training = True
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
