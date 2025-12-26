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

        self.nb.add(self.screen_import, text="1) Import")
        self.nb.add(self.screen_label, text="2) Labeling")
        self.nb.add(self.screen_output, text="3) Output")

        self._build_import()
        self._build_labeling()
        self._build_output()

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

        cam = ttk.Labelframe(frm, text="Camera capture (optional)")
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

        self.cam_index_var = tk.IntVar(value=0)
        ttk.Label(cam, text="Camera index:").grid(row=1, column=2, sticky="w", padx=8, pady=8)
        ttk.Entry(cam, textvariable=self.cam_index_var, width=8).grid(row=1, column=3, sticky="w", padx=8, pady=8)

        ttk.Button(cam, text="Start capture", command=self._start_camera_capture).grid(row=1, column=4, padx=8, pady=8, sticky="e")
        ttk.Button(cam, text="Stop", command=self._stop_camera_capture).grid(row=1, column=5, padx=8, pady=8, sticky="e")

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

    def _start_camera_capture(self):
        if cv2 is None:
            messagebox.showerror("OpenCV missing", "Install opencv-python to use camera capture.")
            return
        if self._cam_running:
            return

        out_dir = self.cam_out_var.get().strip()
        ensure_dir(out_dir)
        interval = float(self.cam_interval_var.get())
        cam_idx = int(self.cam_index_var.get())

        self._cam_running = True

        def loop():
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                self._status.set("Camera failed to open.")
                self._cam_running = False
                return
            self._status.set("Camera capture running...")
            while self._cam_running:
                ret, frame = cap.read()
                if ret:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fn = os.path.join(out_dir, f"cap_{ts}_{int(time.time()*1000)%1000:03d}.jpg")
                    cv2.imwrite(fn, frame)
                    self.after(0, lambda p=fn: self._add_images([p]))
                time.sleep(max(0.05, interval))
            cap.release()
            self.after(0, lambda: self._status.set("Camera capture stopped."))

        self._cam_thread = threading.Thread(target=loop, daemon=True)
        self._cam_thread.start()

    def _stop_camera_capture(self):
        self._cam_running = False

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
                self._status.set("Loading YOLOE model...")
                if self._yoloe_model is None:
                    self._yoloe_model = YOLOE(MODEL_PATH)

                # Apply auto labels per image
                for i, item in enumerate(self.images):
                    if self._stop_autolabel:
                        break

                    # Remove previous auto boxes for this label set (optional)
                    item.boxes = [b for b in item.boxes if b.kind != "auto"]

                    self._status.set(f"Auto-labeling {i+1}/{len(self.images)}: {os.path.basename(item.path)}")
                    results = self._yoloe_model.predict(
                        item.path,
                        refer_image=refer_image,
                        visual_prompts=visual_prompts,
                        predictor=YOLOEVPSegPredictor,
                        conf=conf,
                        iou=iou,
                        verbose=False,
                    )
                    r0 = results[0]
                    if not hasattr(r0, "boxes") or r0.boxes is None:
                        continue

                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    c = r0.boxes.cls.cpu().numpy().astype(int)  # these are the sequential ids
                    for (x1, y1, x2, y2), cid in zip(xyxy, c):
                        lab = seq_to_label.get(int(cid), int(cid))
                        item.boxes.append(Box(float(x1), float(y1), float(x2), float(y2), label=lab, kind="auto"))

                    # refresh current display
                    if self.cur_index == i:
                        self.after(0, self._refresh_viewer)

                self.after(0, lambda: self._status.set("Auto-label done."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Auto-label error", str(e)))
                self.after(0, lambda: self._status.set("Auto-label failed."))

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

        aug = ttk.Labelframe(frm, text="Augmentations (optional)")
        aug.grid(row=2, column=0, sticky="nsew", padx=12, pady=12)
        aug.columnconfigure(0, weight=1)

        self.aug_hflip = tk.BooleanVar(value=False)
        self.aug_vflip = tk.BooleanVar(value=False)
        self.aug_rot90 = tk.BooleanVar(value=False)
        self.aug_rot180 = tk.BooleanVar(value=False)
        self.aug_rot270 = tk.BooleanVar(value=False)
        self.aug_bright = tk.BooleanVar(value=False)

        grid = ttk.Frame(aug)
        grid.pack(fill="x", padx=8, pady=8)

        ttk.Checkbutton(grid, text="Horizontal flip", variable=self.aug_hflip).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Vertical flip", variable=self.aug_vflip).grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Rotate 90", variable=self.aug_rot90).grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Rotate 180", variable=self.aug_rot180).grid(row=1, column=1, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(grid, text="Rotate 270", variable=self.aug_rot270).grid(row=1, column=2, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(grid, text="Brightness jitter", variable=self.aug_bright).grid(row=2, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(grid, text="Brightness factor (e.g., 0.8 or 1.2):").grid(row=2, column=1, sticky="e", padx=6, pady=4)
        self.bright_factor_var = tk.DoubleVar(value=1.2)
        ttk.Entry(grid, textvariable=self.bright_factor_var, width=8).grid(row=2, column=2, sticky="w", padx=6, pady=4)

        note = ttk.Label(
            aug,
            text=("Tip: augmentations create extra images + transformed labels; only simple geometry ops are included here.\n"
                  "If you want mosaic/cutout/HSV/noise, it’s best to integrate Albumentations later."),
        )
        note.pack(fill="x", padx=8, pady=(0, 8))

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
        if fmt not in ("yolo", "coco", "voc"):
            messagebox.showerror("Format", f"Unknown format: {fmt}")
            return

        # Build augmentation plan
        aug_ops = []
        if self.aug_hflip.get():
            aug_ops.append(("hflip",))
        if self.aug_vflip.get():
            aug_ops.append(("vflip",))
        if self.aug_rot90.get():
            aug_ops.append(("rot90",))
        if self.aug_rot180.get():
            aug_ops.append(("rot180",))
        if self.aug_rot270.get():
            aug_ops.append(("rot270",))
        if self.aug_bright.get():
            aug_ops.append(("bright", float(self.bright_factor_var.get())))

        # Export
        try:
            if fmt == "yolo":
                self._export_yolo(out_dir, aug_ops)
            elif fmt == "coco":
                self._export_coco(out_dir, aug_ops)
            else:
                self._export_voc(out_dir, aug_ops)
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

    def _export_yolo(self, out_dir: str, aug_ops):
        images_dir = os.path.join(out_dir, "images")
        labels_dir = os.path.join(out_dir, "labels")
        ensure_dir(images_dir)
        ensure_dir(labels_dir)

        # classes.txt (optional helper)
        used_labels = sorted({b.label for it in self.images for b in it.boxes if b.kind != "prompt"})
        with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
            for lab in used_labels:
                f.write(f"{lab}\n")

        for item in self.images:
            base = Image.open(item.path).convert("RGB")
            boxes = [b for b in item.boxes if b.kind != "prompt"]

            stem = os.path.splitext(os.path.basename(item.path))[0]
            for suffix, img2, b2 in self._iter_augmented_samples(item, base, boxes, aug_ops):
                out_img = os.path.join(images_dir, f"{stem}{suffix}.jpg")
                img2.save(out_img, quality=95)

                w, h = img2.size
                out_lbl = os.path.join(labels_dir, f"{stem}{suffix}.txt")
                with open(out_lbl, "w", encoding="utf-8") as f:
                    for b in b2:
                        x1, y1, x2, y2 = b.as_xyxy()
                        cx, cy, bw, bh = yolo_xyxy_to_yolo_txt(x1, y1, x2, y2, w, h)
                        # label is numeric as requested
                        f.write(f"{b.label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _export_coco(self, out_dir: str, aug_ops):
        images_dir = os.path.join(out_dir, "images")
        ensure_dir(images_dir)

        used_labels = sorted({b.label for it in self.images for b in it.boxes if b.kind != "prompt"})
        label_to_catid = {lab: i + 1 for i, lab in enumerate(used_labels)}

        coco = {
            "info": {"description": "Exported by YOLOE Dataset Labeler"},
            "licenses": [],
            "categories": [{"id": label_to_catid[lab], "name": str(lab)} for lab in used_labels],
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        img_id = 1

        for item in self.images:
            base = Image.open(item.path).convert("RGB")
            boxes = [b for b in item.boxes if b.kind != "prompt"]

            stem = os.path.splitext(os.path.basename(item.path))[0]
            for suffix, img2, b2 in self._iter_augmented_samples(item, base, boxes, aug_ops):
                out_name = f"{stem}{suffix}.jpg"
                out_img = os.path.join(images_dir, out_name)
                img2.save(out_img, quality=95)
                w, h = img2.size

                coco["images"].append({
                    "id": img_id,
                    "file_name": out_name,
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

        with open(os.path.join(out_dir, "annotations.coco.json"), "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

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
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
