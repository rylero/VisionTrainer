import argparse
import os
import subprocess
import sys
import logging
from rfdetr import RFDETRBase

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_onnx_to_engine(onnx_path, engine_path, workspace_size=4096):
    """
    Converts an ONNX model to a TensorRT engine file using trtexec.
    
    Args:
        onnx_path (str): Path to the input ONNX model.
        engine_path (str): Path where the final .engine file will be saved.
        workspace_size (int): Max workspace size in MB for TensorRT optimization.
    """
    logger.info(f"Converting {onnx_path} to {engine_path}...")
    
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found at {onnx_path}. Cannot convert to engine.")
        sys.exit(1)

    # Check if trtexec is available
    try:
        # Standard trtexec command
        # --fp16: Use FP16 precision for better performance on supported GPUs
        # --memPoolSize=workspace:N: Set max workspace size
        command = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
            f"--memPoolSize=workspace:{workspace_size}"
        ]
        
        logger.info(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info("TensorRT conversion successful.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during TensorRT conversion: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("'trtexec' not found. Please ensure TensorRT is installed and 'trtexec' is in your PATH.")
        logger.info("Alternatively, you can manually convert the ONNX file to .engine using the TensorRT Python API.")
        sys.exit(1)

def rfdetr_train(
    dataset_path,
    output_dir="./models",
    engine_output="trained_rfdetr.engine",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    workspace_size=4096
):
    """
    Main training function for RF-DETR.
    
    Args:
        dataset_path (str): Path to the COCO-formatted labeled dataset directory.
        output_dir (str): Directory to save training artifacts and models.
        engine_output (str): Name of the final .engine file.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        grad_accum_steps (int): Gradient accumulation steps.
        lr (float): Learning rate.
        workspace_size (int): TensorRT workspace size in MB.
    """
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # 1. Initialize the model
    logger.info("Initializing RF-DETR model...")
    try:
        model = RFDETRBase()
    except Exception as e:
        logger.error(f"Failed to initialize RF-DETR model: {e}")
        raise

    # 2. Train the model
    logger.info(f"Starting training on dataset: {dataset_path}")
    logger.info(f"Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, grad_accum_steps={grad_accum_steps}")
    
    try:
        model.train(
            dataset_dir=dataset_path,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            tensorboard=True # Enable TensorBoard logging for better monitoring
        )
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise
        
    logger.info("Training completed successfully.")

    # 3. Export to ONNX
    onnx_output_path = os.path.join(output_dir, "model.onnx")
    logger.info(f"Exporting model to ONNX: {onnx_output_path}")
    try:
        model.export(output_path=onnx_output_path)
    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise
    
    # 4. Convert ONNX to .engine
    final_engine_path = os.path.join(output_dir, engine_output)
    convert_onnx_to_engine(onnx_output_path, final_engine_path, workspace_size=workspace_size)
    
    logger.info(f"Success! Final TensorRT engine saved at: {final_engine_path}")
    return final_engine_path

def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR model and export to TensorRT .engine")
    
    # Dataset and Output paths
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the COCO-formatted labeled dataset directory")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save training artifacts and models")
    parser.add_argument("--engine_output", type=str, default="trained_rfdetr.engine", help="Name of the final .engine file")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--workspace_size", type=int, default=4096, help="TensorRT workspace size in MB")
    
    args = parser.parse_args()

    try:
        rfdetr_train(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            engine_output=args.engine_output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            workspace_size=args.workspace_size
        )
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()

