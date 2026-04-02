# file: scripts/predict_pytorch.py
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

from models.unet_model import UNet
from models.deeplab_model import create_deeplabv3
from utils.data_loading import OilSpillDataset
from scripts.preprocess_sar import preprocess_image
from scripts.refine_mask import refine_segmentation_mask


def load_model(checkpoint_path: str, model_name: str, n_classes: int = 3, n_channels: int = 1, device="cpu"):
    """Load a trained PyTorch model (UNet or DeepLab)."""
    if model_name.lower() == "unet":
        model = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name.lower() == "deeplab":
        model = create_deeplabv3(n_classes=n_classes, n_channels=n_channels)
    else:
        raise ValueError("Invalid model name. Choose 'unet' or 'deeplab'.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_tensor: np.ndarray, device="cpu"):
    """
    Run inference on a preprocessed SAR image.
    image_tensor: (1, H, W, 1) numpy array
    Returns: segmentation mask (H, W)
    """
    # Convert to torch tensor
    tensor = torch.from_numpy(image_tensor).permute(0, 3, 1, 2).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        outputs = model(tensor)
        # DeepLab returns OrderedDict
        if isinstance(outputs, dict):
            outputs = outputs["out"]

        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
    return preds.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained PyTorch model")
    parser.add_argument("-i", "--input", required=True, help="Path to input SAR image")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("-m", "--model", choices=["unet", "deeplab"], required=True, help="Model type")
    parser.add_argument("-o", "--output", required=True, help="Path to save the predicted mask")
    parser.add_argument("--refine", action="store_true", help="Apply mask refinement after prediction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model = load_model(args.checkpoint, args.model, n_classes=3, n_channels=1, device=device)

    # 2. Preprocess SAR image
    preprocessed = preprocess_image(args.input, target_size=(512, 512))

    # 3. Predict mask
    raw_mask = predict_image(model, preprocessed, device)

    # Convert mask to 0–255 for saving
    mask_img = (raw_mask * (255 // raw_mask.max())).astype(np.uint8)

    # Save raw mask
    cv2.imwrite(args.output, mask_img)
    print(f"Raw mask saved at {args.output}")

    # 4. Optional refinement
    if args.refine:
        refined = refine_segmentation_mask(args.output, min_area_threshold=100)
        refined_path = str(Path(args.output).with_name("refined_" + Path(args.output).name))
        cv2.imwrite(refined_path, refined)
        print(f"Refined mask saved at {refined_path}")


if __name__ == "__main__":
    main()
