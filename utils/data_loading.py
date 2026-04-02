# file: utils/data_loading.py
import logging
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class OilSpillDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.scale = scale

        # collect all image stems (ignore extension)
        self.ids = [p.stem for p in self.images_dir.glob("*.*")]
        logging.info(f"Creating dataset with {len(self.ids)} examples")

        if not self.ids:
            raise RuntimeError(f"No input files found in {images_dir}")

    def __len__(self):
        return len(self.ids)

    def preprocess_mask(self, mask: Image.Image) -> torch.Tensor:
        # Convert to numpy
        mask_np = np.array(mask, dtype=np.uint8)

        # Map 255 → 1, everything else → 0
        mask_np = np.where(mask_np == 255, 1, 0)

        # If you want a third class, e.g., 127 → 2, uncomment:
        # mask_np = np.where(mask_np == 127, 2, mask_np)

        return torch.as_tensor(mask_np, dtype=torch.long)

    def preprocess_img(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")  # grayscale SAR
        img = np.array(img, dtype=np.float32) / 255.0  # normalize
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        return torch.as_tensor(img, dtype=torch.float32)

    def __getitem__(self, idx):
        name = self.ids[idx]

        # robustly find matching image and mask regardless of extension
        img_file = next(self.images_dir.glob(f"{name}.*"))
        mask_file = next(self.masks_dir.glob(f"{name}.*"))

        img = Image.open(img_file).convert("L")
        mask = Image.open(mask_file).convert("L")

        img_tensor = self.preprocess_img(img)
        mask_tensor = self.preprocess_mask(mask)

        return img_tensor, mask_tensor
