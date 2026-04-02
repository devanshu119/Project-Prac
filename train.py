# file: train.py
import argparse
import logging
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.unet_model import UNet
from models.deeplab_model import create_deeplabv3
from utils.data_loading import OilSpillDataset


def train_net(
    dir_img,
    dir_mask,
    model_name,
    epochs=5,
    batch_size=1,
    learning_rate=1e-5,
    val_percent=0.1,
    save_checkpoint=True,
    img_scale=1.0,
    amp=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Model selection ---
    if model_name.lower() == "unet":
        net = UNet(n_channels=1, n_classes=3)
    elif model_name.lower() == "deeplab":
        net = create_deeplabv3(n_classes=3, n_channels=1)  # single-channel input
    else:
        raise ValueError("Invalid model name. Choose 'unet' or 'deeplab'.")

    net.to(device)
    logging.info(
        f"""Network:
        {net.n_channels if hasattr(net, 'n_channels') else 1} input channels
        {net.n_classes if hasattr(net, 'n_classes') else 3} output classes"""
    )

    # --- Dataset ---
    dataset = OilSpillDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        Image scaling:   {img_scale}
        Mixed Precision: {amp}
    """
    )

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    # --- Training loop ---
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch+1}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, targets = batch  # Dataset returns (image, mask)
                images = images.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    outputs = net(images)
                    # DeepLab returns OrderedDict, UNet returns Tensor
                    logits = outputs["out"] if isinstance(outputs, (dict, OrderedDict)) else outputs
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                pbar.update(images.shape[0])
                pbar.set_postfix({"loss": loss.item()})

        logging.info(f"Epoch {epoch+1} finished! Loss: {epoch_loss/n_train}")

        if save_checkpoint:
            Path("checkpoints/").mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), f"checkpoints/{model_name}_epoch{epoch+1}.pth")
            logging.info(f"Checkpoint saved at epoch {epoch+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model on oil spill dataset")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", "-l", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--model", "-m", type=str, default="unet", help="Model name: 'unet' or 'deeplab'")
    parser.add_argument("--scale", "-s", type=float, default=0.5, help="Downscaling factor of the images")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--no-save", action="store_true", default=False, help="Disable saving checkpoints")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    dir_img = Path("data/train_images")
    dir_mask = Path("data/train_masks")

    train_net(
        dir_img=dir_img,
        dir_mask=dir_mask,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_scale=args.scale,
        amp=args.amp,
        save_checkpoint=not args.no_save,
    )
