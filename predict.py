import argparse
import logging
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.unet_model import UNet   # or replace with the model you trained

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def predict_img(net,
                full_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5,
                n_classes=2):
    net.eval()
    img = full_img.convert("L")  # grayscale
    preprocess = transforms.Compose([
        transforms.Resize((int(img.size[1]*scale_factor), int(img.size[0]*scale_factor))),
        transforms.ToTensor(),
    ])
    img = preprocess(img).unsqueeze(0)  # add batch dim

    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)

        if n_classes > 1:
            probs = torch.softmax(output, dim=1)[0]
            mask = torch.argmax(probs, dim=0).cpu().numpy()
        else:
            probs = torch.sigmoid(output)[0][0]
            mask = (probs > out_threshold).cpu().numpy()

    return mask

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def get_args():
    parser = argparse.ArgumentParser(description='Predict mask from input image')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch20.pth',
                        help='Path to the trained model')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to input SAR image')
    parser.add_argument('--output', '-o', default='output_mask.png',
                        help='Path to save output mask')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for input image')
    parser.add_argument('--classes', '-c', type=int, default=2,
                        help='Number of classes (2 for binary)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=args.classes)
    net.to(device=device)

    logging.info(f'Loading model from {args.model}')
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info(f'Predicting mask for {args.input} ...')
    img = Image.open(args.input)

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=0.5,
                       device=device,
                       n_classes=args.classes)

    out_img = mask_to_image(mask)
    out_img.save(args.output)

    logging.info(f'Mask saved to {args.output}')
