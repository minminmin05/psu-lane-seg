"""Run lane segmentation on a folder of images and save the binary masks to disk.

For each image:
    resize to the network input (128x72) -> TinyLaneNet -> lane probability map
    -> bilinear upsample back to the original resolution (1280x720)
    -> threshold -> 0/255 PNG mask

Masks are written to <output-dir>/masks/<image name>.png (default: result/masks/).

Example:
    python predict.py --images-dir data/images/test --checkpoint checkpoints/tinylanenet/best.pt
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import image_to_tensor, list_images, read_image_rgb, resize_image
from utils import load_model, select_device


@torch.no_grad()
def predict_mask(model, image_rgb, img_size, device, threshold=0.5, upsample="bilinear"):
    """Binary lane mask (HxW uint8, 0 or 255) at the resolution of the given image."""
    h, w = image_rgb.shape[:2]
    tensor = image_to_tensor(resize_image(image_rgb, img_size)).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    if upsample == "nearest":
        # Blocky reference: threshold at network resolution, then enlarge the mask.
        return cv2.resize((prob > threshold).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    # Upsample the probabilities, not the thresholded mask: the lane edge then
    # falls between low-resolution pixels instead of following their blocky outline.
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8) * 255


def overlay_mask(image_rgb, mask, color=(0, 255, 0), alpha=0.4):
    """Tint the masked pixels of an RGB image; used for the before/after pictures."""
    out = image_rgb.copy()
    lane = mask > 0
    out[lane] = (out[lane] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description="Lane segmentation inference -> binary masks.")
    parser.add_argument("--images-dir", default="data/images/test")
    parser.add_argument("--checkpoint", default="checkpoints/tinylanenet/best.pt")
    parser.add_argument("--output-dir", default="result", help="Masks go to <output-dir>/masks/.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--upsample", default="bilinear", choices=["bilinear", "nearest"],
                        help="bilinear: enlarge the probability map, then threshold (default). "
                             "nearest: threshold first, then enlarge the mask (for comparison).")
    parser.add_argument("--save-overlay", action="store_true",
                        help="Also save the image with the predicted lane tinted green to <output-dir>/overlays/.")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda.")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}")
    model, checkpoint = load_model(args.checkpoint, device)
    img_size = tuple(checkpoint["img_size"])
    print(f"Loaded {checkpoint['model_name']} from {args.checkpoint} "
          f"(epoch {checkpoint['epoch']}, input {img_size[0]}x{img_size[1]})")

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    mask_dir = Path(args.output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = Path(args.output_dir) / "overlays"
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    seconds = 0.0
    for image_path in images:
        image_rgb = read_image_rgb(image_path)
        start = time.perf_counter()
        mask = predict_mask(model, image_rgb, img_size, device, args.threshold, args.upsample)
        seconds += time.perf_counter() - start

        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
        if args.save_overlay:
            overlay = cv2.cvtColor(overlay_mask(image_rgb, mask), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_dir / f"{image_path.stem}.jpg"), overlay)

    print(f"Saved {len(images)} masks to {mask_dir} "
          f"({seconds / len(images) * 1000:.1f} ms per image for resize + network + upsample)")


if __name__ == "__main__":
    main()
