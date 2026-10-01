"""Sanity reference: predict the same mask for every frame.

The mask is the average lane shape of the training labels (a pixel is lane if it
is lane in at least half of the training frames). It never looks at the image,
so the IoU it reaches is what this dataset gives away for free: the camera is
fixed on the vehicle and the lane always fills roughly the same lower half of
the picture. A trained network is only worth something if it beats this.

Masks are written in the same format as predict.py, so evaluation.py scores them:

    python constant_baseline.py --output-dir result/constant_baseline
    python evaluation.py --pred-dir result/constant_baseline/masks \
        --output-json result/constant_baseline/metrics.json --output-csv result/constant_baseline/per_image.csv
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import list_images, load_lane_mask, read_image_rgb


def main():
    parser = argparse.ArgumentParser(description="Constant-mask baseline (mean training lane shape).")
    parser.add_argument("--train-labels", default="data/labels/train")
    parser.add_argument("--images-dir", default="data/images/test", help="Frames to write a mask for.")
    parser.add_argument("--output-dir", default="result/constant_baseline")
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        raise SystemExit(f"No images found under {args.images_dir}")
    h, w = read_image_rgb(images[0]).shape[:2]

    label_paths = sorted(Path(args.train_labels).glob("*.txt"))
    if not label_paths:
        raise SystemExit(f"No label files found under {args.train_labels}")
    coverage = np.zeros((h, w), dtype=np.float64)
    for label_path in label_paths:
        coverage += load_lane_mask(label_path, w, h)
    mask = (coverage / len(label_paths) >= 0.5).astype(np.uint8) * 255

    mask_dir = Path(args.output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
    print(f"Mean lane shape of {len(label_paths)} training frames covers {mask.mean() / 255 * 100:.1f}% of the image.")
    print(f"Saved {len(images)} identical masks to {mask_dir}")


if __name__ == "__main__":
    main()
