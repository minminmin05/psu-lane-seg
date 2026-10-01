"""Before / after pictures of the lane segmentation.

One row per frame, three tiles:

    input frame (before) | ground-truth lane | predicted lane (after)

Frames are either named explicitly, or picked from the per-frame results that
evaluation.py wrote: evenly spread over the test set (typical results) or the
lowest-IoU ones (failure cases).

Example:
    python visualize.py --pick spread --num 4 --output assets/snapshots.jpg
    python visualize.py --pick worst  --num 4 --output assets/snapshots_worst.jpg
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import label_path_for_image, list_images, load_lane_mask, read_image_rgb
from evaluation import confusion, frame_metrics
from predict import overlay_mask, predict_mask
from utils import load_model, select_device

SURFACE, INK = "#fcfcfb", "#0b0b0b"
LANE_TINT = (0, 255, 0)
MODEL_TITLES = {"tinylanenet": "TinyLaneNet", "unet_baseline": "UNet baseline"}


def pick_frames(image_paths, mode, num, metrics_json):
    """Choose which frames to show; returns a list of image paths."""
    if mode == "spread":
        indices = np.linspace(0, len(image_paths) - 1, num + 2).round().astype(int)[1:-1]
        return [image_paths[i] for i in indices]
    # worst: lowest IoU according to evaluation.py
    with open(metrics_json) as f:
        per_image = json.load(f)["per_image"]
    by_stem = {p.stem: p for p in image_paths}
    worst = sorted(per_image, key=lambda r: r["iou"])
    return [by_stem[r["image"]] for r in worst if r["image"] in by_stem][:num]


def main():
    parser = argparse.ArgumentParser(description="Before/after lane segmentation snapshots.")
    parser.add_argument("--images-dir", default="data/images/test")
    parser.add_argument("--labels-dir", default="data/labels/test")
    parser.add_argument("--checkpoint", default="checkpoints/tinylanenet/best.pt")
    parser.add_argument("--frames", nargs="*", default=None, help="Frame names (file stems) to show.")
    parser.add_argument("--pick", choices=["spread", "worst"], default="spread",
                        help="How to choose frames when --frames is not given.")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--metrics-json", default="result/metrics.json", help="Needed for --pick worst.")
    parser.add_argument("--output", default="assets/snapshots.jpg")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    image_paths = list_images(args.images_dir)
    if not image_paths:
        raise SystemExit(f"No images found under {args.images_dir}")
    if args.frames:
        wanted = set(args.frames)
        chosen = [p for p in image_paths if p.stem in wanted]
    else:
        chosen = pick_frames(image_paths, args.pick, args.num, args.metrics_json)
    if not chosen:
        raise SystemExit("No frames selected.")

    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    img_size = tuple(checkpoint["img_size"])

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    rows = len(chosen)
    fig, axes = plt.subplots(rows, 3, figsize=(13.5, 2.62 * rows + 0.5), facecolor=SURFACE, squeeze=False)
    model_title = MODEL_TITLES.get(checkpoint["model_name"], checkpoint["model_name"])
    column_titles = ["Input (before)", "Ground truth", f"{model_title} prediction (after)"]

    for row, image_path in enumerate(chosen):
        image = read_image_rgb(image_path)
        h, w = image.shape[:2]
        gt_mask = load_lane_mask(label_path_for_image(image_path, args.labels_dir), w, h) * 255
        pred_mask = predict_mask(model, image, img_size, device)
        iou = frame_metrics(*confusion(pred_mask, gt_mask))["iou"]

        tiles = [image, overlay_mask(image, gt_mask, LANE_TINT), overlay_mask(image, pred_mask, LANE_TINT)]
        captions = [image_path.stem, "", f"IoU {iou:.3f}"]
        for col, (tile, caption) in enumerate(zip(tiles, captions)):
            ax = axes[row][col]
            ax.imshow(tile)
            ax.axis("off")
            if row == 0:
                ax.set_title(column_titles[col], fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)
            if caption:
                ax.text(0.012, 0.035, caption, transform=ax.transAxes, fontsize=10.5, color="white",
                        ha="left", va="bottom",
                        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "boxstyle": "round,pad=0.3"})

    fig.subplots_adjust(left=0.008, right=0.992, top=1 - 0.42 / fig.get_figheight(), bottom=0.008,
                        wspace=0.015, hspace=0.03)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"pil_kwargs": {"quality": 90}} if output.suffix.lower() in (".jpg", ".jpeg") else {}
    fig.savefig(output, dpi=120, facecolor=SURFACE, **save_kwargs)
    print(f"Saved {output} ({rows} frames)")


if __name__ == "__main__":
    main()
