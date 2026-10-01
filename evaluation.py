"""Evaluate predicted lane masks against the YOLO-seg ground truth with pixel-wise IoU.

For every frame the mask written by predict.py is compared, at full resolution,
with the mask rasterized from the ground-truth lane polygon.

Metrics defined by the assignment:
  - detection flag : a frame is "detected" (positive) when its IoU >= 0.6
  - detection rate : detected frames / all frames
  - average IoU of the detected frames

Also reported for reference: mean IoU over all frames, Dice, precision, recall,
pixel accuracy, and where the wrong pixels are: how wide the band of errors along
the true lane edge is, and how many of the wrong pixels lie close to that edge.

Example:
    python evaluation.py --labels-dir data/labels/test --pred-dir result/masks
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from dataset import load_lane_mask

NEAR_EDGE_PX = 10


def confusion(pred_mask, gt_mask):
    """Pixel counts (tp, fp, fn, tn) of a predicted mask against the ground truth."""
    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return tp, fp, fn, pred.size - tp - fp - fn


def ratio(numerator, denominator, empty=1.0):
    """numerator / denominator; `empty` when there is nothing to measure (0 / 0)."""
    return numerator / denominator if denominator else empty


def frame_metrics(tp, fp, fn, tn):
    return {
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }


def edge_stats(pred_mask, gt_mask, near_px=NEAR_EDGE_PX):
    """Where are the wrong pixels relative to the true lane edge?

    Returns (wrong pixels, wrong pixels within near_px of the edge, edge length in pixels).
    Only the edge inside the picture counts: where the lane runs out of the frame
    there is no edge for the network to find.
    """
    gt = (gt_mask > 0).astype(np.uint8)
    wrong = (pred_mask > 0) != (gt > 0)
    # Distance of every pixel to the nearest pixel of the other class, i.e. to the lane edge.
    distance = np.where(gt > 0, cv2.distanceTransform(gt, cv2.DIST_L2, 3),
                        cv2.distanceTransform(1 - gt, cv2.DIST_L2, 3))
    h, w = gt.shape
    contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_px = 0
    for contour in contours:
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        edge_px += int(np.count_nonzero((x > 0) & (x < w - 1) & (y > 0) & (y < h - 1)))
    return int(np.count_nonzero(wrong)), int(np.count_nonzero(wrong & (distance <= near_px))), edge_px


def main():
    parser = argparse.ArgumentParser(description="Evaluate lane masks via pixel-wise IoU.")
    parser.add_argument("--labels-dir", default="data/labels/test", help="YOLO-seg ground-truth label directory.")
    parser.add_argument("--pred-dir", default="result/masks", help="Predicted mask PNGs from predict.py.")
    parser.add_argument("--iou-threshold", type=float, default=0.6, help="IoU >= threshold counts as detected.")
    parser.add_argument("--output-json", default="result/metrics.json")
    parser.add_argument("--output-csv", default="result/per_image.csv")
    args = parser.parse_args()

    label_paths = sorted(Path(args.labels_dir).glob("*.txt"))
    if not label_paths:
        print(f"No label files found under {args.labels_dir}")
        return

    results = []
    wrong_px = near_edge_px = edge_px = total_px = 0
    for label_path in label_paths:
        pred_path = Path(args.pred_dir) / f"{label_path.stem}.png"
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE) if pred_path.exists() else None
        if pred_mask is None:
            # No usable prediction is a miss, not a frame to leave out of the statistics.
            print(f"Warning: no prediction for {label_path.stem}, counted as not detected.")
            metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "pixel_accuracy": 0.0}
        else:
            h, w = pred_mask.shape
            gt_mask = load_lane_mask(label_path, w, h)
            metrics = frame_metrics(*confusion(pred_mask, gt_mask))
            wrong, near, edge = edge_stats(pred_mask, gt_mask)
            wrong_px += wrong
            near_edge_px += near
            edge_px += edge
            total_px += h * w
        results.append({"image": label_path.stem, **metrics,
                        "detected": metrics["iou"] >= args.iou_threshold})

    n = len(results)
    detected = [r for r in results if r["detected"]]
    ious = np.array([r["iou"] for r in results])
    summary = {
        "num_images": n,
        "iou_threshold": args.iou_threshold,
        "num_detected": len(detected),
        "detection_rate": len(detected) / n,
        "avg_iou_detected": float(np.mean([r["iou"] for r in detected])) if detected else 0.0,
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "min_iou": float(ious.min()),
        "mean_dice": float(np.mean([r["dice"] for r in results])),
        "mean_precision": float(np.mean([r["precision"] for r in results])),
        "mean_recall": float(np.mean([r["recall"] for r in results])),
        "mean_pixel_accuracy": float(np.mean([r["pixel_accuracy"] for r in results])),
        "wrong_pixel_rate": ratio(wrong_px, total_px, empty=0.0),
        "error_band_px": ratio(wrong_px, edge_px, empty=0.0),
        f"wrong_within_{NEAR_EDGE_PX}px_of_edge": ratio(near_edge_px, wrong_px),
    }

    print("=== Lane Segmentation Evaluation ===")
    print(f"Images evaluated            : {n}")
    print(f"Detected (IoU >= {args.iou_threshold})      : {len(detected)} ({summary['detection_rate'] * 100:.1f}%)")
    print(f"Average IoU (detected)      : {summary['avg_iou_detected']:.4f}")
    print(f"Mean IoU (all images)       : {summary['mean_iou']:.4f}  "
          f"(median {summary['median_iou']:.4f}, min {summary['min_iou']:.4f})")
    print(f"Mean Dice / precision / recall : {summary['mean_dice']:.4f} / "
          f"{summary['mean_precision']:.4f} / {summary['mean_recall']:.4f}")
    print(f"Mean pixel accuracy         : {summary['mean_pixel_accuracy']:.4f}")
    print(f"Wrong pixels                : {summary['wrong_pixel_rate'] * 100:.2f}% of the image, "
          f"a band {summary['error_band_px']:.1f} px wide along the lane edge; "
          f"{summary[f'wrong_within_{NEAR_EDGE_PX}px_of_edge'] * 100:.0f}% lie within {NEAR_EDGE_PX} px of it")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "per_image": results}, f, indent=2)
        print(f"Saved summary to {args.output_json}")
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved per-image results to {args.output_csv}")


if __name__ == "__main__":
    main()
