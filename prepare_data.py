"""Build the lane-only dataset used by this repo from the PSU-reservoir YOLO-seg export.

Source (Assignment-8 CVAT export, Ultralytics YOLO-seg format, 5 classes, any split layout):

    <src>/images/**/psu_track_XXXX.jpg
    <src>/labels/**/psu_track_XXXX.txt      # class_id x1 y1 x2 y2 ... (normalized to [0, 1])

Output (single class, 0 = lane):

    <dst>/images/{train,val,test}/*.jpg
    <dst>/labels/{train,val,test}/*.txt     # only the Lane polygons survive

Two things happen here:

1. Only the polygons of the `Lane` class are kept. Track lines (polylines) and
   sideway annotations are dropped -- this assignment is lane segmentation only.

2. The frames are re-split 65:35 (train+val : test). The frames come from one
   continuous drive around the reservoir, so neighbouring frames are near
   duplicates; a random split would put almost every test frame right next to a
   training frame. Instead the sequence is cut into blocks of `--block` frames
   and each block is split in temporal order:

        |<------- train ------->|<- val ->|<------ test ------>|
        0                       58        65                  100

   Every part of the track is still represented in training, but each test frame
   belongs to a 35-frame stretch the network has never seen.

Example:
    python prepare_data.py --src D:/datasets/psu_reservoir_dataset --dst data
"""
import argparse
import re
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def natural_key(stem):
    """Sort 'psu_track_2' before 'psu_track_10'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", stem)]


def find_pairs(src):
    """Return [(stem, image_path, label_path)] sorted by frame order."""
    src = Path(src)
    labels = {p.stem: p for p in (src / "labels").rglob("*.txt")}
    pairs = []
    for image_path in (src / "images").rglob("*"):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue
        if image_path.stem not in labels:
            print(f"Warning: no label file for {image_path.name}, skipping.")
            continue
        pairs.append((image_path.stem, image_path, labels[image_path.stem]))
    return sorted(pairs, key=lambda t: natural_key(t[0]))


def block_split(n, block, val_frac, test_frac):
    """Assign each of n ordered frames to train/val/test, block by block."""
    n_test = round(block * test_frac)
    n_val = round(block * val_frac)
    n_train = block - n_val - n_test
    assignment = []
    for i in range(n):
        pos = i % block
        if pos < n_train:
            assignment.append("train")
        elif pos < n_train + n_val:
            assignment.append("val")
        else:
            assignment.append("test")
    return assignment


def lane_only_lines(label_path, lane_class_id):
    """Keep the polygons of the lane class and re-label them as class 0."""
    kept = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class_id + at least 3 (x, y) points
            continue
        if int(parts[0]) != lane_class_id:
            continue
        kept.append(" ".join(["0"] + parts[1:]))
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Source YOLO-seg dataset root (contains images/ and labels/).")
    parser.add_argument("--dst", default="data", help="Output root for the lane-only dataset.")
    parser.add_argument("--lane-class-id", type=int, default=0, help="Class id of the lane polygon in the source labels.")
    parser.add_argument("--block", type=int, default=100, help="Frames per temporal block.")
    parser.add_argument("--val-frac", type=float, default=0.07, help="Part of each block held out for validation.")
    parser.add_argument("--test-frac", type=float, default=0.35, help="Part of each block held out for testing.")
    parser.add_argument("--splits-dir", default="splits", help="Where to write the frame lists of each split.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing dataset under --dst.")
    args = parser.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise SystemExit("--dst must not overlap --src: the source dataset is never modified.")

    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"No image/label pairs found under {src}")
    assignment = block_split(len(pairs), args.block, args.val_frac, args.test_frac)

    out_dirs = [dst / kind / split for kind in ("images", "labels") for split in SPLITS]
    if any(d.exists() and any(d.iterdir()) for d in out_dirs) and not args.overwrite:
        raise SystemExit(f"{dst} already holds a dataset; pass --overwrite to rebuild it.")
    for out_dir in out_dirs:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    members = {split: [] for split in SPLITS}
    n_polygons = 0
    n_without_lane = 0
    for (stem, image_path, label_path), split in zip(pairs, assignment):
        lines = lane_only_lines(label_path, args.lane_class_id)
        n_polygons += len(lines)
        n_without_lane += not lines
        shutil.copy2(image_path, dst / "images" / split / image_path.name)
        (dst / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        members[split].append(stem)

    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (splits_dir / f"{split}.txt").write_text("\n".join(members[split]) + "\n")

    total = len(pairs)
    print(f"Source frames : {total}")
    print(f"Lane polygons : {n_polygons} ({n_without_lane} frames without a lane)")
    for split in SPLITS:
        print(f"{split:<5} : {len(members[split]):4d} frames ({len(members[split]) / total * 100:.1f}%)")
    print(f"Dataset written to {dst.resolve()}")
    print(f"Split lists written to {splits_dir.resolve()}")


if __name__ == "__main__":
    main()
