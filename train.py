"""Train TinyLaneNet from scratch for single-class lane segmentation.

Dataset layout (built by prepare_data.py, Ultralytics YOLO-seg convention):

    <data-root>/images/{train,val}/*.jpg
    <data-root>/labels/{train,val}/*.txt

Outputs:
    checkpoints/<run-name>/{best,last}.pt     model weights (.pt)
    runs/<run-name>/                          TensorBoard event file + history.csv

Example:
    python train.py --data-root data --epochs 80 --batch-size 16
    tensorboard --logdir runs
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import AUGMENT_MODES, DEFAULT_IMG_SIZE, LaneSegDataset
from model import build_model, count_parameters
from utils import select_device


def dice_loss(logits, targets, eps=1.0):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    total = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (total + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE scores every pixel on its own; Dice scores the overlap of the whole
    region, which is what the IoU metric rewards. Both accept soft targets."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss(logits, targets)


@torch.no_grad()
def iou_per_image(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).flatten(1)
    targets = (targets > 0.5).flatten(1)
    intersection = (preds & targets).sum(dim=1).float()
    union = (preds | targets).sum(dim=1).float()
    return (intersection + eps) / (union + eps)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    """One pass over the loader. Returns sample-averaged (loss, IoU)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_iou, n_samples = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images = images.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            batch = images.size(0)
            total_loss += loss.item() * batch
            total_iou += iou_per_image(logits, masks).sum().item()
            n_samples += batch

    n_samples = max(n_samples, 1)
    return total_loss / n_samples, total_iou / n_samples


def warmup_cosine(step, warmup_steps, total_steps, final_ratio=0.01):
    """LR multiplier: linear warm-up, then cosine decay to final_ratio."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def preview_grid(model, dataset, device, n=4):
    """Image strip for TensorBoard: input | ground truth | prediction, for n validation frames."""
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n).round().astype(int)
    rows = []
    for idx in indices:
        image, mask = dataset[int(idx)]
        prob = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0].cpu()
        rows.append(torch.cat([image, mask.expand(3, -1, -1), (prob > 0.5).float().expand(3, -1, -1)], dim=2))
    return torch.cat(rows, dim=1)


def architecture_kwargs(args):
    """TinyLaneNet settings that differ from the final design (only the ablation flags set any)."""
    kwargs = {}
    if args.model == "tinylanenet":
        if args.widths:
            kwargs["widths"] = args.widths
        if args.dilations:
            kwargs["dilations"] = args.dilations
        if args.no_skip:
            kwargs["skips"] = False
        if args.no_global_context:
            kwargs["global_context"] = False
    return kwargs


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = select_device(args.device)
    print(f"Using device: {device}")

    data_root = Path(args.data_root)
    img_size = tuple(args.img_size)
    train_ds = LaneSegDataset(data_root / "images" / "train", data_root / "labels" / "train",
                              img_size, augment=args.augment, soft_mask=not args.hard_mask)
    val_ds = LaneSegDataset(data_root / "images" / "val", data_root / "labels" / "val",
                            img_size, augment="none", soft_mask=not args.hard_mask)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"Need images under {data_root}/images/train and .../val (run prepare_data.py first).")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Input: {img_size[0]}x{img_size[1]}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # Every layer is created with fresh random weights here: nothing is loaded from disk.
    mean, std = train_ds.channel_stats()
    model_kwargs = dict(architecture_kwargs(args), mean=mean, std=std)
    model = build_model(args.model, **model_kwargs).to(device)
    model = model.to(memory_format=torch.channels_last)  # NHWC: ~1.4x faster convolutions on CPU
    print(f"Model: {args.model} | {count_parameters(model):,} parameters, trained from scratch")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, warmup_steps, total_steps))

    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / args.run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    history_file = open(log_dir / "history.csv", "w", newline="")
    history = csv.writer(history_file)
    history.writerow(["epoch", "lr", "train_loss", "val_loss", "train_iou", "val_iou"])

    def save(path, epoch, val_loss, val_iou):
        torch.save({
            "model_name": args.model,
            "model_kwargs": model_kwargs,
            "model_state_dict": model.state_dict(),
            "img_size": list(img_size),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_iou": val_iou,
        }, path)

    best = {"epoch": 0, "val_iou": -1.0, "val_loss": float("inf")}
    start = time.time()
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_loss, train_iou = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)
        writer.add_scalar("LR", lr, epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            writer.add_image("val/input_gt_pred", preview_grid(model, val_ds, device), epoch)
        history.writerow([epoch, f"{lr:.6g}", f"{train_loss:.6f}", f"{val_loss:.6f}",
                          f"{train_iou:.6f}", f"{val_iou:.6f}"])
        history_file.flush()

        tqdm.write(f"Epoch {epoch:03d}/{args.epochs} | lr={lr:.2e} | "
                   f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
                   f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        save(checkpoint_dir / "last.pt", epoch, val_loss, val_iou)
        if val_iou > best["val_iou"]:
            best = {"epoch": epoch, "val_iou": val_iou, "val_loss": val_loss}
            save(checkpoint_dir / "best.pt", epoch, val_loss, val_iou)

    history_file.close()
    writer.close()
    elapsed = time.time() - start
    print(f"Training complete in {elapsed / 60:.1f} min. "
          f"Best val IoU {best['val_iou']:.4f} at epoch {best['epoch']}. "
          f"Checkpoints saved under {checkpoint_dir}")
    return {"best": best, "checkpoint_dir": str(checkpoint_dir), "train_minutes": elapsed / 60,
            "final": {"train_loss": train_loss, "val_loss": val_loss, "train_iou": train_iou, "val_iou": val_iou}}


def build_parser():
    parser = argparse.ArgumentParser(description="Train TinyLaneNet from scratch for lane segmentation.")
    parser.add_argument("--data-root", default="data", help="Root dir containing images/ and labels/.")
    parser.add_argument("--img-size", type=int, nargs=2, default=list(DEFAULT_IMG_SIZE), metavar=("W", "H"),
                        help="Network input size (width height); keep it 16:9 like the camera.")
    parser.add_argument("--model", default="tinylanenet", choices=["tinylanenet", "unet_baseline"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--augment", default="full", choices=AUGMENT_MODES,
                        help="Training augmentation: none, photometric only, or photometric + geometric.")
    parser.add_argument("--hard-mask", action="store_true",
                        help="Train on hard 0/1 targets instead of area-averaged soft masks.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--run-name", default="tinylanenet")

    ablation = parser.add_argument_group("architecture ablations (TinyLaneNet only; defaults are the final design)")
    ablation.add_argument("--widths", type=int, nargs=4, default=None, metavar="C",
                          help="Channels of the four resolution levels (default 16 32 64 96).")
    ablation.add_argument("--dilations", type=int, nargs="+", default=None,
                          help="Dilation of each block at the 1/8 level (default 1 2 3).")
    ablation.add_argument("--no-skip", action="store_true", help="Decoder without encoder skip connections.")
    ablation.add_argument("--no-global-context", action="store_true", help="Drop the global-context gate.")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
