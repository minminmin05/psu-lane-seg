"""Plot the training curves of a run from its TensorBoard event file.

Reads the scalars train.py logged (Loss/train, Loss/val, IoU/train, IoU/val, LR)
and writes one PNG with four panels: loss on a linear axis, the same loss on a
log axis (the tail is invisible on a linear one), IoU, and the learning rate.

Example:
    python plot_curves.py --log-dir runs/tinylanenet --output assets/loss_curve.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Fixed roles: a series keeps its colour in every panel.
SERIES = {"train": "#2a78d6", "val": "#eb6834"}
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def read_scalars(log_dir):
    """{tag: ([epoch, ...], [value, ...])} from the event file(s) in log_dir."""
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(tag)
        scalars[tag] = ([e.step for e in events], [e.value for e in events])
    return scalars


def style_axes(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=26)
    ax.set_xlabel("Epoch", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(which="both", colors=INK_MUTED, labelsize=9, length=0)
    ax.margins(x=0.02)


def plot_train_val(ax, scalars, prefix, value_format):
    for name, color in SERIES.items():
        epochs, values = scalars[f"{prefix}/{name}"]
        ax.plot(epochs, values, color=color, linewidth=1.5, solid_capstyle="round", solid_joinstyle="round",
                label=f"{name}  (final {value_format.format(values[-1])})")
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
                       fontsize=9.5, handlelength=1.4, columnspacing=1.6, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def mark_point(ax, x, y, color, text, offset):
    """One dot with a surface ring plus a short label: the single value worth reading off the chart."""
    ax.plot([x], [y], marker="o", markersize=7, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=5, clip_on=False)
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", fontsize=9, color=INK_SECONDARY,
                ha="right", va="center")


def main():
    parser = argparse.ArgumentParser(description="Plot loss / IoU curves from TensorBoard logs.")
    parser.add_argument("--log-dir", default="runs/tinylanenet")
    parser.add_argument("--output", default="assets/loss_curve.png")
    args = parser.parse_args()

    scalars = read_scalars(args.log_dir)
    missing = [t for t in ("Loss/train", "Loss/val", "IoU/train", "IoU/val", "LR") if t not in scalars]
    if missing:
        raise SystemExit(f"No scalars {missing} under {args.log_dir}")

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=SURFACE)
    (ax_loss, ax_log), (ax_iou, ax_lr) = axes

    plot_train_val(ax_loss, scalars, "Loss", "{:.4f}")
    style_axes(ax_loss, "Loss (BCE + Dice)", "Loss")
    ax_loss.set_ylim(bottom=0)

    plot_train_val(ax_log, scalars, "Loss", "{:.4f}")
    ax_log.set_yscale("log")
    ax_log.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    style_axes(ax_log, "Loss (BCE + Dice), log scale", "Loss")

    plot_train_val(ax_iou, scalars, "IoU", "{:.4f}")
    style_axes(ax_iou, "IoU at network resolution", "IoU")
    epochs, val_iou = scalars["IoU/val"]
    best = max(range(len(val_iou)), key=val_iou.__getitem__)
    ax_iou.set_ylim(0.9, 1.0)  # the first few epochs start below the axis; the plateau is the point
    mark_point(ax_iou, epochs[best], val_iou[best], SERIES["val"],
               f"best val {val_iou[best]:.4f} (epoch {epochs[best]})", offset=(-10, -16))

    epochs, lr = scalars["LR"]
    ax_lr.plot(epochs, lr, color=INK_SECONDARY, linewidth=1.5, solid_capstyle="round", solid_joinstyle="round")
    style_axes(ax_lr, "Learning rate (warm-up, then cosine decay)", "Learning rate")
    ax_lr.set_ylim(bottom=0)
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax_lr.yaxis.get_offset_text().set_color(INK_MUTED)
    ax_lr.yaxis.get_offset_text().set_fontsize(9)

    fig.tight_layout(h_pad=2.2, w_pad=2.5)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
