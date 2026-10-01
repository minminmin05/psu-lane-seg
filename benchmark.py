"""Measure the inference memory footprint and speed of a trained checkpoint.

    python benchmark.py --checkpoint checkpoints/tinylanenet/best.pt

Reports
  1. Model size      parameters, weights in RAM (FP32), checkpoint file on disk
  2. Activations     output size of every stage, and the peak tensor memory that
                     is allocated at the same moment during one forward pass
  3. Process memory  resident memory (RSS) of this Python process: runtime only,
                     after loading the model, after inference
  4. Speed           time per frame for the network alone and for the whole
                     predict pipeline (resize + network + upsample to full size)

The process-memory numbers are taken first, in a fresh process, so run this
script on its own rather than importing it.
"""
import argparse
import contextlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.profiler import ProfilerActivity, profile

from dataset import list_images, read_image_rgb
from model import count_parameters
from predict import predict_mask
from utils import load_model, select_device

MB = 2 ** 20


def rss_mb():
    return psutil.Process().memory_info().rss / MB


def peak_rss_mb():
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / MB
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / MB if sys.platform == "darwin" else peak / 1024  # bytes on macOS, kB on Linux


def stage_outputs(model, x):
    """(name, output shape, MB) for each top-level stage of the network."""
    rows = []
    hooks = [module.register_forward_hook(
        lambda m, inputs, out, name=name: rows.append((name, list(out.shape[1:]), out.numel() * out.element_size() / MB)))
        for name, module in model.named_children()]
    with torch.no_grad():
        model(x)
    for hook in hooks:
        hook.remove()
    return rows


@contextlib.contextmanager
def quiet_profiler():
    """The profiler backend logs start/stop lines straight to the C-level stderr; hide them."""
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def peak_activation_mb(model, x):
    """Peak memory held by tensors during one forward pass (weights not included).

    CPU: replay the allocations/frees recorded by the profiler and keep the maximum.
    GPU: read the allocator's own high-water mark.
    """
    with torch.no_grad():
        model(x)  # warm-up: one-off allocations of the backend are not part of a steady-state frame
        if x.is_cuda:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            model(x)
            torch.cuda.synchronize()
            return (torch.cuda.max_memory_allocated() - before) / MB
        with quiet_profiler(), profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            model(x)
    allocated = peak = 0
    for event in sorted(prof.events(), key=lambda e: e.time_range.start):
        allocated += event.self_cpu_memory_usage
        peak = max(peak, allocated)
    return peak / MB


def time_ms(fn, repeats, warmup=10):
    """Median and minimum wall time of fn() in milliseconds.

    The median is the typical frame; the minimum is what the code costs when
    nothing else on the machine gets in the way.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times)), float(np.min(times))


def main():
    parser = argparse.ArgumentParser(description="Inference memory footprint and speed of a checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/tinylanenet/best.pt")
    parser.add_argument("--image", default=None,
                        help="A full-size frame for the pipeline test (default: first image in data/images/test).")
    parser.add_argument("--device", default="cpu", help="cpu (default: the laptop case), cuda or auto.")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch (default: all cores).")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output-json", default="result/benchmark.json")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = select_device(args.device)

    # --- 3. process memory: measured before anything else touches the allocator --------------
    rss_runtime = rss_mb()
    model, checkpoint = load_model(args.checkpoint, device)
    rss_loaded = rss_mb()

    img_w, img_h = checkpoint["img_size"]
    x = torch.rand(1, 3, img_h, img_w, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(x)
    rss_inference = rss_mb()

    if args.image:
        frame = read_image_rgb(args.image)
    else:
        candidates = list_images("data/images/test")
        frame = read_image_rgb(candidates[0]) if candidates else np.random.randint(0, 256, (720, 1280, 3), np.uint8)
    for _ in range(20):
        predict_mask(model, frame, (img_w, img_h), device)
    rss_pipeline = rss_mb()
    rss_peak = peak_rss_mb()

    # --- 1. model size -------------------------------------------------------------------------
    n_params = count_parameters(model)
    weights_mb = sum(t.numel() * t.element_size() for t in model.state_dict().values()) / MB
    file_mb = Path(args.checkpoint).stat().st_size / MB

    # --- 2. activations ------------------------------------------------------------------------
    stages = stage_outputs(model, x)
    activation_mb = peak_activation_mb(model, x)

    # --- 4. speed ------------------------------------------------------------------------------
    def forward():
        with torch.no_grad():
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    net_median, net_min = time_ms(forward, args.repeats)
    pipe_median, pipe_min = time_ms(lambda: predict_mask(model, frame, (img_w, img_h), device), args.repeats)

    report = {
        "checkpoint": str(args.checkpoint),
        "model": checkpoint["model_name"],
        "input": [1, 3, img_h, img_w],
        "frame": list(frame.shape),
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "parameters": n_params,
        "weights_mb": weights_mb,
        "checkpoint_file_mb": file_mb,
        "stage_outputs": [{"stage": n, "shape": s, "mb": mb} for n, s, mb in stages],
        "peak_activation_mb": activation_mb,
        "rss_mb": {"runtime": rss_runtime, "model_loaded": rss_loaded, "after_inference": rss_inference,
                   "after_pipeline": rss_pipeline, "peak": rss_peak},
        "latency_ms": {"network_median": net_median, "network_min": net_min,
                       "pipeline_median": pipe_median, "pipeline_min": pipe_min},
    }

    print(f"=== Inference footprint: {checkpoint['model_name']} @ {img_w}x{img_h} on {device} "
          f"({torch.get_num_threads()} threads) ===")
    print("Model size")
    print(f"  parameters               : {n_params:,}")
    print(f"  weights in RAM (FP32)    : {weights_mb:.2f} MB")
    print(f"  checkpoint file          : {file_mb:.2f} MB")
    print("Activations (batch 1)")
    for name, shape, mb in stages:
        print(f"  {name:<8} {'x'.join(map(str, shape)):<12} {mb:.3f} MB")
    print(f"  peak allocated at once   : {activation_mb:.2f} MB")
    print(f"  weights + peak activations: {weights_mb + activation_mb:.2f} MB")
    print("Process memory (RSS)")
    print(f"  Python + PyTorch + OpenCV: {rss_runtime:.0f} MB")
    print(f"  + model loaded           : {rss_loaded:.0f} MB ({rss_loaded - rss_runtime:+.1f})")
    print(f"  + network inference      : {rss_inference:.0f} MB ({rss_inference - rss_runtime:+.1f})")
    print(f"  + full {frame.shape[1]}x{frame.shape[0]} pipeline : {rss_pipeline:.0f} MB ({rss_pipeline - rss_runtime:+.1f})")
    print(f"  peak                     : {rss_peak:.0f} MB")
    print("Speed (batch 1)")
    print(f"  network only             : {net_median:.2f} ms median, {net_min:.2f} ms min "
          f"({1000 / net_median:.0f} FPS)")
    print(f"  resize + network + upsample: {pipe_median:.2f} ms median, {pipe_min:.2f} ms min "
          f"({1000 / pipe_median:.0f} FPS)")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {args.output_json}")


if __name__ == "__main__":
    main()
