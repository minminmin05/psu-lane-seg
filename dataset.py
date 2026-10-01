"""Dataset utilities for lane segmentation from Ultralytics YOLO-seg polygon labels.

Expected on-disk layout (Ultralytics convention, built by prepare_data.py):

    <data_root>/images/{train,val,test}/*.jpg
    <data_root>/labels/{train,val,test}/*.txt

Each label file has one line per polygon instance:

    class_id x1 y1 x2 y2 x3 y3 ... xn yn

with all coordinates normalized to [0, 1] relative to the *original* image size.
Only polygons of the lane class are used; anything else in a label file is ignored.

Source frames are 1280x720 but the network works on a much smaller 16:9 image
(128x72 by default). Polygons are rasterized at the original resolution first and
then area-averaged down, so a pixel of the low-resolution target holds the
fraction of that pixel covered by lane (a soft, anti-aliased mask) instead of a
hard 0/1 decision. That keeps sub-pixel information about where the lane edge
runs. (In the README's ablation it scored the same as hard 0/1 targets, which
remain available through `soft_mask=False`.)
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
LANE_CLASS_ID = 0
DEFAULT_IMG_SIZE = (128, 72)  # (width, height)


def list_images(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path_for_image(image_path, labels_dir):
    return Path(labels_dir) / f"{Path(image_path).stem}.txt"


def load_yolo_seg_polygons(label_path, img_w, img_h, class_id=LANE_CLASS_ID):
    """Parse a YOLO-seg label file into a list of int32 pixel-coordinate polygons."""
    label_path = Path(label_path)
    polygons = []
    if not label_path.exists():
        return polygons
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 (x, y) points
                continue
            if int(parts[0]) != class_id:
                continue
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            polygons.append(np.round(coords).astype(np.int32))
    return polygons


def polygons_to_mask(polygons, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, color=1)
    return mask


def read_image_rgb(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_lane_mask(label_path, img_w, img_h):
    """Binary {0, 1} lane mask at the original image resolution."""
    return polygons_to_mask(load_yolo_seg_polygons(label_path, img_w, img_h), img_w, img_h)


def resize_image(image, img_size):
    """Area-average an RGB uint8 image down to the network input size (width, height)."""
    return cv2.resize(image, tuple(img_size), interpolation=cv2.INTER_AREA)


def resize_mask(mask, img_size):
    """Area-average a {0, 1} mask to float32 lane coverage in [0, 1] at (width, height)."""
    return cv2.resize(mask.astype(np.float32), tuple(img_size), interpolation=cv2.INTER_AREA)


def image_to_tensor(image):
    """HxWx3 uint8 RGB -> 3xHxW float32 in [0, 1]. Normalization lives inside the model."""
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)


AUGMENT_MODES = ("none", "photometric", "full")


def augment(image, mask, rng, geometric=True):
    """Light augmentation of a network-resolution sample.

    image: HxWx3 uint8, mask: HxW float32. Returns the same types.

    Photometric (outdoor light / sensor variation):
      - white-balance shift: independent gain per colour channel
      - brightness shift: additive offset, sun vs. cloud vs. tree shade
      - slight Gaussian blur: defocus / motion while driving
    Geometric (camera pose variation), applied to image and mask together:
      - horizontal flip: a lane is still a lane when mirrored
      - small rotation / zoom / shift: camera roll, pitch and mounting tolerance
    """
    h, w = mask.shape

    if geometric and rng.random() < 0.5:
        image = image[:, ::-1]
        mask = mask[:, ::-1]

    if geometric and rng.random() < 0.5:
        angle = rng.uniform(-4.0, 4.0)
        scale = rng.uniform(0.92, 1.12)
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        matrix[0, 2] += rng.uniform(-0.05, 0.05) * w
        matrix[1, 2] += rng.uniform(-0.05, 0.05) * h
        image = cv2.warpAffine(np.ascontiguousarray(image), matrix, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), matrix, (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    img = image.astype(np.float32)

    if rng.random() < 0.5:
        gains = rng.uniform(0.90, 1.10, size=3).astype(np.float32)
        img = img * gains[np.newaxis, np.newaxis, :]

    if rng.random() < 0.5:
        img = img + rng.uniform(-30.0, 30.0)

    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:
        sigma = rng.uniform(0.3, 1.0) * w / 128.0  # same blur relative to the image at any input size
        img = cv2.GaussianBlur(img, (0, 0), sigma)

    return img, np.clip(mask, 0.0, 1.0)


class LaneSegDataset(Dataset):
    """Lane images + masks, resized once and kept in RAM (a 128x72 frame is 27 kB)."""

    def __init__(self, images_dir, labels_dir, img_size=DEFAULT_IMG_SIZE, augment="none", soft_mask=True):
        if augment not in AUGMENT_MODES:
            raise ValueError(f"augment must be one of {AUGMENT_MODES}, got {augment!r}")
        self.image_paths = list_images(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = tuple(img_size)
        self.augment = augment

        self.images, self.masks = [], []
        for image_path in self.image_paths:
            image = read_image_rgb(image_path)
            h, w = image.shape[:2]
            mask = load_lane_mask(label_path_for_image(image_path, self.labels_dir), w, h)
            mask = resize_mask(mask, self.img_size)
            if not soft_mask:
                mask = (mask >= 0.5).astype(np.float32)
            self.images.append(resize_image(image, self.img_size))
            self.masks.append(mask)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        if self.augment != "none":
            # Seed from torch so every DataLoader worker draws its own stream.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            image, mask = augment(image, mask, rng, geometric=self.augment == "full")
        image_t = image_to_tensor(image)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0)
        return image_t, mask_t

    def channel_stats(self):
        """Per-channel mean / std of the cached images in [0, 1]; used to set the input normalization."""
        pixels = np.concatenate([img.reshape(-1, 3) for img in self.images]).astype(np.float64) / 255.0
        return pixels.mean(axis=0).tolist(), pixels.std(axis=0).tolist()
