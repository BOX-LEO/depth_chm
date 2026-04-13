"""Shared helpers used across the `scripts/` pipeline stages."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation


def read_tif_height(file_path: str) -> np.ndarray:
    """Read band 1 of a GeoTIFF as float32 and flip vertically."""
    with rasterio.open(file_path) as src:
        chm = src.read(1).astype(np.float32)
    return np.flipud(chm)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def is_local_model(model_path: str) -> bool:
    return os.path.isdir(model_path) or os.path.isdir(os.path.abspath(model_path))


def load_model_and_processor(model_path: str, device: torch.device | None = None):
    """Load a DepthAnything model + processor, handling local dirs vs HF Hub IDs."""
    is_local = is_local_model(model_path)
    resolved_path = os.path.abspath(model_path) if is_local else model_path
    processor = AutoImageProcessor.from_pretrained(resolved_path, local_files_only=is_local)
    model = DepthAnythingForDepthEstimation.from_pretrained(resolved_path, local_files_only=is_local)
    if device is not None:
        model = model.to(device)
    return processor, model


def resize_prediction(pred: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize a 2-D prediction tensor to `target_shape=(H, W)`."""
    return F.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=target_shape,
        mode='bilinear',
        align_corners=True,
    ).squeeze()


def list_tiles(directory: str, extensions: Sequence[str]) -> list[str]:
    """Sorted absolute paths of files in `directory` whose name ends with any of `extensions` (case-insensitive)."""
    exts = tuple(e.lower() for e in extensions)
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    )
