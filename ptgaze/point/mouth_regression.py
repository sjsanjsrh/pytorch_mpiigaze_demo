"""Utility helpers for mouth regression inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


MOUTH_TOP = 13
MOUTH_BOTTOM = 14
NOSE_TIP = 1
CHIN = 152


class MouthRegressor(nn.Module):
    """Two-head regressor producing openness and lip position values."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.openness_head = nn.Linear(hidden_dim, 1)
            self.lip_position_head = nn.Linear(hidden_dim, 1)
        else:
            self.backbone = nn.Identity()
            self.openness_head = nn.Linear(input_dim, 1)
            self.lip_position_head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        openness = torch.sigmoid(self.openness_head(features))
        lip_position = torch.tanh(self.lip_position_head(features))
        return openness, lip_position


def load_regressor(model_path: Path, device: torch.device) -> Tuple[MouthRegressor, dict]:
    """Load a trained regressor bundle."""
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    model = MouthRegressor(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def compute_mouth_metrics(coords_xy: np.ndarray) -> Tuple[float, float]:
    """Return mouth height ratio and center offset using MediaPipe indices."""
    mouth_height = abs(coords_xy[MOUTH_BOTTOM][1] - coords_xy[MOUTH_TOP][1])
    face_segment = abs(coords_xy[CHIN][1] - coords_xy[NOSE_TIP][1])
    face_segment = max(face_segment, 1e-6)
    ratio = float(mouth_height / face_segment)
    mid_y = (coords_xy[MOUTH_TOP][1] + coords_xy[MOUTH_BOTTOM][1]) * 0.5
    center = float((mid_y - coords_xy[NOSE_TIP][1]) / face_segment)
    return ratio, center


def prepare_regressor_features(
    coords_norm: np.ndarray,
    dims: int,
    indices: Optional[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Flatten normalized landmarks and apply saved scaling."""
    coords_use = coords_norm[:, :dims].astype(np.float32)
    ratio, center = compute_mouth_metrics(coords_use[:, :2])
    flat = coords_use.reshape(-1)
    if indices is not None:
        flat = flat[indices]
    features = np.concatenate(([ratio, center], flat)).astype(np.float32)
    std_safe = np.where(std == 0, 1.0, std).astype(np.float32)
    features = (features - mean.astype(np.float32)) / std_safe
    return features, ratio, center


def normalise_landmarks(
    landmarks: np.ndarray,
    width: float,
    height: float,
    dims: int,
) -> np.ndarray:
    """Normalise landmark coordinates by frame resolution."""
    coords = np.asarray(landmarks, dtype=np.float32)
    if coords.ndim == 1:
        coords = coords.reshape(-1, max(dims, 1))
    if coords.shape[1] < dims:
        pad_width = dims - coords.shape[1]
        pad = np.zeros((coords.shape[0], pad_width), dtype=np.float32)
        coords = np.hstack((coords, pad))
    elif coords.shape[1] > dims:
        coords = coords[:, :dims]
    if dims >= 1 and width > 0:
        coords[:, 0] /= float(width)
    if dims >= 2 and height > 0:
        coords[:, 1] /= float(height)
    return coords
