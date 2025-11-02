"""Visualise selected mouth landmarks on a standard face wireframe.

The script loads the dataset recorded with `mouth_rec.py` and the reduced
landmark indices emitted by `analysis/landmark_feature_selection.py`. It draws a
canonical MediaPipe face-mesh wireframe and highlights the vertices that retain
at least one coordinate after feature selection.

Example
-------
python analysis/visualize_selected_landmarks.py `
    --dataset data/mouth_landmark_dataset.npz `
    --indices artifacts/landmark_selection/mouth_landmarks_selected_indices.npy `
    --sample-idx 0 `
    --out artifacts/landmark_selection/selected_landmarks.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    data = np.load(path)
    landmarks = data["landmarks"].astype(np.float32)
    ratios = data["ratios"].astype(np.float32)
    meta_shape: Tuple[int, ...] = ()
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            if "landmark_shape" in meta:
                meta_shape = tuple(int(v) for v in meta["landmark_shape"])
    return landmarks, ratios, meta_shape


def load_indices(path: Path) -> np.ndarray:
    indices = np.load(path)
    if indices.ndim != 1:
        raise ValueError("Selected indices file must be 1-D array")
    return indices.astype(np.int64)


def compute_unique_points(indices: np.ndarray, dims: int) -> np.ndarray:
    point_indices = indices // dims
    return np.unique(point_indices)


def infer_dims(feature_len: int, meta_shape: Tuple[int, ...]) -> int:
    if meta_shape:
        return int(meta_shape[-1])
    for candidate in (3, 2):
        if feature_len % candidate == 0:
            return candidate
    raise ValueError("Unable to infer landmark dimensionality")


def plot_landmarks(
    sample: np.ndarray,
    selected_points: np.ndarray,
    ratio: float,
    out_path: Path | None,
    show: bool,
    dims: int,
) -> None:
    coords = sample.reshape(-1, dims)
    xy = coords[:, :2]

    # Convert MediaPipe tessellation to numpy array of segments; drop edges with
    # missing indices in case the recorded landmarks omit some MediaPipe points.
    edges = np.array(list(FACEMESH_TESSELATION), dtype=np.int32)
    valid = np.all(edges < xy.shape[0], axis=1)
    edges = edges[valid]
    segments = np.stack([xy[edges[:, 0]], xy[edges[:, 1]]], axis=1)
    segments[:, :, 1] *= -1  # flip Y for display
    xy_plot = xy.copy()
    xy_plot[:, 1] *= -1

    plt.figure(figsize=(7, 7))
    lc = LineCollection(segments, colors="#888888", linewidths=0.5, alpha=0.6)
    ax = plt.gca()
    ax.add_collection(lc)
    ax.scatter(xy_plot[:, 0], xy_plot[:, 1], s=5, c="#B0B0B0", alpha=0.6)

    selected_points = selected_points[selected_points < xy.shape[0]]
    if selected_points.size > 0:
        ax.scatter(
            xy_plot[selected_points, 0],
            xy_plot[selected_points, 1],
            s=35,
            c="crimson",
            label="selected",
        )

    ax.set_title(f"Selected mouth landmarks on face mesh (ratio={ratio:.3f})")
    ax.set_aspect("equal")
    ax.axis("off")
    if selected_points.size > 0:
        ax.legend(loc="upper right")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=240)
        print(f"[INFO] Figure saved -> {out_path}")

    if show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise selected landmarks")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset npz")
    parser.add_argument("--indices", type=Path, required=True, help="Path to selected indices npy")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to visualise")
    parser.add_argument("--out", type=Path, default=None, help="Optional PNG output path")
    parser.add_argument("--show", action="store_true", help="Display matplotlib window")
    args = parser.parse_args()

    landmarks, ratios, landmark_shape = load_dataset(args.dataset)
    if not (0 <= args.sample_idx < landmarks.shape[0]):
        raise IndexError(f"sample_idx {args.sample_idx} is out of range (0..{landmarks.shape[0]-1})")

    indices = load_indices(args.indices)
    dims = infer_dims(landmarks.shape[1], landmark_shape)
    selected_points = compute_unique_points(indices, dims)

    sample = landmarks[args.sample_idx]
    plot_landmarks(sample, selected_points, ratios[args.sample_idx], args.out, args.show, dims)


if __name__ == "__main__":
    main()
