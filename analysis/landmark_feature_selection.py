"""Utility to rank and select informative mouth landmarks (PyTorch edition).

The script assumes the dataset was recorded with `mouth_rec.py` and produces a
reduced landmark matrix suited for lightweight CPU inference. Instead of
scikit-learn it trains a multinomial logistic regression (single linear layer)
with an L1 penalty implemented in PyTorch. Feature importance is measured by
the average absolute weight across classes.

Example
-------
python analysis/landmark_feature_selection.py `
    --dataset data `
    --keep-top 200 `
    --out-prefix artifacts/landmark_selection/mouth_landmarks

Outputs
-------
- `<prefix>_selected_indices.npy` : indices of retained landmark coordinates
- `<prefix>_reduced_dataset.npz`  : dataset containing only the retained
    landmark coordinates alongside the original meta features
- `<prefix>_ranking.json`         : full ranking with L1-weight magnitudes
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MEDIAPIPE_FACE_LANDMARKS = 468


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    labels = data["labels"]
    ratios = data["ratios"].astype(np.float32)
    centers = data["centers"].astype(np.float32)
    landmarks = data["landmarks"].astype(np.float32)
    timestamps = data["timestamps"].astype(np.float64)
    return labels, ratios, centers, landmarks, timestamps


def resolve_dataset_paths(target: str | Path) -> List[Path]:
    pattern = str(target)
    matched: List[Path] = []

    if any(token in pattern for token in "*?[]"):
        matched = sorted(Path(p) for p in glob.glob(pattern))
    else:
        path = Path(pattern)
        if path.is_dir():
            shards = sorted(path.glob("mouth_landmark_dataset_*.npz"))
            if shards:
                matched = shards
            else:
                matched = sorted(path.glob("*.npz"))
        elif path.is_file():
            matched = [path]

    if not matched:
        raise FileNotFoundError(f"No dataset files found for '{target}'.")

    dedup: dict[str, Path] = {}
    for item in matched:
        if item.is_file():
            try:
                key = item.resolve().as_posix()
            except FileNotFoundError:
                # Resolve can fail for paths that disappear; skip them.
                continue
            dedup[key] = item

    resolved = list(dedup.values())
    if len(resolved) > 1:
        filtered = [p for p in resolved if p.name != "mouth_landmark_dataset.npz"]
        if filtered:
            resolved = filtered

    if not resolved:
        raise FileNotFoundError(f"No valid dataset files found for '{target}'.")

    return sorted(resolved)


def load_datasets(paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_chunks: List[np.ndarray] = []
    ratio_chunks: List[np.ndarray] = []
    center_chunks: List[np.ndarray] = []
    landmark_chunks: List[np.ndarray] = []
    timestamp_chunks: List[np.ndarray] = []

    for path in paths:
        labels, ratios, centers, landmarks, timestamps = load_dataset(path)
        label_chunks.append(labels)
        ratio_chunks.append(ratios)
        center_chunks.append(centers)
        landmark_chunks.append(landmarks)
        timestamp_chunks.append(timestamps)

    labels = np.concatenate(label_chunks, axis=0)
    ratios = np.concatenate(ratio_chunks, axis=0)
    centers = np.concatenate(center_chunks, axis=0)
    landmarks = np.concatenate(landmark_chunks, axis=0)
    timestamps = np.concatenate(timestamp_chunks, axis=0)
    return labels, ratios, centers, landmarks, timestamps


def infer_landmark_dims(total_coords: int) -> int:
    if total_coords % MEDIAPIPE_FACE_LANDMARKS == 0:
        dims = total_coords // MEDIAPIPE_FACE_LANDMARKS
        if dims in (2, 3):
            return dims
    if total_coords % 3 == 0:
        return 3
    return 2


def prepare_features(
    ratios: np.ndarray,
    centers: np.ndarray,
    landmarks: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    features = np.column_stack([ratios, centers, landmarks])
    feature_names: List[str] = ["ratio", "center"] + [f"lm_{i}" for i in range(landmarks.shape[1])]
    return features, feature_names


def standardise(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    normalised = (x - mean) / std
    return normalised.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def filter_low_confidence(
    labels: np.ndarray,
    ratios: np.ndarray,
    centers: np.ndarray,
    landmarks: np.ndarray,
    timestamps: np.ndarray,
    min_spread: float | None,
    max_zero_fraction: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = labels.shape[0]
    mask = np.ones(num_samples, dtype=bool)

    finite_mask = (
        np.isfinite(ratios)
        & np.isfinite(centers)
        & np.isfinite(timestamps)
        & np.isfinite(landmarks).all(axis=1)
    )
    mask &= finite_mask

    non_zero_mask = ~(np.all(landmarks == 0.0, axis=1))
    mask &= non_zero_mask

    dims = infer_landmark_dims(landmarks.shape[1])
    coords = landmarks.reshape(landmarks.shape[0], -1, dims)

    if min_spread is not None:
        spread = coords.max(axis=1) - coords.min(axis=1)
        max_span = np.max(spread, axis=1)
        mask &= max_span >= float(min_spread)

    if max_zero_fraction is not None:
        zero_fraction = (coords == 0.0).mean(axis=(1, 2))
        mask &= zero_fraction <= float(max_zero_fraction)

    if mask.all():
        return labels, ratios, centers, landmarks, timestamps

    filtered_labels = labels[mask]
    filtered_ratios = ratios[mask]
    filtered_centers = centers[mask]
    filtered_landmarks = landmarks[mask]
    filtered_timestamps = timestamps[mask]

    removed = num_samples - filtered_labels.shape[0]
    print(f"[INFO] Filtered out {removed} low-confidence samples (remaining: {filtered_labels.shape[0]}).")

    return (
        filtered_labels,
        filtered_ratios,
        filtered_centers,
        filtered_landmarks,
        filtered_timestamps,
    )


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_sparse_linear(
    x: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_lambda: float,
    device: torch.device,
) -> LinearClassifier:
    classes, y_indices = np.unique(labels, return_inverse=True)
    num_classes = len(classes)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y_indices.astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LinearClassifier(x.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            l1_penalty = 0.0
            if l1_lambda > 0.0:
                l1_penalty = model.linear.weight.abs().mean() * l1_lambda
            total_loss = loss + l1_penalty

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"[INFO] Epoch {epoch + 1}/{epochs} - loss={avg_loss:.4f}")

    model.linear.weight.data = model.linear.weight.data.cpu()
    model.linear.bias.data = model.linear.bias.data.cpu()

    return model


def rank_features(model: LinearClassifier, feature_names: List[str]) -> List[Tuple[str, float]]:
    with torch.no_grad():
        weights = model.linear.weight.abs().mean(dim=0).numpy()
    ranked = sorted(zip(feature_names, weights.tolist()), key=lambda item: item[1], reverse=True)
    return ranked


def select_indices(ranked: List[Tuple[str, float]], keep_top: int) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    selected = []
    kept = []
    for name, score in ranked:
        if name in ("ratio", "center"):
            kept.append((name, score))
            continue
        idx = int(name.split("_")[1])
        selected.append((idx, score))
    top = selected[: max(0, keep_top)]
    indices = np.array([idx for idx, _ in top], dtype=np.int32)
    kept.extend((f"lm_{idx}", score) for idx, score in top)
    return indices, kept


def save_outputs(
    out_prefix: Path,
    labels: np.ndarray,
    ratios: np.ndarray,
    centers: np.ndarray,
    landmarks: np.ndarray,
    timestamps: np.ndarray,
    indices: np.ndarray,
    feature_ranking: List[Tuple[str, float]],
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_prefix.with_name(out_prefix.name + "_selected_indices"), indices)

    reduced_landmarks = landmarks[:, indices]
    np.savez(
        out_prefix.with_name(out_prefix.name + "_reduced_dataset"),
        labels=labels,
        ratios=ratios,
        centers=centers,
        landmarks=reduced_landmarks,
        timestamps=timestamps,
    )

    summary_path = out_prefix.with_name(out_prefix.name + "_ranking.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "kept_landmark_indices": indices.tolist(),
                "feature_ranking": feature_ranking,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select informative mouth landmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path, directory, or glob for dataset shards (e.g. data or data/mouth_landmark_dataset_*.npz)",
    )
    parser.add_argument("--keep-top", type=int, default=200, help="Number of landmark coordinates to keep")
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Prefix for outputs (without extension)",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for sparse linear model")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--l1-lambda", type=float, default=1e-3, help="Strength of L1 penalty on weights")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use CUDA if available",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=None,
        help="Minimum landmark spread threshold for keeping a sample (set to disable)",
    )
    parser.add_argument(
        "--max-zero-fraction",
        type=float,
        default=None,
        help="Maximum allowed fraction of zero landmark coordinates per sample",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable dataset shuffling before training (enabled by default when combining shards)",
    )
    args = parser.parse_args()

    dataset_paths = resolve_dataset_paths(args.dataset)
    print("[INFO] Using dataset shards:\n" + "\n".join(f"  - {path}" for path in dataset_paths))

    labels, ratios, centers, landmarks, timestamps = load_datasets(dataset_paths)
    print(f"[INFO] Loaded {labels.shape[0]} samples from {len(dataset_paths)} file(s).")

    (
        labels,
        ratios,
        centers,
        landmarks,
        timestamps,
    ) = filter_low_confidence(
        labels,
        ratios,
        centers,
        landmarks,
        timestamps,
        min_spread=args.min_spread,
        max_zero_fraction=args.max_zero_fraction,
    )

    x_raw, feature_names = prepare_features(ratios, centers, landmarks)
    x_norm, mean, std = standardise(x_raw)

    if not args.no_shuffle:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(x_norm.shape[0])
        x_norm = x_norm[order]
        labels = labels[order]
        ratios = ratios[order]
        centers = centers[order]
        landmarks = landmarks[order]
        timestamps = timestamps[order]
        print(f"[INFO] Shuffled dataset with seed {args.seed}.")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU")

    model = fit_sparse_linear(
        x_norm,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_lambda=args.l1_lambda,
        device=device,
    )
    ranked = rank_features(model, feature_names)
    indices, kept_features = select_indices(ranked, args.keep_top)

    save_outputs(args.out_prefix, labels, ratios, centers, landmarks, timestamps, indices, kept_features)
    print(f"[INFO] Selected {len(indices)} landmark coordinates.")
    print(f"[INFO] Ranking exported next to {args.out_prefix}.")


if __name__ == "__main__":
    main()
