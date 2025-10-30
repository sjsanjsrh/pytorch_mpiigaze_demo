"""Utility to rank and select informative mouth landmarks (PyTorch edition).

The script assumes the dataset was recorded with `mouth_rec.py` and produces a
reduced landmark matrix suited for lightweight CPU inference. Instead of
scikit-learn it trains a multinomial logistic regression (single linear layer)
with an L1 penalty implemented in PyTorch. Feature importance is measured by
the average absolute weight across classes.

Example
-------
python analysis/landmark_feature_selection.py \
        --dataset data/mouth_landmark_dataset.npz \
        --keep-top 200 \
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
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    labels = data["labels"]
    ratios = data["ratios"].astype(np.float32)
    centers = data["centers"].astype(np.float32)
    landmarks = data["landmarks"].astype(np.float32)
    timestamps = data["timestamps"].astype(np.float64)
    return labels, ratios, centers, landmarks, timestamps


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
    parser.add_argument("--dataset", type=Path, required=True, help="Path to mouth_landmark_dataset.npz")
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
    args = parser.parse_args()

    labels, ratios, centers, landmarks, timestamps = load_dataset(args.dataset)
    x_raw, feature_names = prepare_features(ratios, centers, landmarks)
    x_norm, mean, std = standardise(x_raw)

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
