from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    labels: np.ndarray
    ratios: np.ndarray
    centers: np.ndarray
    landmarks: np.ndarray
    timestamps: np.ndarray
    landmark_shape: Tuple[int, ...]


def load_dataset(path: Path) -> DatasetBundle:
    data = np.load(path)
    meta_shape: Tuple[int, ...] = ()
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            if "landmark_shape" in meta:
                meta_shape = tuple(int(v) for v in meta["landmark_shape"])

    return DatasetBundle(
        labels=data["labels"],
        ratios=data["ratios"].astype(np.float32),
        centers=data["centers"].astype(np.float32),
        landmarks=data["landmarks"].astype(np.float32),
        timestamps=data["timestamps"].astype(np.float64),
        landmark_shape=meta_shape,
    )


def infer_dims(feature_len: int, meta_shape: Tuple[int, ...]) -> int:
    if meta_shape:
        return int(meta_shape[-1])
    for candidate in (3, 2):
        if feature_len % candidate == 0:
            return candidate
    raise ValueError("Unable to infer landmark dimensionality")


def labels_to_targets(labels: np.ndarray) -> np.ndarray:
    """Convert categorical labels to (openness, lip_position) pairs.
    
    Use strong targets to ensure clear separation:
    - lip_raise: 0.8 (strong raise signal)
    - lip_lower: -0.8 (strong lower signal)
    - neutral states: 0.0
    """
    openness = np.zeros(len(labels), dtype=np.float32)
    lip_position = np.zeros(len(labels), dtype=np.float32)
    
    for i, label in enumerate(labels):
        if label == "mouth_open":
            openness[i] = 1.0
            lip_position[i] = 0.0
        elif label == "mouth_closed":
            openness[i] = 0.0
            lip_position[i] = 0.0
        elif label == "lip_raise":
            openness[i] = 0.0
            lip_position[i] = 0.8  # Strong positive
        elif label == "lip_lower":
            openness[i] = 0.0
            lip_position[i] = -0.8  # Strong negative
    
    return np.column_stack([openness, lip_position])


def prepare_features(
    bundle: DatasetBundle,
    indices: Optional[np.ndarray],
    image_width: Optional[float],
    image_height: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    dims = infer_dims(bundle.landmarks.shape[1], bundle.landmark_shape)
    landmarks = bundle.landmarks.astype(np.float32).copy()

    if image_width and image_width > 0.0:
        landmarks[:, 0::dims] /= float(image_width)
    if dims >= 2 and image_height and image_height > 0.0:
        landmarks[:, 1::dims] /= float(image_height)

    if indices is not None:
        landmarks = landmarks[:, indices]

    features = np.column_stack([bundle.ratios, bundle.centers, landmarks]).astype(np.float32)
    targets = labels_to_targets(bundle.labels)

    return features, targets, dims


def standardise(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    normalised = (features - mean) / std
    return normalised.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


class MouthRegressor(nn.Module):
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
        openness = torch.sigmoid(self.openness_head(features))  # [0, 1]
        lip_position = torch.tanh(self.lip_position_head(features))  # [-1, 1]
        return openness, lip_position


def split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    pivot = int(x.shape[0] * (1.0 - val_ratio))
    train_idx, val_idx = indices[:pivot], indices[pivot:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).mean())


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(pred - target).mean())


def evaluate(model: MouthRegressor, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float, Dict[str, float]]:
    """Evaluate model and return metrics + distribution stats."""
    model.eval()
    all_openness_pred = []
    all_lip_pred = []
    all_openness_target = []
    all_lip_target = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            openness_pred, lip_pred = model(batch_x)
            
            all_openness_pred.append(openness_pred.cpu().numpy())
            all_lip_pred.append(lip_pred.cpu().numpy())
            all_openness_target.append(batch_y[:, 0].numpy())
            all_lip_target.append(batch_y[:, 1].numpy())
    
    openness_pred = np.concatenate(all_openness_pred).flatten()
    lip_pred = np.concatenate(all_lip_pred).flatten()
    openness_target = np.concatenate(all_openness_target)
    lip_target = np.concatenate(all_lip_target)
    
    openness_mae = mae(openness_pred, openness_target)
    lip_mae = mae(lip_pred, lip_target)
    openness_mse = mse(openness_pred, openness_target)
    lip_mse = mse(lip_pred, lip_target)
    
    # Distribution statistics - overall and per target class
    stats = {
        "lip_pred_mean": float(lip_pred.mean()),
        "lip_pred_std": float(lip_pred.std()),
        "lip_pred_min": float(lip_pred.min()),
        "lip_pred_max": float(lip_pred.max()),
    }
    
    # Per-class statistics
    for target_val, label in [(-0.7, "lower"), (0.0, "neutral"), (0.7, "raise")]:
        mask = np.abs(lip_target - target_val) < 0.1
        if mask.sum() > 0:
            stats[f"lip_pred_{label}_mean"] = float(lip_pred[mask].mean())
            stats[f"lip_pred_{label}_std"] = float(lip_pred[mask].std())
    
    return openness_mae, lip_mae, openness_mse, lip_mse, stats
    lip_mae = mae(lip_pred, lip_target)
    openness_mse = mse(openness_pred, openness_target)
    lip_mse = mse(lip_pred, lip_target)
    
    return openness_mae, lip_mae, openness_mse, lip_mse


def train(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    indices = np.load(args.indices).astype(np.int64) if args.indices else None

    features, targets, dims = prepare_features(
        dataset,
        indices,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    features, mean, std = standardise(features)

    x_train, y_train, x_val, y_val = split_train_val(features, targets, args.val_ratio, args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU")

    model = MouthRegressor(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    train_loader = build_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = build_loader(x_val, y_val, args.batch_size, shuffle=False)

    best_loss = float('inf')
    best_state = None

    print(f"[INFO] Training samples: {x_train.shape[0]}, Validation: {x_val.shape[0]}")
    print(f"[INFO] Features: {features.shape[1]}, Hidden: {args.hidden_dim}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            openness_pred, lip_pred = model(batch_x)
            
            # Weighted loss: emphasize lip_position learning
            loss_openness = criterion(openness_pred.squeeze(), batch_y[:, 0])
            loss_lip = criterion(lip_pred.squeeze(), batch_y[:, 1]) * 2.0  # 2x weight
            loss = loss_openness + loss_lip

            if args.l1_lambda > 0.0:
                l1 = 0.0
                for param in model.parameters():
                    l1 += param.abs().sum()
                loss = loss + args.l1_lambda * l1 / batch_x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        epoch_loss = running_loss / x_train.shape[0]
        o_mae, l_mae, o_mse, l_mse, stats = evaluate(model, val_loader, device)
        val_total_mae = (o_mae + l_mae) / 2
        
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"[INFO] Epoch {epoch:03d} | loss={epoch_loss:.4f} | "
                  f"open_mae={o_mae:.4f} lip_mae={l_mae:.4f} | avg_mae={val_total_mae:.4f}")
            print(f"       lip_pred: mean={stats['lip_pred_mean']:.3f} std={stats['lip_pred_std']:.3f} "
                  f"range=[{stats['lip_pred_min']:.3f}, {stats['lip_pred_max']:.3f}]")
            
            # Per-class predictions
            if 'lip_pred_lower_mean' in stats:
                print(f"       lower: {stats['lip_pred_lower_mean']:.3f}±{stats['lip_pred_lower_std']:.3f} | "
                      f"neutral: {stats.get('lip_pred_neutral_mean', 0):.3f}±{stats.get('lip_pred_neutral_std', 0):.3f} | "
                      f"raise: {stats.get('lip_pred_raise_mean', 0):.3f}±{stats.get('lip_pred_raise_std', 0):.3f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] Restored best checkpoint with loss={best_loss:.4f}")

    output_path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": features.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "mean": mean,
            "std": std,
            "indices": indices,
            "dims": dims,
            "model_type": "regressor",
            "image_width": args.image_width,
            "image_height": args.image_height,
        },
    }
    torch.save(bundle, output_path)
    print(f"[INFO] Model saved -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mouth regressor (openness + lip_position)")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to mouth_landmark_dataset.npz")
    parser.add_argument("--indices", type=Path, default=None, help="Optional .npy indices for reduced landmarks")
    parser.add_argument("--out", type=Path, required=True, help="Output path for trained model bundle (.pt)")
    parser.add_argument("--image-width", type=float, default=640.0, help="Image width used during data capture")
    parser.add_argument("--image-height", type=float, default=480.0, help="Image height used during data capture")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--l1-lambda", type=float, default=0.0, help="L1 penalty strength")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden size (0 for linear)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout applied when hidden-dim > 0")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
