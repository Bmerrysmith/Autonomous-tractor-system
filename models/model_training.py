"""Standalone model training using NumPy only.

This module trains a compact MLP for lane/localization target regression.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.synthetic_data import SupervisedLaneDataset, generate_supervised_lane_dataset


@dataclass(frozen=True)
class TrainArtifacts:
    history: dict[str, list[float]]
    target_names: list[str]
    y_true_test: np.ndarray
    y_pred_test: np.ndarray
    rmse_per_target: np.ndarray
    mae_per_target: np.ndarray
    r2_per_target: np.ndarray
    files: list[Path]


class NumpyMLP:
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int], output_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        h1, h2 = hidden_dims
        self.w1 = rng.normal(0.0, 0.08, (input_dim, h1))
        self.b1 = np.zeros((1, h1))
        self.w2 = rng.normal(0.0, 0.08, (h1, h2))
        self.b2 = np.zeros((1, h2))
        self.w3 = rng.normal(0.0, 0.08, (h2, output_dim))
        self.b3 = np.zeros((1, output_dim))

        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def params(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "w3": self.w3,
            "b3": self.b3,
        }

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        z1 = x @ self.w1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = self._relu(z2)
        y = a2 @ self.w3 + self.b3
        return y, (x, z1, a1, z2, a2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, cache: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
        x, z1, a1, z2, a2 = cache
        n = y_true.shape[0]
        dy = (2.0 / n) * (y_pred - y_true)

        dw3 = a2.T @ dy
        db3 = np.sum(dy, axis=0, keepdims=True)

        da2 = dy @ self.w3.T
        dz2 = da2 * (z2 > 0)
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.w2.T
        dz1 = da1 * (z1 > 0)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2, "w3": dw3, "b3": db3}

    def step_adam(self, grads: dict[str, np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.t += 1
        for name, param in self.params().items():
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (g * g)

            m_hat = self.m[name] / (1.0 - beta1**self.t)
            v_hat = self.v[name] / (1.0 - beta2**self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _split_dataset(ds: SupervisedLaneDataset, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    idx = np.arange(ds.features.shape[0])
    rng.shuffle(idx)

    n = len(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    i_train = idx[:n_train]
    i_val = idx[n_train : n_train + n_val]
    i_test = idx[n_train + n_val :]

    return (
        ds.features[i_train],
        ds.targets[i_train],
        ds.features[i_val],
        ds.targets[i_val],
        ds.features[i_test],
        ds.targets[i_test],
    )


def _normalize(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True) + 1e-8
    return (train_x - mu) / sigma, (val_x - mu) / sigma, (test_x - mu) / sigma, mu, sigma


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def _mae_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def _r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0) + 1e-8
    return 1.0 - ss_res / ss_tot


def train_lane_model(output_dir: Path, seed: int = 17, epochs: int = 220, batch_size: int = 128, lr: float = 1e-3) -> TrainArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_supervised_lane_dataset(seed=seed)

    x_train, y_train, x_val, y_val, x_test, y_test = _split_dataset(dataset, seed=seed)
    x_train, x_val, x_test, mu, sigma = _normalize(x_train, x_val, x_test)

    model = NumpyMLP(input_dim=x_train.shape[1], hidden_dims=(64, 64), output_dim=y_train.shape[1], seed=seed)

    history = {"train_loss": [], "val_loss": []}
    n_train = x_train.shape[0]
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        order = rng.permutation(n_train)
        x_epoch = x_train[order]
        y_epoch = y_train[order]

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            xb = x_epoch[start:end]
            yb = y_epoch[start:end]

            y_pred_b, cache = model.forward(xb)
            grads = model.backward(y_pred_b, yb, cache)
            model.step_adam(grads, lr=lr)

        y_train_pred, _ = model.forward(x_train)
        y_val_pred, _ = model.forward(x_val)
        history["train_loss"].append(_mse(y_train, y_train_pred))
        history["val_loss"].append(_mse(y_val, y_val_pred))

    y_test_pred, _ = model.forward(x_test)

    rmse = _rmse_per_target(y_test, y_test_pred)
    mae = _mae_per_target(y_test, y_test_pred)
    r2 = _r2_per_target(y_test, y_test_pred)

    checkpoint = {
        "weights": {k: v.tolist() for k, v in model.params().items()},
        "feature_mean": mu.tolist(),
        "feature_std": sigma.tolist(),
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
    }
    summary = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "rmse_per_target": rmse.tolist(),
        "mae_per_target": mae.tolist(),
        "r2_per_target": r2.tolist(),
    }

    ckpt_path = output_dir / "lane_model_checkpoint.json"
    metrics_path = output_dir / "lane_model_metrics.json"
    with ckpt_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return TrainArtifacts(
        history=history,
        target_names=dataset.target_names,
        y_true_test=y_test,
        y_pred_test=y_test_pred,
        rmse_per_target=rmse,
        mae_per_target=mae,
        r2_per_target=r2,
        files=[ckpt_path, metrics_path],
    )
