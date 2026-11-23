"""Multi-input GRU stroke classifier.

    acc  x,y,z  ─┐
    gyro x,y,z  ─┴─> v_t (6-vector per time step) ─> GRU ─> FC ─> softmax ─> class probabilities

Each 6-channel IMU sample is one GRU input vector; the final hidden state goes through a fully
connected layer to class logits (softmax applied at inference, cross-entropy on logits in training).

Inputs are scaled by the sensor full-scale range (accel / 16 g, gyro / 2000 deg/s) rather than dataset
statistics, so the same model can run on the ICM-20948 stream from firmware/firmware.ino
(+-16 g, +-2000 deg/s; note that firmware reports accel in g, TennisEye in m/s^2 -> see `scale_imu`).

    python ml/stroke_gru.py                  # train on all TennisEye shots, save ml/checkpoints/stroke_gru.pt
    python ml/stroke_gru.py --epochs 80 --hidden 96
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "tenniseye"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"

G = 9.80665
ACC_FS_MS2 = 16 * G          # +-16 g full scale, in m/s^2
GYRO_FS_DPS = 2000.0         # +-2000 deg/s full scale

CLASSES = ["serve", "forehand", "backhand", "volley"]
STROKE_TO_CLASS = {"serve": 0, "forehand": 1, "backhand": 2, "forehand_volley": 3, "backhand_volley": 3}


# ---------------------------------------------------------------- data

def scale_imu(x: np.ndarray, acc_unit: str = "ms2") -> np.ndarray:
    """(..., 6) raw [ax ay az gx gy gz] -> full-scale normalised (roughly [-1, 1]).
    acc_unit: 'ms2' (TennisEye) or 'g' (our firmware)."""
    x = np.asarray(x, dtype=np.float32).copy()
    acc_fs = ACC_FS_MS2 if acc_unit == "ms2" else 16.0
    x[..., :3] /= acc_fs
    x[..., 3:] /= GYRO_FS_DPS
    return x


def load_tenniseye(data_dir: Path = DATA_DIR):
    """Returns X (N, 100, 6) scaled float32, y (N,) int64 class ids, groups (N,) player ids, shots DataFrame."""
    shots = pd.read_parquet(data_dir / "tenniseye_shots.parquet").sort_values("shot_id").reset_index(drop=True)
    samples = pd.read_parquet(data_dir / "tenniseye_samples.parquet").sort_values(["shot_id", "sample"])
    raw = samples[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy().reshape(len(shots), -1, 6)
    X = scale_imu(raw, "ms2")
    y = shots.stroke.astype(str).map(STROKE_TO_CLASS).to_numpy().astype(np.int64)
    return X, y, shots.player_id.to_numpy(), shots


# ---------------------------------------------------------------- model

class StrokeGRU(nn.Module):
    def __init__(self, n_inputs: int = 6, hidden: int = 64, layers: int = 2, n_classes: int = len(CLASSES),
                 dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(n_inputs, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, 6) -> logits (batch, n_classes)."""
        _, h = self.gru(x)
        return self.head(h[-1])

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=-1)


# ---------------------------------------------------------------- augmentation

def random_rotations(n: int, max_deg: float, device) -> torch.Tensor:
    """n random rotation matrices with angle ~ U(0, max_deg) about a random axis (Rodrigues)."""
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=1)
    ang = torch.rand(n, device=device) * math.radians(max_deg)
    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(1, 2)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return torch.eye(3, device=device) + s * K + (1 - c) * (K @ K)


def augment(x: torch.Tensor, rot_deg: float = 15.0, shift: int = 8, scale: float = 0.1,
            noise: float = 0.01) -> torch.Tensor:
    """Sensor-mounting rotation (same R for acc and gyro), time shift, amplitude scale, noise."""
    n, t, _ = x.shape
    R = random_rotations(n, rot_deg, x.device)
    acc = torch.einsum("nij,ntj->nti", R, x[..., :3])
    gyr = torch.einsum("nij,ntj->nti", R, x[..., 3:])
    x = torch.cat([acc, gyr], dim=-1)
    # time shift with edge padding
    offs = torch.randint(-shift, shift + 1, (n,), device=x.device)
    idx = (torch.arange(t, device=x.device)[None, :] - offs[:, None]).clamp(0, t - 1)
    x = torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))
    x = x * (1 + scale * (2 * torch.rand(n, 1, 1, device=x.device) - 1))
    return x + noise * torch.randn_like(x)


# ---------------------------------------------------------------- training

@dataclass
class TrainConfig:
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    epochs: int = 60
    batch: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-3
    augment: bool = True
    seed: int = 0


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))          # sqrt-inverse frequency: helps rare volleys
    return torch.tensor(w / w.mean())


def train_model(X: np.ndarray, y: np.ndarray, cfg: TrainConfig = TrainConfig(), device: str | None = None,
                verbose: bool = False) -> StrokeGRU:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = StrokeGRU(hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout).to(device)
    Xt, yt = torch.tensor(X, device=device), torch.tensor(y, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(X) / cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, len(CLASSES)).to(device))

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        total = 0.0
        for i in range(0, len(perm), cfg.batch):
            b = perm[i:i + cfg.batch]
            xb = augment(Xt[b]) if cfg.augment else Xt[b]
            loss = loss_fn(model(xb), yt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:3d}  loss {total / len(Xt):.4f}")
    return model.eval()


@torch.no_grad()
def predict(model: StrokeGRU, X: np.ndarray, device: str | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    return model.predict_proba(torch.tensor(X, device=device)).cpu().numpy()


def save(model: StrokeGRU, cfg: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "classes": CLASSES,
                "input": "[ax ay az]/16g, [gx gy gz]/2000dps, 100 Hz, 100 samples centred on impact"}, path)


def load(path: Path, device: str = "cpu") -> StrokeGRU:
    ck = torch.load(path, map_location=device)
    cfg = TrainConfig(**ck["config"])
    model = StrokeGRU(hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f, v in asdict(TrainConfig()).items():
        if isinstance(v, bool):
            ap.add_argument(f"--no-{f}", dest=f, action="store_false")
        else:
            ap.add_argument(f"--{f.replace('_', '-')}", dest=f, type=type(v), default=v)
    ap.add_argument("--out", type=Path, default=CKPT_DIR / "stroke_gru.pt")
    args = ap.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})

    X, y, _, _ = load_tenniseye()
    print(f"Training on {len(X)} shots: " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, np.bincount(y))))
    model = train_model(X, y, cfg, verbose=True)
    acc = (predict(model, X).argmax(1) == y).mean()
    save(model, cfg, args.out)
    print(f"Train accuracy {acc:.1%}. Saved {args.out}")


if __name__ == "__main__":
    main()
