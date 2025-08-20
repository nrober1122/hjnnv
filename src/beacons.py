"""
Toy problem: learn to estimate 2D position & velocity from short histories of range-only measurements to fixed beacons.

State (at time t): s_t = [x_t, y_t, vx_t, vy_t]
Dynamics: constant-velocity with small process noise.
Observations: ranges to 3 beacons, with Gaussian noise.
Input to NN: concatenated ranges at times t-1 and t  -> 6-D observation vector
Output from NN: [x_t, y_t, vx_t, vy_t]  -> 4-D state

This setup is:
- Low-dimensional (6→4) and easy to sample.
- Nonlinear/invertible enough to be interesting (range geometry), yet trivial to simulate.
- Verifiable: you can grid-sweep (x,y) and check predicted states, or compare with an EKF/UKF later.

Run this file directly to train a small MLP and print test-set errors.
"""

from dataclasses import dataclass
import math
import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    seed: int = 123
    dt: float = 0.1
    T: int = 60                   # steps per trajectory
    n_traj_train: int = 1500
    n_traj_val: int = 200
    n_traj_test: int = 200

    # state init bounds
    x_range: Tuple[float, float] = (-5.0, 5.0)
    y_range: Tuple[float, float] = (-5.0, 5.0)
    vx_range: Tuple[float, float] = (-1.0, 1.0)
    vy_range: Tuple[float, float] = (-1.0, 1.0)

    # process & measurement noise
    sigma_process_pos: float = 0.01
    sigma_process_vel: float = 0.005
    sigma_range: float = 0.02     # range noise

    # beacon positions (non-collinear for observability)
    beacons: Tuple[Tuple[float, float], ...] = ((0.0, 0.0), (6.0, 0.0), (0.0, 6.0))

    # training
    batch_size: int = 256
    epochs: int = 25
    lr: float = 1e-3
    hidden: int = 128

cfg = Config()

# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)


def simulate_trajectory(cfg: Config):
    dt = cfg.dt
    T = cfg.T
    # sample initial state
    x = np.random.uniform(*cfg.x_range)
    y = np.random.uniform(*cfg.y_range)
    vx = np.random.uniform(*cfg.vx_range)
    vy = np.random.uniform(*cfg.vy_range)

    xs, ys, vxs, vys = [], [], [], []
    # simple constant-velocity model with tiny process noise
    for _ in range(T):
        # process noise
        x += vx * dt + np.random.randn() * cfg.sigma_process_pos
        y += vy * dt + np.random.randn() * cfg.sigma_process_pos
        vx += np.random.randn() * cfg.sigma_process_vel
        vy += np.random.randn() * cfg.sigma_process_vel
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)

    states = np.stack([xs, ys, vxs, vys], axis=1)  # (T,4)

    # observations: ranges to beacons
    beacons = np.array(cfg.beacons)  # (3,2)
    ranges = []
    for t in range(T):
        pos = np.array([states[t,0], states[t,1]])
        d = np.linalg.norm(beacons - pos, axis=1)  # (3,)
        d_noisy = d + np.random.randn(3) * cfg.sigma_range
        ranges.append(d_noisy)
    ranges = np.stack(ranges, axis=0)  # (T,3)

    return states, ranges


class RangeHistoryDataset(Dataset):
    """Supervised pairs: input = [r_{t-1}, r_t] (6-D), target = state s_t (4-D)."""
    def __init__(self, cfg: Config, n_traj: int):
        self.inputs = []
        self.targets = []
        for _ in range(n_traj):
            s, r = simulate_trajectory(cfg)
            for t in range(1, cfg.T):
                inp = np.concatenate([r[t-1], r[t]], axis=0)  # (6,)
                tgt = s[t]  # (4,)
                self.inputs.append(inp)
                self.targets.append(tgt)
        self.inputs = np.stack(self.inputs, axis=0).astype(np.float32)
        self.targets = np.stack(self.targets, axis=0).astype(np.float32)

        # standardize inputs for stability
        self.in_mean = self.inputs.mean(axis=0, keepdims=True)
        self.in_std = self.inputs.std(axis=0, keepdims=True) + 1e-6
        self.inputs = (self.inputs - self.in_mean) / self.in_std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MLP(nn.Module):
    def __init__(self, in_dim=6, out_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def make_loaders(cfg: Config):
    train_ds = RangeHistoryDataset(cfg, cfg.n_traj_train)
    val_ds = RangeHistoryDataset(cfg, cfg.n_traj_val)
    test_ds = RangeHistoryDataset(cfg, cfg.n_traj_test)

    train_loader = DataLoader(train_ds, batch_size=cgf.batch_size if (cgf:=cfg) else cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds


def train(cfg: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, train_ds = make_loaders(cfg)

    model = MLP(in_dim=6, out_dim=4, hidden=cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.MSELoss()

    def run(loader):
        model.eval()
        se = 0.0
        n = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)
                se += ((yhat - y)**2).sum(dim=0).cpu().numpy()
                n += y.shape[0]
        mse = se / n
        rmse = np.sqrt(mse)
        return rmse  # per-dimension

    best_val = float('inf')
    best_state = None

    for epoch in range(1, cfg.epochs+1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
        sched.step()
        val_rmse = run(val_loader)
        val_pos_rmse = float(np.linalg.norm(val_rmse[:2]))
        if val_pos_rmse < best_val:
            best_val = val_pos_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | val RMSE [x,y,vx,vy] = {val_rmse}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse = run(test_loader)
    print("\nTest RMSE per dim [x,y,vx,vy]:", test_rmse)
    print(f"Position RMSE (L2 over x,y): {float(np.linalg.norm(test_rmse[:2])):.4f}")
    print(f"Velocity RMSE (L2 over vx,vy): {float(np.linalg.norm(test_rmse[2:])):.4f}")

    


if __name__ == "__main__":
    train(cfg)
