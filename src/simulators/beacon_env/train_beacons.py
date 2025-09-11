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

import ipdb
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from learned_models.beacon.estimators import MLP


# def nnv_loss(x: torch.Tensor,
#              pred: torch.Tensor,
#              target: torch.Tensor,
#              bounded_model: BoundedModule,
#              epsilon=0.2
#              ) -> torch.Tensor:
#     # Compute the standard MSE loss
#     mse_loss = nn.MSELoss()(pred, target)

#     # Compute the certified bounds
#     bounded_x = BoundedTensor(x, PerturbationLpNorm(norm=np.inf, eps=epsilon))
#     # ipdb.set_trace()
#     bounded_pred = bounded_model.compute_bounds(x=(bounded_x,), method="CROWN")

#     # Compute the N-dimensional volume of the certified bounds box for the target
#     lower, upper = bounded_pred
#     # Volume of a box in N dimensions: product of (upper_i - lower_i) for each dimension
#     cert_loss = torch.mean(torch.prod(upper-lower, axis=1))

#     # ipdb.set_trace()

#     # target_bounds = BoundedTensor(target, PerturbationLpNorm(2, eps=epsilon))
#     # cert_loss = BoundedModule(pred_bounds, target_bounds)

#     return mse_loss + 0.1*cert_loss

def nnv_loss(x: torch.Tensor,
             pred: torch.Tensor,
             target: torch.Tensor,
             bounded_model: BoundedModule,
             epsilon: float = 0.2,
             beta: float = 0.1) -> torch.Tensor:
    # Standard MSE
    mse_loss = nn.MSELoss()(pred, target)
    cert_loss = 0.0

    if beta > 0:
        # Certified bounds under perturbations
        bounded_x = BoundedTensor(x, PerturbationLpNorm(norm=np.inf, eps=epsilon))
        lower, upper = bounded_model.compute_bounds(x=(bounded_x,), method="CROWN")

        # Penalize bound width per dimension (mean width)
        widths = upper - lower  # (batch, dim)
        cert_loss = torch.mean(torch.mean(widths, dim=1))  

        # Alternatively: log-volume
        # cert_loss = torch.mean(torch.sum(torch.log(widths + 1e-12), dim=1))

    return mse_loss + beta * cert_loss

def beta_schedule_linear(epoch, max_beta=0.1, ramp_epochs=50, start_epoch=0):
    if epoch < start_epoch:
        return 0.0
    # shift epoch so ramp starts at start_epoch
    ramp_progress = (epoch - start_epoch) / ramp_epochs
    return min(max_beta, max_beta * ramp_progress)

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    seed: int = 123
    dt: float = 0.1
    T: int = 60                   # steps per trajectory
    n_traj_train: int = 15000
    n_traj_val: int = 3000
    n_traj_test: int = 3000

    # state init bounds
    x_range: Tuple[float, float] = (-3.0, 12.0)
    y_range: Tuple[float, float] = (-3.0, 12.0)
    vx_range: Tuple[float, float] = (-3.0, 3.0)
    vy_range: Tuple[float, float] = (-3.0, 3.0)

    # process & measurement noise
    sigma_process_pos: float = 0.01
    sigma_process_vel: float = 0.005
    sigma_range: float = 0.1     # range noise

    # beacon positions (non-collinear for observability)
    # beacons: Tuple[Tuple[float, float], ...] = ((0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (4.0, 4.0))
    beacons: Tuple[Tuple[float, float], ...] = ((0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (10.0, 10.0))

    # training
    batch_size: int = 256
    epochs: int = 75
    lr: float = 1e-3
    hidden: int = 256

    # loss_fn: callable = nnv_loss
    epsilon: float = 0.1

    results_dir = "/home/nick/code/hjnnv/src/learned_models/" \
        "beacon/estimators/image_estimator_test/scratch"


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
        pos = np.array([states[t, 0], states[t, 1]])
        d = np.linalg.norm(beacons - pos, axis=1)  # (3,)
        d_noisy = d + np.random.randn(len(cfg.beacons)) * cfg.sigma_range
        ranges.append(d_noisy)
    ranges = np.stack(ranges, axis=0)  # (T,3)

    return states, ranges


# class RangeHistoryDataset(Dataset):
#     """Supervised pairs: input = [r_{t-1}, r_t] (6-D), target = state s_t (4-D)."""
#     def __init__(self, cfg: Config, n_traj: int):
#         self.inputs = []
#         self.targets = []
#         for _ in range(n_traj):
#             s, r = simulate_trajectory(cfg)
#             for t in range(2, cfg.T):
#                 inp = np.concatenate([r[t-2], r[t-1], r[t]], axis=0)  # (12,)
#                 tgt = s[t]  # (4,)
#                 self.inputs.append(inp)
#                 self.targets.append(tgt)
#         self.inputs = np.stack(self.inputs, axis=0).astype(np.float32)
#         self.targets = np.stack(self.targets, axis=0).astype(np.float32)

#         # standardize inputs for stability
#         self.in_mean = self.inputs.mean(axis=0, keepdims=True)
#         self.in_std = self.inputs.std(axis=0, keepdims=True) + 1e-6
#         self.inputs = (self.inputs - self.in_mean) / self.in_std

#     def __len__(self):
#         return self.inputs.shape[0]

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets[idx]


# def make_loaders(cfg: Config):
#     train_ds = RangeHistoryDataset(cfg, cfg.n_traj_train)
#     val_ds = RangeHistoryDataset(cfg, cfg.n_traj_val)
#     test_ds = RangeHistoryDataset(cfg, cfg.n_traj_test)

#     train_loader = DataLoader(train_ds, batch_size=cgf.batch_size if (cgf:=cfg) else cfg.batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

#     return train_loader, val_loader, test_loader, train_ds

class RangeHistoryDataset(Dataset):
    """Supervised pairs: input = [r_{t-2}, r_{t-1}, r_t] (12-D), target = state s_t (4-D)."""
    def __init__(self, cfg: Config, n_traj: int, in_mean=None, in_std=None):
        self.inputs = []
        self.targets = []
        for _ in range(n_traj):
            s, r = simulate_trajectory(cfg)
            for t in range(2, cfg.T):
                inp = np.concatenate([r[t-2], r[t-1], r[t]], axis=0)  # (12,)
                tgt = s[t]  # (4,)
                self.inputs.append(inp)
                self.targets.append(tgt)

        self.inputs = np.stack(self.inputs, axis=0).astype(np.float32)
        self.targets = np.stack(self.targets, axis=0).astype(np.float32)

        if in_mean is None or in_std is None:
            # If no stats provided, compute them (used only for training set)
            self.in_mean = self.inputs.mean(axis=0, keepdims=True)
            self.in_std = self.inputs.std(axis=0, keepdims=True) + 1e-6
        else:
            # Use provided stats (for val/test sets)
            self.in_mean = in_mean
            self.in_std = in_std

        # Apply normalization
        self.inputs = (self.inputs - self.in_mean) / self.in_std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_loaders(cfg: Config):
    # Build training set and compute normalization stats
    train_ds = RangeHistoryDataset(cfg, cfg.n_traj_train)
    in_mean, in_std = train_ds.in_mean, train_ds.in_std

    # Apply *same* normalization stats to val/test sets
    val_ds = RangeHistoryDataset(cfg, cfg.n_traj_val, in_mean, in_std)
    test_ds = RangeHistoryDataset(cfg, cfg.n_traj_test, in_mean, in_std)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds


def train(cfg: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, train_ds = make_loaders(cfg)

    model = MLP(in_dim=12, out_dim=4, hidden=cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.MSELoss()
    # loss_fn = nnv_loss

    # bound_opts: dict = {
    #     'relu': "CROWN-IBP",
    #     # 'relu': "IBP",
    #     'sparse_intermediate_bounds': False,
    #     'sparse_conv_intermediate_bounds': False,
    #     'sparse_intermediate_bounds_with_ibp': False,
    #     'sparse_features_alpha': False,
    #     'sparse_spec_alpha': False,
    #     'reduce_layers': False,
    #     # 'zero-lb': True,
    #     # 'same-slope': True,
    # }
    # dummy_input = torch.randn(1, 12).to(device)
    # bounded_model = BoundedModule(
    #     model,
    #     dummy_input,
    #     bound_opts=bound_opts,
    #     device=device
    # )

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
        beta = beta_schedule_linear(epoch, max_beta=0.05, ramp_epochs=60, start_epoch=20)
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            yhat = model(x)
            # loss = loss_fn(x, yhat, y, bounded_model, beta=beta)
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

    def get_obs(state):
        pos = np.array([state[0], state[1]])
        d = np.linalg.norm(cfg.beacons - pos, axis=1)  # (3,)
        d_noisy = d + np.random.randn(len(cfg.beacons)) * cfg.sigma_range
        return d_noisy
    
    # state0 = test_loader.dataset.targets[0]
    # state1 = np.array([state0[0] + state0[2]*cfg.dt, state0[1] + state0[3]*cfg.dt, state0[2], state0[3]])
    # obs0 = get_obs(state0)
    # obs1 = get_obs(state1)
    # obs = np.concatenate([obs0, obs1], axis=0, dtype='float32')
    # state1_est = model(torch.tensor(obs).to(device)).detach().cpu().numpy()

    # ipdb.set_trace()

    os.makedirs(cfg.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(cfg.results_dir, f"mlp_{timestamp}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_mean': train_ds.in_mean,
        'in_std': train_ds.in_std,
        'config_dict': cfg.__dict__,
    }, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train(cfg)
