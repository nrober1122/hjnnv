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
import json
from datetime import datetime

from learned_models.beacon.estimators import MLP, CNN
from dynamic_models.beacon import BeaconDynamics
import jax

# -------------------------------
# Config
# -------------------------------
@dataclass
class Config:
    seed: int = 123
    dt: float = 0.1
    T: int = 60
    n_traj_train: int = 4000
    n_traj_val: int = 800
    n_traj_test: int = 800
    x_range: Tuple[float, float] = (-1.0, 11.0)
    y_range: Tuple[float, float] = (-1.0, 11.0)
    vx_range: Tuple[float, float] = (-3.0, 3.0)
    vy_range: Tuple[float, float] = (-3.0, 3.0)
    sigma_process_pos: float = 0.01
    sigma_process_vel: float = 0.005
    sigma_range: float = 0.01
    
    beacons: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (10.0, 10.0)
    )
    image_size: Tuple[int, int] = (16, 64)  # (H,W)
    lm_h: float = 1.618 * 3
    lm_w: float = 1.0 * 0.5
    focal_length: float = 0.05
    max_theta: float = math.pi / 2

    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    hidden: int = 256
    stride: int = 4 # for CNN
    obs_mode: str = "images"   # "distances" or "images"
    data_dir: str = "/home/nrober/code/hjnnv/hjnnv/data/scratch/beacons/beacons_image_data"
    results_dir: str = "/home/nrober/code/hjnnv/hjnnv/src/learned_models/beacon/estimators/image_estimator_32x64_nofade/scratch"


cfg = Config()


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


# -------------------------------
# Dataset for image sequences
# -------------------------------
class ImageHistoryDataset(Dataset):
    """Input = [img_{t-2}, img_{t-1}, img_t], shape (3,H,W,1). Output = state (4,)"""
    def __init__(self, cfg: Config, n_traj: int, image_size=(128, 128)):
        self.inputs, self.targets = [], []
        H, W = image_size
        dyn = BeaconDynamics(
            dt=cfg.dt,
            max_input=1.0,
            max_position_disturbance=cfg.sigma_process_pos,
            max_vel_disturbance=cfg.sigma_process_vel,
            range_disturbance=cfg.sigma_range,
            obs_type=cfg.obs_mode,
            model_name="image_estimator",
            random_seed=cfg.seed
        )
        dyn.beacons = np.array(cfg.beacons)

        for k in range(n_traj):
            print(f"Generating images for trajectory {k+1}/{n_traj}...")
            states, _ = simulate_trajectory(cfg)
            # generate image sequence for each timestep
            for t in range(2, cfg.T):
                imgs = []
                for tau in [t-2, t-1, t]:
                    subkey = jax.random.PRNGKey(random.randint(0, int(1e6)))
                    img = np.array(dyn.observe_image(states[tau], image_width=W, image_height=H, lm_h=cfg.lm_h, lm_w=cfg.lm_w, subkey=subkey, focal_length=cfg.focal_length, max_theta=cfg.max_theta))
                    imgs.append(img)  # (H,W,1)
                inp = np.concatenate(imgs, axis=-1)  # (H,W,3)
                self.inputs.append(inp.astype(np.float32) / 1.0)  # keep [0,1]
                self.targets.append(states[t].astype(np.float32))

        self.inputs = np.stack(self.inputs, axis=0)
        self.targets = np.stack(self.targets, axis=0)

    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]


# -------------------------------
# Data loaders
# -------------------------------
def make_loaders(cfg: Config):
    if cfg.obs_mode == "distances":
        train_ds = RangeHistoryDataset(cfg, cfg.n_traj_train)
        in_mean, in_std = train_ds.in_mean, train_ds.in_std
        val_ds = RangeHistoryDataset(cfg, cfg.n_traj_val, in_mean, in_std)
        test_ds = RangeHistoryDataset(cfg, cfg.n_traj_test, in_mean, in_std)
    else:  # images
        H, W = cfg.image_size
        train_ds = ImageHistoryDataset(cfg, cfg.n_traj_train, image_size=(H, W))
        val_ds = ImageHistoryDataset(cfg, cfg.n_traj_val, image_size=(H, W))
        test_ds = ImageHistoryDataset(cfg, cfg.n_traj_test, image_size=(H, W))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds


# -------------------------------
# Generic NumpyDataset
# -------------------------------
class NumpyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

# -------------------------------
# Data loading / generation helper
# -------------------------------
def get_dataloaders(cfg, make_loaders_fn):
    os.makedirs(cfg.data_dir, exist_ok=True)
    metadata_path = os.path.join(cfg.data_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        print("Generating new dataset...")
        train_loader, val_loader, test_loader, train_ds = make_loaders_fn(cfg)

        # Save arrays
        np.save(os.path.join(cfg.data_dir, "train_inputs.npy"), train_ds.inputs)
        np.save(os.path.join(cfg.data_dir, "train_targets.npy"), train_ds.targets)
        np.save(os.path.join(cfg.data_dir, "val_inputs.npy"), val_loader.dataset.inputs)
        np.save(os.path.join(cfg.data_dir, "val_targets.npy"), val_loader.dataset.targets)
        np.save(os.path.join(cfg.data_dir, "test_inputs.npy"), test_loader.dataset.inputs)
        np.save(os.path.join(cfg.data_dir, "test_targets.npy"), test_loader.dataset.targets)

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(vars(cfg), f, indent=2)

        return train_loader, val_loader, test_loader, train_ds

    else:
        print("Loading dataset from disk...")
        train_inputs = np.load(os.path.join(cfg.data_dir, "train_inputs.npy"))
        train_targets = np.load(os.path.join(cfg.data_dir, "train_targets.npy"))
        val_inputs = np.load(os.path.join(cfg.data_dir, "val_inputs.npy"))
        val_targets = np.load(os.path.join(cfg.data_dir, "val_targets.npy"))
        test_inputs = np.load(os.path.join(cfg.data_dir, "test_inputs.npy"))
        test_targets = np.load(os.path.join(cfg.data_dir, "test_targets.npy"))

        train_ds = NumpyDataset(train_inputs, train_targets)
        val_ds = NumpyDataset(val_inputs, val_targets)
        test_ds = NumpyDataset(test_inputs, test_targets)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        return train_loader, val_loader, test_loader, train_ds

# -------------------------------
# Training
# -------------------------------
def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use cleaner helper
    train_loader, val_loader, test_loader, train_ds = get_dataloaders(cfg, make_loaders)

    # Model
    if cfg.obs_mode == "distances":
        model = MLP(in_dim=12, out_dim=4, hidden=cfg.hidden, stride=cfg.stride).to(device)
    else:
        C = train_ds.inputs.shape[-1]   # should be 3
        H, W = train_ds.inputs.shape[1:3]
        model = CNN(input_channels=C, out_dim=4, hidden1=cfg.hidden, hidden2=cfg.hidden, H=H, W=W).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    print("Training...")

    def run(loader):
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if cfg.obs_mode == "images":
                    x = x.permute(0, 3, 1, 2)
                yhat = model(x)
                se += ((yhat - y)**2).sum(dim=0).cpu().numpy()
                n += y.shape[0]
        rmse = np.sqrt(se / n)
        return rmse

    best_val, best_state = float("inf"), None
    for epoch in range(1, cfg.epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if cfg.obs_mode == "images":
                x = x.permute(0, 3, 1, 2)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
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

    os.makedirs(cfg.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(cfg.results_dir, f"{cfg.obs_mode}_estimator_{timestamp}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config_dict": cfg.__dict__,
    }, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train(cfg)



# # -------------------------------
# # Training
# # -------------------------------
# def train(cfg: Config):
#     print("Generating Data...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if cfg.obs_mode == "images":
#         if not os.listdir(cfg.data_dir):
#             # raise RuntimeError(f"Data directory {cfg.data_dir} is empty. Please generate image data before training.")
#             train_loader, val_loader, test_loader, train_ds = make_loaders(cfg)
#             os.makedirs(cfg.data_dir, exist_ok=True)
#             np.save(os.path.join(cfg.data_dir, "train_inputs.npy"), train_ds.inputs)
#             np.save(os.path.join(cfg.data_dir, "train_targets.npy"), train_ds.targets)
#             np.save(os.path.join(cfg.data_dir, "val_inputs.npy"), val_loader.dataset.inputs)
#             np.save(os.path.join(cfg.data_dir, "val_targets.npy"), val_loader.dataset.targets)
#             np.save(os.path.join(cfg.data_dir, "test_inputs.npy"), test_loader.dataset.inputs)
#             np.save(os.path.join(cfg.data_dir, "test_targets.npy"), test_loader.dataset.targets)
#     else:
#         train_inputs = np.load(os.path.join(cfg.data_dir, "train_inputs.npy"))
#         train_targets = np.load(os.path.join(cfg.data_dir, "train_targets.npy"))
#         val_inputs = np.load(os.path.join(cfg.data_dir, "val_inputs.npy"))
#         val_targets = np.load(os.path.join(cfg.data_dir, "val_targets.npy"))
#         test_inputs = np.load(os.path.join(cfg.data_dir, "test_inputs.npy"))
#         test_targets = np.load(os.path.join(cfg.data_dir, "test_targets.npy"))

#         class NumpyDataset(Dataset):
#             def __init__(self, inputs, targets):
#                 self.inputs = inputs
#                 self.targets = targets
#             def __len__(self):
#                 return self.inputs.shape[0]
#             def __getitem__(self, idx):
#                 return self.inputs[idx], self.targets[idx]

#         train_ds = NumpyDataset(train_inputs, train_targets)
#         val_ds = NumpyDataset(val_inputs, val_targets)
#         test_ds = NumpyDataset(test_inputs, test_targets)
#         train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
#         val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
#         test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
#         # train_loader, val_loader, test_loader, train_ds = make_loaders(cfg)

#     if cfg.obs_mode == "distance":
#         model = MLP(in_dim=12, out_dim=4, hidden=cfg.hidden).to(device)
#     else:
#         C = train_ds.inputs.shape[-1]   # should be 3
#         H, W = train_ds.inputs.shape[1:3]
#         model = CNN(input_channels=C, out_dim=4, H=H, W=W).to(device)

#     opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
#     loss_fn = nn.MSELoss()

#     print("Training...")

#     def run(loader):
#         model.eval()
#         se, n = 0.0, 0
#         with torch.no_grad():
#             for x, y in loader:
#                 x, y = x.to(device), y.to(device)
#                 if cfg.obs_mode == "images":
#                     x = x.permute(0, 3, 1, 2)  # (B,C,H,W)
#                 yhat = model(x)
#                 se += ((yhat - y)**2).sum(dim=0).cpu().numpy()
#                 n += y.shape[0]
#         rmse = np.sqrt(se / n)
#         return rmse

#     best_val, best_state = float("inf"), None

#     for epoch in range(1, cfg.epochs+1):
#         model.train()
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#             if cfg.obs_mode == "images":
#                 x = x.permute(0, 3, 1, 2)
#             opt.zero_grad()
#             yhat = model(x)
#             loss = loss_fn(yhat, y)
#             loss.backward()
#             opt.step()
#         val_rmse = run(val_loader)
#         val_pos_rmse = float(np.linalg.norm(val_rmse[:2]))
#         if val_pos_rmse < best_val:
#             best_val = val_pos_rmse
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#         if epoch % 5 == 0 or epoch == 1:
#             print(f"Epoch {epoch:02d} | val RMSE [x,y,vx,vy] = {val_rmse}")

#     if best_state is not None:
#         model.load_state_dict(best_state)

#     test_rmse = run(test_loader)
#     print("\nTest RMSE per dim [x,y,vx,vy]:", test_rmse)

#     os.makedirs(cfg.results_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path = os.path.join(cfg.results_dir, f"{cfg.obs_mode}_estimator_{timestamp}.pt")
#     torch.save({
#         "model_state_dict": model.state_dict(),
#         "config_dict": cfg.__dict__,
#     }, model_path)
#     print(f"Saved model to {model_path}")


# if __name__ == "__main__":
#     train(cfg)
