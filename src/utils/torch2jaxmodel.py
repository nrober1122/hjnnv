import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch

from learned_models.beacon.estimators import MLP, CNN
from utils.mlp2jax import torch_mlp2jax


def torch_to_jax_model(model):
    """Convert a PyTorch MLP or small CNN into a pure JAX function with baked-in weights."""

    # ---------------- MLP ----------------
    if isinstance(model, MLP):
        layers = [l for l in model.net if isinstance(l, torch.nn.Linear)]
        ws, bs = [], []
        for layer in layers:
            ws.append(jnp.array(layer.weight.detach().numpy().T))  # (in, out)
            bs.append(jnp.array(layer.bias.detach().numpy()))

        def jax_model(x):
            x = jnp.asarray(x, dtype=jnp.float32)
            added_batch = False
            if x.ndim == 1:
                x = x[None, :]
                added_batch = True

            for i in range(len(ws)):
                x = jnp.dot(x, ws[i]) + bs[i]
                if i < len(ws) - 1:
                    x = jnp.maximum(x, 0.0)

            return x[0] if added_batch else x

        return jax_model

    # ---------------- CNN ----------------
    if isinstance(model, CNN):
        convs = [model.conv1, model.conv2]
        # fcs = [model.fc1, model.fc2, model.fc3]
        fcs = [model.fc1, model.fc4]  # for the smaller model version

        # Extract conv weights as OIHW (same ordering PyTorch uses)
        conv_ws, conv_bs, conv_strides = [], [], []
        for i, conv in enumerate(convs):
            # PyTorch: (out_channels, in_channels, kh, kw)
            w = np.asarray(conv.weight.detach().cpu())    # OIHW
            b = np.asarray(conv.bias.detach().cpu())      # (out,)
            conv_ws.append(jnp.array(w))                  # keep OIHW ordering
            conv_bs.append(jnp.array(b))
            # use the conv's actual stride (tuple) to be safe
            stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
            conv_strides.append(tuple(stride))

        # Extract FC weights (keep shapes so jnp.dot works with x.reshape((B,-1)))
        fc_ws, fc_bs = [], []
        for fc in fcs:
            # torch Linear weight shape: (out, in). For jnp.dot with (B, in) we want (in, out)
            fc_ws.append(jnp.array(fc.weight.detach().cpu().numpy().T))  # (in, out)
            fc_bs.append(jnp.array(fc.bias.detach().cpu().numpy()))      # (out,)

        def jax_model(x):
            # Expect x in NCHW or (C,H,W) or (H,W,C)?? We choose: keep NCHW to match torch_input
            x = jnp.asarray(x, dtype=jnp.float32)
            if x.ndim == 3:
                # If user passed (C,H,W), add batch dim to make (1,C,H,W)
                x = x[None, ...]

            # confirm NCHW layout: (B, C, H, W)
            # Apply convs using OIHW kernels and NCHW dimension_numbers
            for i, (w, b) in enumerate(zip(conv_ws, conv_bs)):
                x = lax.conv_general_dilated(
                    lhs=x,
                    rhs=w,  # OIHW
                    window_strides=conv_strides[i],
                    padding="VALID",
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                )
                # add bias: shape broadcast (1, out_ch, 1, 1)
                x = x + b.reshape(1, -1, 1, 1)
                x = jnp.maximum(x, 0.0)  # ReLU

            # Flatten for FC: (B, C, H, W) -> (B, N)
            x = x.reshape((x.shape[0], -1))

            # Apply FCs (fc_ws are (in, out))
            for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
                x = jnp.dot(x, w) + b  # (B, in) @ (in, out) + (out,)
                if i < len(fc_ws) - 1:
                    x = jnp.maximum(x, 0.0)

            return x[0] if x.shape[0] == 1 else x

        return jax_model
    # elif isinstance(model, CNN):
    #     convs = [model.conv1, model.conv2]
    #     fcs = [model.fc1, model.fc2, model.fc3]

    #     # Extract conv weights as HWIO
    #     conv_ws, conv_bs, conv_strides = [], [], []
    #     for i, conv in enumerate(convs):
    #         w = np.asarray(conv.weight.detach().cpu())
    #         b = np.asarray(conv.bias.detach().cpu())
    #         w = jnp.array(w.transpose(2, 3, 1, 0))  # HWIO
    #         b = jnp.array(b)
    #         conv_ws.append(w)
    #         conv_bs.append(b)
    #         conv_strides.append((4, 4) if i == 0 else (4, 4))  # can adjust if needed

    #     # Extract FC weights
    #     fc_ws, fc_bs = [], []
    #     for fc in fcs:
    #         fc_ws.append(jnp.array(fc.weight.detach().numpy().T))
    #         fc_bs.append(jnp.array(fc.bias.detach().numpy()))

    #     def jax_model(x):
    #         x = jnp.asarray(x, dtype=jnp.float32)
    #         if x.ndim == 3:
    #             x = x[None, ...]  # add batch dimension

    #         # Conv layers with explicit tuple padding
    #         for i, (w, b) in enumerate(zip(conv_ws, conv_bs)):
    #             pad_h = (0, w.shape[0] - 1)
    #             pad_w = (0, w.shape[1] - 1)
    #             padding = (pad_h, pad_w)

    #             # x = lax.conv_general_dilated(
    #             #     x,
    #             #     w,
    #             #     window_strides=conv_strides[i],
    #             #     padding=padding,
    #             #     dimension_numbers=("NHWC", "HWIO", "NHWC"),
    #             # )
    #             # x = x + b.reshape(1, 1, 1, -1)
    #             w = jnp.array(conv.weight.detach().cpu().numpy())  # (out, in, h, w) → OIHW
    #             b = jnp.array(conv.bias.detach().cpu().numpy())

    #             # Use dimension_numbers for NCHW:
    #             x = lax.conv_general_dilated(
    #                 x,
    #                 w,
    #                 window_strides=conv_strides[i],
    #                 padding="VALID",
    #                 dimension_numbers=("NCHW", "OIHW", "NCHW"),
    #             )
    #             x = x + b.reshape(1, -1, 1, 1)
    #             x = jnp.maximum(x, 0.0)

    #         # Flatten for FC
    #         x = x.reshape((x.shape[0], -1))
    #         for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
    #             x = jnp.dot(x, w) + b
    #             if i < len(fc_ws) - 1:
    #                 x = jnp.maximum(x, 0.0)

    #         return x[0] if x.shape[0] == 1 else x

    #     return jax_model

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


if __name__ == "__main__":
    model = CNN(input_channels=3, out_dim=4, H=128, W=128)
    model_name = "image_estimator"
    checkpoint = torch.load(
        "/home/nick/code/hjnnv/src/learned_models/beacon/estimators/"
        + model_name
        + "/best_model.pt",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    config = checkpoint.get("config_dict", None)

    from dynamic_models.beacon import BeaconDynamics
    max_input = 1.0
    max_position_disturbance = 0.1
    max_vel_disturbance = 0.01
    epsilon = 0.05
    dt = 0.1
    seed = 1
    image_dynamics = BeaconDynamics(
        dt=dt,
        max_input=max_input,
        max_position_disturbance=max_position_disturbance,
        max_vel_disturbance=max_vel_disturbance,
        range_disturbance=epsilon,
        obs_type="images",
        model_name=model_name,
        random_seed=seed
    )
    state = jnp.array([5.0, 5.0, 0.0, 0.0])
    obs = image_dynamics.get_observation(state, time=0)

    torch_input = torch.tensor(np.array(obs)).reshape((1, 3, 128, 128))

    # obs = obs.squeeze(axis=2)
    print(obs.reshape((1, 3, 128, 128)))
    jax_model = torch_to_jax_model(model)
    jax_output = jax_model(obs.reshape((3, 128, 128)))
    torch_output = model.forward(torch_input)

    print("JAX output:", jax_output)
    print("Torch output:", torch_output)

# import jax
# import jax.numpy as jnp
# from flax import linen as nn
# from learned_models.beacon.estimators import MLP, CNN

# # --- JAX equivalents ---


# import torch
# import jax
# import jax.numpy as jnp
# from typing import Tuple, Dict, Callable


# import jax
# import jax.numpy as jnp
# from jax import lax
# import numpy as np
# import torch

# def mlp2jax(model: torch.nn.Module):
#     """Convert a small MLP to a pure JAX function with baked-in weights."""
#     layers = [l for l in model.net if isinstance(l, torch.nn.Linear)]
#     ws, bs = [], []
#     for layer in layers:
#         ws.append(jnp.array(layer.weight.detach().numpy().T))  # (in, out)
#         bs.append(jnp.array(layer.bias.detach().numpy()))

#     def jax_model(x):
#         x = jnp.asarray(x, dtype=jnp.float32)
#         added_batch = False
#         if x.ndim == 1:
#             x = x[None, :]
#             added_batch = True

#         for i in range(len(ws)):
#             x = jnp.dot(x, ws[i]) + bs[i]
#             if i < len(ws) - 1:  # ReLU for all but last layer
#                 x = jnp.maximum(x, 0.0)

#         return x[0] if added_batch else x

#     return jax_model


# def cnn2jax(model: torch.nn.Module):
#     """Convert a small CNN (2 conv + MLP) to pure JAX function."""
#     convs = [model.conv1, model.conv2]
#     fcs = [model.fc1, model.fc2, model.fc3]

#     # Extract conv weights as HWIO
#     conv_ws, conv_bs = [], []
#     for conv in convs:
#         w = np.asarray(conv.weight.detach().cpu())
#         b = np.asarray(conv.bias.detach().cpu())
#         w = jnp.array(w.transpose(2, 3, 1, 0))  # HWIO
#         b = jnp.array(b)
#         conv_ws.append(w)
#         conv_bs.append(b)

#     # Extract FC weights
#     fc_ws, fc_bs = [], []
#     for fc in fcs:
#         fc_ws.append(jnp.array(fc.weight.detach().numpy().T))
#         fc_bs.append(jnp.array(fc.bias.detach().numpy()))

#     def jax_model(x):
#         x = jnp.asarray(x, dtype=jnp.float32)
#         added_batch = False
#         if x.ndim == 3:
#             x = x[None, ...]
#             added_batch = True

#         # Conv layers (NHWC)
#         for w, b in zip(conv_ws, conv_bs):
#             x = lax.conv_general_dilated(
#                 x,
#                 w,
#                 window_strides=(4, 4),
#                 padding="VALID",
#                 dimension_numbers=("NHWC", "HWIO", "NHWC"),
#             )
#             x = x + b.reshape(1, 1, 1, -1)
#             x = jnp.maximum(x, 0.0)

#         # Flatten for FC
#         x = x.reshape((x.shape[0], -1))
#         for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
#             x = jnp.dot(x, w) + b
#             if i < len(fc_ws) - 1:
#                 x = jnp.maximum(x, 0.0)

#         return x[0] if added_batch else x

#     return jax_model


# def torch_to_jax_model(model):
#     if isinstance(model, MLP):
#         return mlp2jax(model)
#     elif isinstance(model, CNN):
#         return cnn2jax(model)
#     else:
#         raise ValueError(f"Unsupported model type: {type(model)}")




# # -------------------------
# # MLP conversion
# # -------------------------
# def torch_mlp_to_jax_fn(torch_model: torch.nn.Module
#                         ) -> Tuple[Callable[[Dict, jnp.ndarray], jnp.ndarray], Dict]:
#     """
#     Convert a PyTorch MLP to a pure JAX function and parameter dict.
    
#     Returns:
#         forward_fn(params, x)
#         params dict
#     """
#     params = {}
#     linear_layers = [l for l in torch_model.net if isinstance(l, torch.nn.Linear)]
    
#     for i, layer in enumerate(linear_layers):
#         # PyTorch Linear: weight shape (out_features, in_features)
#         params[f"w{i}"] = jnp.array(layer.weight.detach().numpy().T)  # transpose to (in, out)
#         params[f"b{i}"] = jnp.array(layer.bias.detach().numpy())
    
#     def forward_fn(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
#         x = jnp.dot(x, params["w0"]) + params["b0"]
#         x = jax.nn.relu(x)
#         x = jnp.dot(x, params["w1"]) + params["b1"]
#         x = jax.nn.relu(x)
#         x = jnp.dot(x, params["w2"]) + params["b2"]
#         x = jax.nn.relu(x)
#         x = jnp.dot(x, params["w3"]) + params["b3"]
#         return x

#     return forward_fn, params

# # -------------------------
# # CNN conversion
# # -------------------------
# def torch_cnn_to_jax_fn(torch_model: torch.nn.Module
#                         ) -> Tuple[Callable[[Dict, jnp.ndarray], jnp.ndarray], Dict]:
#     """
#     Convert a PyTorch CNN to a pure JAX function + params dict.
#     Works with small CNNs like your example (no padding, stride convs).
#     """
#     params = {}

#     # Conv layers
#     conv_layers = [torch_model.conv1, torch_model.conv2]
#     for i, conv in enumerate(conv_layers):
#         # PyTorch Conv2d weight: (out_channels, in_channels, kh, kw)
#         params[f"conv_w{i}"] = jnp.array(conv.weight.detach().numpy())
#         params[f"conv_b{i}"] = jnp.array(conv.bias.detach().numpy())

#     # Fully connected layers
#     fc_layers = [torch_model.fc1, torch_model.fc2, torch_model.fc3]
#     for i, fc in enumerate(fc_layers):
#         params[f"fc_w{i}"] = jnp.array(fc.weight.detach().numpy().T)
#         params[f"fc_b{i}"] = jnp.array(fc.bias.detach().numpy())

#     def forward_fn(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
#         # x shape: (B, C, H, W)
#         # Conv1
#         x = jax.lax.conv_general_dilated(
#             lhs=x,
#             rhs=params["conv_w0"],
#             window_strides=(4, 4),
#             padding="VALID",
#             dimension_numbers=("NCHW", "OIHW", "NCHW"),
#         )
#         x = x + params["conv_b0"][None, :, None, None]
#         x = jax.nn.relu(x)

#         # Conv2
#         x = jax.lax.conv_general_dilated(
#             lhs=x,
#             rhs=params["conv_w1"],
#             window_strides=(4, 4),
#             padding="VALID",
#             dimension_numbers=("NCHW", "OIHW", "NCHW"),
#         )
#         x = x + params["conv_b1"][None, :, None, None]
#         x = jax.nn.relu(x)

#         # Flatten
#         x = x.reshape((x.shape[0], -1))

#         # Fully connected layers
#         x = jnp.dot(x, params["fc_w0"]) + params["fc_b0"]
#         x = jax.nn.relu(x)
#         x = jnp.dot(x, params["fc_w1"]) + params["fc_b1"]
#         x = jax.nn.relu(x)
#         x = jnp.dot(x, params["fc_w2"]) + params["fc_b2"]
#         return x

#     return forward_fn, params

# # -------------------------
# # Wrapper utility
# # -------------------------
# def torch_to_jax_model(torch_model):
#     if isinstance(torch_model, MLP):
#         return torch_mlp_to_jax_fn(torch_model)
#     elif isinstance(torch_model, CNN):
#         return torch_cnn_to_jax_fn(torch_model)
#     else:
#         raise ValueError(f"Unsupported model type: {type(torch_model)}")
