import jax

import simulators
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch

from learned_models.beacon.estimators import MLP, CNN
from simulators.NASA_ULI_Xplane_Simulator.src.tiny_taxinet_train.model_tiny_taxinet import TinyTaxiNetDNN
from simulators.NASA_ULI_Xplane_Simulator.src.train_DNN.model_taxinet import TaxiNetCNN
from utils.mlp2jax import torch_mlp2jax



import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import torch
import torch.nn as nn


def torch_to_jax_model(model):
    """Convert a PyTorch MLP, TinyTaxiNetDNN, CNN, or TaxiNetCNN into a pure JAX function."""
    
    def convert_mlp(linear_layers):
        # Extract weights and biases from PyTorch layers as pure JAX arrays
        ws = [jnp.asarray(layer.weight.detach().cpu().numpy().T, dtype=jnp.float32) for layer in linear_layers]
        bs = [jnp.asarray(layer.bias.detach().cpu().numpy(), dtype=jnp.float32) for layer in linear_layers]

        # Stop gradients and device-put to ensure they’re static constants
        ws = [jax.lax.stop_gradient(jax.device_put(w)) for w in ws]
        bs = [jax.lax.stop_gradient(jax.device_put(b)) for b in bs]

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

        # Compile once to stabilize constants
        jax_model = jax.jit(jax_model)
        _ = jax_model(jnp.zeros((linear_layers[0].in_features,), dtype=jnp.float32))
        return jax_model

    def convert_cnn(conv_layers, fc_layers, strides=None, paddings=None):
        conv_ws, conv_bs = [], []
        kernel_sizes = []
        for conv in conv_layers:
            w = jnp.asarray(conv.weight.detach().cpu().numpy(), dtype=jnp.float32)  # OIHW
            b = jnp.asarray(conv.bias.detach().cpu().numpy(), dtype=jnp.float32)
            kH, kW = w.shape[2], w.shape[3]
            kernel_sizes.append((kH, kW))
            w = w.transpose(2, 3, 1, 0)  # HWIO
            conv_ws.append(jax.lax.stop_gradient(jax.device_put(w)))
            conv_bs.append(jax.lax.stop_gradient(jax.device_put(b)))

        if strides is None:
            strides = [(4, 4)] * len(conv_layers)
        if paddings is None:
            paddings_jax = "VALID"
            use_valid_padding = True
        else:
            paddings_jax = paddings
            use_valid_padding = False

        fc_ws = [jax.lax.stop_gradient(jax.device_put(jnp.asarray(fc.weight.detach().cpu().numpy().T, dtype=jnp.float32))) for fc in fc_layers]
        fc_bs = [jax.lax.stop_gradient(jax.device_put(jnp.asarray(fc.bias.detach().cpu().numpy(), dtype=jnp.float32))) for fc in fc_layers]

        def _to_nhwc(x):
            x = jnp.asarray(x)
            if x.ndim == 3:
                return jnp.transpose(x, (1, 2, 0)), "NCHW_single"
            elif x.ndim == 4:
                return jnp.transpose(x, (0, 2, 3, 1)), "NCHW_batch"
            else:
                raise ValueError(f"Unsupported input ndim={x.ndim} for CNN (expect 3 or 4).")

        def jax_model(x):
            x = jnp.asarray(x, dtype=jnp.float32)
            x, _ = _to_nhwc(x)
            added_batch = False
            if x.ndim == 3:
                x = x[None, ...]
                added_batch = True

            for i, (w, b) in enumerate(zip(conv_ws, conv_bs)):
                pad_arg = "VALID" if use_valid_padding else paddings_jax[i]
                x = lax.conv_general_dilated(
                    x,
                    w,
                    window_strides=strides[i],
                    padding=pad_arg,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )
                x = x + b.reshape((1, 1, 1, -1))
                x = jnp.maximum(x, 0.0)

            x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC → NCHW
            x = x.reshape((x.shape[0], -1))
            for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
                x = jnp.dot(x, w) + b
                if i < len(fc_ws) - 1:
                    x = jnp.maximum(x, 0.0)
            return x[0] if added_batch else x

        jax_model = jax.jit(jax_model)
        dummy_in = jnp.zeros((3, 224, 224), dtype=jnp.float32)
        _ = jax_model(dummy_in)
        return jax_model

    # # ------------------- Helper: MLP-like conversion -------------------
    # def convert_mlp(linear_layers):
    #     ws, bs = [], []
    #     for layer in linear_layers:
    #         ws.append(jnp.array(layer.weight.detach().numpy().T))
    #         bs.append(jnp.array(layer.bias.detach().numpy()))

    #     def jax_model(x):
    #         # Accept numpy / torch / jax arrays
    #         x = jnp.asarray(x, dtype=jnp.float32)
    #         added_batch = False
    #         if x.ndim == 1:
    #             x = x[None, :]
    #             added_batch = True
    #         for i in range(len(ws)):
    #             x = jnp.dot(x, ws[i]) + bs[i]
    #             if i < len(ws) - 1:
    #                 x = jnp.maximum(x, 0.0)
    #         return x[0] if added_batch else x

    #     return jax_model

    # # ------------------- Helper: CNN-like conversion -------------------
    # # def convert_cnn(conv_layers, fc_layers, strides=None, paddings=None):
    # #     # Extract conv weights as HWIO and conv biases
    # #     conv_ws, conv_bs = [], []
    # #     for conv in conv_layers:
    # #         # PyTorch conv.weight is (out_ch, in_ch, kH, kW) -> convert to HWIO
    # #         w = np.asarray(conv.weight.detach().cpu())  # OIHW
    # #         b = np.asarray(conv.bias.detach().cpu())
    # #         w = jnp.array(w.transpose(2, 3, 1, 0))  # HWIO
    # #         b = jnp.array(b)
    # #         conv_ws.append(w)
    # #         conv_bs.append(b)

    # #     if strides is None:
    # #         strides = [(4, 4)] * len(conv_layers)
    # #     if paddings is None:
    # #         # default explicit paddings (before, after) per spatial dim to avoid autotune issues
    # #         paddings = []
    # #         for w in conv_ws:
    # #             pad_h = (0, w.shape[0] - 1)
    # #             pad_w = (0, w.shape[1] - 1)
    # #             paddings.append((pad_h, pad_w))

    # #     # FC weights (transpose PyTorch out,in -> our dot expects (in, out) stored)
    # #     fc_ws, fc_bs = [], []
    # #     for fc in fc_layers:
    # #         fc_ws.append(jnp.array(fc.weight.detach().numpy().T))
    # #         fc_bs.append(jnp.array(fc.bias.detach().numpy()))

    # #     def _to_nhwc(x):
    # #         """Convert x (single or batch) to NHWC if it looks like NCHW/C,H,W."""
    # #         # x is jnp array already
    # #         if x.ndim == 3:
    # #             # single image: C,H,W or H,W,C
    # #             c, h, w = x.shape
    # #             # Heuristic: if first dim is small (<=4) and last dim large => treat as C,H,W
    # #             if c <= 4 and (h > 4 or w > 4):
    # #                 # C,H,W -> H,W,C
    # #                 return jnp.transpose(x, (1, 2, 0)), "NCHW_single"
    # #             else:
    # #                 # H,W,C -> leave
    # #                 return x, "NHWC_single"
    # #         elif x.ndim == 4:
    # #             # batch: N,C,H,W or N,H,W,C
    # #             n, d1, d2, d3 = x.shape
    # #             # If second dim small and third/fourth large: assume N,C,H,W
    # #             if d1 <= 4 and (d2 > 4 or d3 > 4):
    # #                 return jnp.transpose(x, (0, 2, 3, 1)), "NCHW_batch"
    # #             else:
    # #                 return x, "NHWC_batch"
    # #         else:
    # #             raise ValueError(f"Unsupported input ndim={x.ndim} for CNN (expect 3 or 4).")

    # #     def _from_nhwc(out, original_layout):
    # #         """If caller passed NCHW_single or NCHW_batch, we keep output semantics (we return array same shape semantics)."""
    # #         # For these models the final output is a flat vector (B, out_dim) so no transpose needed.
    # #         return out

    # #     def jax_model(x):
    # #         x = jnp.asarray(x, dtype=jnp.float32)
    # #         added_batch = False
    # #         # Convert to NHWC if necessary and remember original layout
    # #         x, layout = _to_nhwc(x)

    # #         # Ensure batch dimension exists
    # #         if x.ndim == 3:
    # #             x = x[None, ...]
    # #             added_batch = True

    # #         # Convolution sequence (NHWC input, HWIO kernels)
    # #         for i, (w, b) in enumerate(zip(conv_ws, conv_bs)):
    # #             x = lax.conv_general_dilated(
    # #                 x,
    # #                 w,
    # #                 window_strides=strides[i],
    # #                 padding=paddings[i],
    # #                 dimension_numbers=("NHWC", "HWIO", "NHWC"),
    # #             )
    # #             x = x + b.reshape(1, 1, 1, -1)
    # #             x = jnp.maximum(x, 0.0)

    # #         # Flatten and FCs
    # #         x = x.reshape((x.shape[0], -1))
    # #         for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
    # #             x = jnp.dot(x, w) + b
    # #             if i < len(fc_ws) - 1:
    # #                 x = jnp.maximum(x, 0.0)

    # #         # If caller passed a single image originally, return single vector
    # #         if added_batch:
    # #             return x[0]
    # #         return x

    # #     return jax_model
    # def convert_cnn(conv_layers, fc_layers, strides=None, paddings=None):
    #     # Extract conv weights as HWIO and conv biases
    #     conv_ws, conv_bs = [], []
    #     kernel_sizes = []
    #     for conv in conv_layers:
    #         # PyTorch conv.weight is (out_ch, in_ch, kH, kW) -> convert to HWIO
    #         w = jnp.asarray(conv.weight.detach().cpu())  # OIHW
    #         b = jnp.asarray(conv.bias.detach().cpu())
    #         kH = w.shape[2]
    #         kW = w.shape[3]
    #         kernel_sizes.append((kH, kW))
    #         w = jnp.array(w.transpose(2, 3, 1, 0), dtype=jnp.float32)  # HWIO
    #         b = jnp.array(b, dtype=jnp.float32)
    #         conv_ws.append(w)
    #         conv_bs.append(b)

    #     # Strides: accept either provided or default to (4,4) repeated
    #     if strides is None:
    #         strides = [(4, 4)] * len(conv_layers)

    #     # Paddings: We will default to 'VALID' (no padding) to match PyTorch padding=0.
    #     # If explicit paddings were provided by the caller, we accept them (as lists of tuple pairs).
    #     # But prefer using 'VALID' for exact equivalence to padding=0.
    #     use_valid_padding = True
    #     if paddings is None:
    #         paddings_jax = "VALID"
    #     else:
    #         # if user passed paddings as list-of-tuple-of-tuples, use them; otherwise fall back to VALID.
    #         paddings_jax = paddings
    #         use_valid_padding = False

    #     # FC weights (transpose PyTorch out,in -> dot expects (in, out))
    #     fc_ws, fc_bs = [], []
    #     for fc in fc_layers:
    #         fc_ws.append(jnp.array(fc.weight.detach().numpy().T, dtype=jnp.float32))
    #         fc_bs.append(jnp.array(fc.bias.detach().numpy(), dtype=jnp.float32))

    #     # # after creating conv_ws list, print kernel shapes and dtypes
    #     # for i, w in enumerate(conv_ws):
    #     #     print(f"conv{i}: kernel (H,W,in,out) = {tuple(w.shape)}, dtype={w.dtype}")
    #     # for i, b in enumerate(conv_bs):
    #     #     print(f"conv{i}: bias shape = {b.shape}, dtype={b.dtype}")

    #     # # For fc layers:
    #     # for i, w in enumerate(fc_ws):
    #     #     print(f"fc{i}: weight shape (in,out) = {tuple(w.shape)}, dtype={w.dtype}")
    #     #     print(f"fc{i}: bias shape = {fc_bs[i].shape}, dtype={fc_bs[i].dtype}")

    #     # def _to_nhwc(x):
    #     #     """Convert x (single or batch) to NHWC if it looks like NCHW/C,H,W.
    #     #     Returns (x_nhwc, layout_tag) where layout_tag is for debugging.
    #     #     """
    #     #     x = jnp.asarray(x)
    #     #     if x.ndim == 3:
    #     #         c, h, w = x.shape
    #     #         if c <= 4 and (h > 4 or w > 4):
    #     #             return jnp.transpose(x, (1, 2, 0)), "NCHW_single"
    #     #         else:
    #     #             return x, "NHWC_single"
    #     #     elif x.ndim == 4:
    #     #         n, d1, d2, d3 = x.shape
    #     #         if d1 <= 4 and (d2 > 4 or d3 > 4):
    #     #             return jnp.transpose(x, (0, 2, 3, 1)), "NCHW_batch"
    #     #         else:
    #     #             return x, "NHWC_batch"
    #     #     else:
    #     #         raise ValueError(f"Unsupported input ndim={x.ndim} for CNN (expect 3 or 4).")
    #     def _to_nhwc(x):
    #         """Always convert PyTorch NCHW (or single C,H,W) to NHWC for JAX."""
    #         x = jnp.asarray(x)
    #         if x.ndim == 3:
    #             # Single image C,H,W → H,W,C
    #             return jnp.transpose(x, (1, 2, 0)), "NCHW_single"
    #         elif x.ndim == 4:
    #             n, c, h, w = x.shape
    #             # Batch N,C,H,W → N,H,W,C
    #             return jnp.transpose(x, (0, 2, 3, 1)), "NCHW_batch"
    #         else:
    #             raise ValueError(f"Unsupported input ndim={x.ndim} for CNN (expect 3 or 4).")

    #     def jax_model(x):
    #         x = jnp.asarray(x, dtype=jnp.float32)
    #         # added_batch = False
    #         # # Convert to NHWC
    #         # x, layout = _to_nhwc(x)

    #         # if x.ndim == 3:
    #         #     x = x[None, ...]
    #         #     added_batch = True
    #         x, layout = _to_nhwc(x)   # always NHWC
    #         added_batch = False
    #         if x.ndim == 3:
    #             x = x[None, ...]
    #             added_batch = True

    #         # Convolution sequence (NHWC input, HWIO kernels)
    #         for i, (w, b) in enumerate(zip(conv_ws, conv_bs)):
    #             # print("Input NHWC:", x.shape)
    #             # print("Kernel HWIO:", w.shape)
    #             # print("Bias:", b.shape)

    #             if use_valid_padding:
    #                 pad_arg = "VALID"
    #             else:
    #                 pad_arg = paddings_jax[i]

    #             x = lax.conv_general_dilated(
    #                 x,
    #                 w,
    #                 window_strides=strides[i],
    #                 padding=pad_arg,
    #                 dimension_numbers=("NHWC", "HWIO", "NHWC"),
    #             )
    #             # add bias: shape (out_ch,) -> broadcast over batch, H, W
    #             x = x + b.reshape((1, 1, 1, -1))
    #             x = jnp.maximum(x, 0.0)
    #             # print("After conv:", x.shape)
    #         # After conv sequence
    #         x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC → NCHW
    #         # Flatten and FCs
    #         x = x.reshape((x.shape[0], -1))
    #         for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
    #             x = jnp.dot(x, w) + b
    #             if i < len(fc_ws) - 1:
    #                 x = jnp.maximum(x, 0.0)

    #         if added_batch:
    #             return x[0]
    #         return x

    #     return jax_model

    # ------------------- MLP -------------------
    if isinstance(model, MLP):
        linear_layers = [l for l in model.net if isinstance(l, torch.nn.Linear)]
        return convert_mlp(linear_layers)

    # ------------------- TinyTaxiNetDNN -------------------
    elif isinstance(model, TinyTaxiNetDNN):
        linear_layers = [model.fc1, model.fc2, model.fc3, model.fc4]
        return convert_mlp(linear_layers)

    # ------------------- CNN -------------------
    elif isinstance(model, CNN):
        conv_layers = [model.conv1, model.conv2]
        fc_layers = [model.fc1, model.fc2, model.fc3]
        # default paddings and strides will be computed in convert_cnn
        return convert_cnn(conv_layers, fc_layers)

    # ------------------- TaxiNetCNN -------------------
    elif isinstance(model, TaxiNetCNN):
        conv_layers = [model.conv1, model.conv2]
        fc_layers = [model.fc1, model.fc2, model.fc3]
        strides = [(4, 4), (4, 4)]
        # Use explicit paddings (tuples) to avoid autotuning problems
        paddings = []
        # for conv in conv_layers:
        #     kH, kW = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        #     pad_h = (0, kH - 1)
        #     pad_w = (0, kW - 1)
        #     paddings.append((pad_h, pad_w))
        for conv in conv_layers:
            kH, kW = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
            pad_h = (0, 0)
            pad_w = (0, 0)
            paddings.append((pad_h, pad_w))

        return convert_cnn(conv_layers, fc_layers, strides=strides, paddings=paddings)

    # ------------------- Unsupported -------------------
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

# def torch_to_jax_model(model):
#     """Convert a PyTorch MLP, TinyTaxiNetDNN, CNN, or TaxiNetCNN into a pure JAX function."""

#     # ------------------- Helper: MLP-like conversion -------------------
#     def convert_mlp(linear_layers):
#         ws, bs = [], []
#         for layer in linear_layers:
#             ws.append(jnp.array(layer.weight.detach().numpy().T))
#             bs.append(jnp.array(layer.bias.detach().numpy()))

#         def jax_model(x):
#             x = jnp.asarray(x, dtype=jnp.float32)
#             added_batch = False
#             if x.ndim == 1:
#                 x = x[None, :]
#                 added_batch = True
#             for i in range(len(ws)):
#                 x = jnp.dot(x, ws[i]) + bs[i]
#                 if i < len(ws) - 1:
#                     x = jnp.maximum(x, 0.0)
#             return x[0] if added_batch else x

#         return jax_model

#     # ------------------- Helper: CNN-like conversion -------------------
#     def convert_cnn(conv_layers, fc_layers, strides=None, paddings=None):
#         conv_ws, conv_bs = [], []
#         for conv in conv_layers:
#             w = np.asarray(conv.weight.detach().cpu())  # OIHW
#             b = np.asarray(conv.bias.detach().cpu())
#             w = jnp.array(w.transpose(2, 3, 1, 0))  # HWIO
#             b = jnp.array(b)
#             conv_ws.append(w)
#             conv_bs.append(b)

#         if strides is None:
#             strides = [(4, 4)] * len(conv_layers)
#         if paddings is None:
#             paddings = ["VALID"] * len(conv_layers)

#         fc_ws, fc_bs = [], []
#         for fc in fc_layers:
#             fc_ws.append(jnp.array(fc.weight.detach().numpy().T))
#             fc_bs.append(jnp.array(fc.bias.detach().numpy()))

#         def jax_model(x):
#             x = jnp.asarray(x, dtype=jnp.float32)
#             if x.ndim == 3:
#                 x = x[None, ...]  # add batch dimension

#             # Conv layers
#             for i, (w, b) in enumerate(zip(conv_ws, conv_bs)):
#                 x = lax.conv_general_dilated(
#                     x,
#                     w,
#                     window_strides=strides[i],
#                     padding=paddings[i],
#                     dimension_numbers=("NHWC", "HWIO", "NHWC"),
#                 )
#                 x = x + b.reshape(1, 1, 1, -1)
#                 x = jnp.maximum(x, 0.0)

#             # Flatten + FCs
#             x = x.reshape((x.shape[0], -1))
#             for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
#                 x = jnp.dot(x, w) + b
#                 if i < len(fc_ws) - 1:
#                     x = jnp.maximum(x, 0.0)

#             return x[0] if x.shape[0] == 1 else x

#         return jax_model

#     # ------------------- MLP -------------------
#     if isinstance(model, MLP):
#         linear_layers = [l for l in model.net if isinstance(l, torch.nn.Linear)]
#         return convert_mlp(linear_layers)

#     # ------------------- TinyTaxiNetDNN -------------------
#     elif isinstance(model, TinyTaxiNetDNN):
#         linear_layers = [model.fc1, model.fc2, model.fc3, model.fc4]
#         return convert_mlp(linear_layers)

#     # ------------------- CNN -------------------
#     elif isinstance(model, CNN):
#         conv_layers = [model.conv1, model.conv2]
#         fc_layers = [model.fc1, model.fc2, model.fc3]
#         return convert_cnn(conv_layers, fc_layers)

#     # ------------------- TaxiNetCNN -------------------
#     elif isinstance(model, TaxiNetCNN):
#         conv_layers = [model.conv1, model.conv2]
#         fc_layers = [model.fc1, model.fc2, model.fc3]
#         strides = [(4, 4), (4, 4)]
#         paddings = ["VALID", "VALID"]
#         return convert_cnn(conv_layers, fc_layers, strides, paddings)

#     # ------------------- Unsupported -------------------
#     else:
#         raise ValueError(f"Unsupported model type: {type(model)}")

# def torch_to_jax_model(model):
#     """Convert a PyTorch MLP or small CNN into a pure JAX function with baked-in weights."""

#     # ---------------- MLP ----------------
#     if isinstance(model, MLP) or isinstance(model, TinyTaxiNetDNN):
#         layers = [l for l in model.net if isinstance(l, torch.nn.Linear)]
#         ws, bs = [], []
#         for layer in layers:
#             ws.append(jnp.array(layer.weight.detach().numpy().T))  # (in, out)
#             bs.append(jnp.array(layer.bias.detach().numpy()))

#         def jax_model(x):
#             x = jnp.asarray(x, dtype=jnp.float32)
#             added_batch = False
#             if x.ndim == 1:
#                 x = x[None, :]
#                 added_batch = True

#             for i in range(len(ws)):
#                 x = jnp.dot(x, ws[i]) + bs[i]
#                 if i < len(ws) - 1:
#                     x = jnp.maximum(x, 0.0)

#             return x[0] if added_batch else x

#         return jax_model

#     # ---------------- CNN ----------------
#     if isinstance(model, CNN):
#         convs = [model.conv1, model.conv2]
#         # fcs = [model.fc1, model.fc2, model.fc3]
#         fcs = [model.fc1, model.fc4]  # for the smaller model version

#         # Extract conv weights as OIHW (same ordering PyTorch uses)
#         conv_ws, conv_bs, conv_strides = [], [], []
#         for i, conv in enumerate(convs):
#             # PyTorch: (out_channels, in_channels, kh, kw)
#             w = np.asarray(conv.weight.detach().cpu())    # OIHW
#             b = np.asarray(conv.bias.detach().cpu())      # (out,)
#             conv_ws.append(jnp.array(w))                  # keep OIHW ordering
#             conv_bs.append(jnp.array(b))
#             # use the conv's actual stride (tuple) to be safe
#             stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
#             conv_strides.append(tuple(stride))

#         # Extract FC weights (keep shapes so jnp.dot works with x.reshape((B,-1)))
#         fc_ws, fc_bs = [], []
#         for fc in fcs:
#             # torch Linear weight shape: (out, in). For jnp.dot with (B, in) we want (in, out)
#             fc_ws.append(jnp.array(fc.weight.detach().cpu().numpy().T))  # (in, out)
#             fc_bs.append(jnp.array(fc.bias.detach().cpu().numpy()))      # (out,)

#         def jax_model(x):
#             # Expect x in NCHW or (C,H,W) or (H,W,C)?? We choose: keep NCHW to match torch_input
#             x = jnp.asarray(x, dtype=jnp.float32)
#             if x.ndim == 3:
#                 # If user passed (C,H,W), add batch dim to make (1,C,H,W)
#                 x = x[None, ...]

#             # confirm NCHW layout: (B, C, H, W)
#             # Apply convs using OIHW kernels and NCHW dimension_numbers
#             for i, (w, b) in enumerate(zip(conv_ws, conv_bs)):
#                 x = lax.conv_general_dilated(
#                     lhs=x,
#                     rhs=w,  # OIHW
#                     window_strides=conv_strides[i],
#                     padding="VALID",
#                     dimension_numbers=("NCHW", "OIHW", "NCHW"),
#                 )
#                 # add bias: shape broadcast (1, out_ch, 1, 1)
#                 x = x + b.reshape(1, -1, 1, 1)
#                 x = jnp.maximum(x, 0.0)  # ReLU

#             # Flatten for FC: (B, C, H, W) -> (B, N)
#             x = x.reshape((x.shape[0], -1))

#             # Apply FCs (fc_ws are (in, out))
#             for i, (w, b) in enumerate(zip(fc_ws, fc_bs)):
#                 x = jnp.dot(x, w) + b  # (B, in) @ (in, out) + (out,)
#                 if i < len(fc_ws) - 1:
#                     x = jnp.maximum(x, 0.0)

#             return x[0] if x.shape[0] == 1 else x

#         return jax_model
    



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

    # else:
    #     raise ValueError(f"Unsupported model type: {type(model)}")


# if __name__ == "__main__":
#     model = CNN(input_channels=3, out_dim=4, H=128, W=128)
#     model_name = "image_estimator"
#     checkpoint = torch.load(
#         "/home/nick/code/hjnnv/src/learned_models/beacon/estimators/"
#         + model_name
#         + "/best_model.pt",
#         map_location="cpu",
#     )
#     model.load_state_dict(checkpoint["model_state_dict"])

#     config = checkpoint.get("config_dict", None)

#     from dynamic_models.beacon import BeaconDynamics
#     max_input = 1.0
#     max_position_disturbance = 0.1
#     max_vel_disturbance = 0.01
#     epsilon = 0.05
#     dt = 0.1
#     seed = 1
#     image_dynamics = BeaconDynamics(
#         dt=dt,
#         max_input=max_input,
#         max_position_disturbance=max_position_disturbance,
#         max_vel_disturbance=max_vel_disturbance,
#         range_disturbance=epsilon,
#         obs_type="images",
#         model_name=model_name,
#         random_seed=seed
#     )
#     state = jnp.array([5.0, 5.0, 0.0, 0.0])
#     obs = image_dynamics.get_observation(state, time=0)

#     torch_input = torch.tensor(np.array(obs)).reshape((1, 3, 128, 128))

#     # obs = obs.squeeze(axis=2)
#     print(obs.reshape((1, 3, 128, 128)))
#     jax_model = torch_to_jax_model(model)
#     jax_output = jax_model(obs.reshape((3, 128, 128)))
#     torch_output = model.forward(torch_input)

#     print("JAX output:", jax_output)
#     print("Torch output:", torch_output)

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

class JaxTaxiNetCNN:
    def __init__(self, pt_model, use_valid_padding=True):
        # Extract conv weights and biases
        self.conv_ws = []
        self.conv_bs = []
        conv_layers = [pt_model.conv1, pt_model.conv2]
        for conv in conv_layers:
            w_pt = conv.weight.detach().cpu().numpy()
            w_jax = np.transpose(w_pt, (2, 3, 1, 0))  # HWIO
            self.conv_ws.append(jax.device_put(w_jax))
            self.conv_bs.append(jax.device_put(conv.bias.detach().cpu().numpy()))

        # Extract FC weights and biases
        fc_layers = [pt_model.fc1, pt_model.fc2, pt_model.fc3]
        self.fc_ws = []
        self.fc_bs = []
        for fc in fc_layers:
            w = fc.weight.detach().cpu().numpy().T  # (in, out)
            b = fc.bias.detach().cpu().numpy()
            self.fc_ws.append(jax.device_put(w))
            self.fc_bs.append(jax.device_put(b))

        self.strides = [(4, 4), (4, 4)]
        self.use_valid_padding = use_valid_padding

    def conv_layer(self, x, layer_idx):
        """Run only a single conv layer (NHWC in/out)."""
        w = self.conv_ws[layer_idx]
        b = self.conv_bs[layer_idx]
        pad_arg = "VALID" if self.use_valid_padding else ((0, 0), (0, 0))
        x = lax.conv_general_dilated(
            x, w,
            window_strides=self.strides[layer_idx],
            padding=pad_arg,
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        x = x + b.reshape((1, 1, 1, -1))
        return jnp.maximum(x, 0.0)

    def __call__(self, x):
        added_batch = False
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == 3:
            x = x[None, ...]
            added_batch = True
        # Convert NCHW → NHWC if needed
        if x.shape[-1] != self.conv_ws[0].shape[2]:
            x = jnp.transpose(x, (0, 2, 3, 1))

        # Conv sequence
        for i in range(len(self.conv_ws)):
            x = self.conv_layer(x, i)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # FC sequence
        for i, (w, b) in enumerate(zip(self.fc_ws, self.fc_bs)):
            x = jnp.dot(x, w) + b
            if i < len(self.fc_ws) - 1:
                x = jnp.maximum(x, 0.0)

        if added_batch:
            return x[0]
        return x

def debug_cnn_conversion(pt_model, jax_model, test_input):
    """
    Compare outputs of PyTorch and JAX CNN layer by layer.
    test_input: torch tensor with shape (N, C, H, W)
    """
    import numpy as np
    from jax import device_get
    import torch.nn.functional as F

    pt_model.eval()

    x_pt = test_input.clone()
    x_jax = test_input.detach().cpu().numpy()

    # Convert input to NHWC for JAX
    x_jax = np.transpose(x_jax, (0, 2, 3, 1))
    x_jax = jnp.asarray(x_jax, dtype=jnp.float32)

    print("Input shape:", x_pt.shape)

    # ---- Conv1 ----
    x_pt1 = F.relu(pt_model.conv1(x_pt)).detach().cpu().numpy()
    x_pt1_nhwc = np.transpose(x_pt1, (0, 2, 3, 1))  # NCHW → NHWC
    x_jax1 = jax_model.conv_layer(x_jax, layer_idx=0)
    diff1 = np.abs(x_pt1_nhwc - device_get(x_jax1))
    print(f"Conv1: max diff {diff1.max():.6f}, mean diff {diff1.mean():.6f}")

    # ---- Conv2 ----
    x_pt2 = F.relu(pt_model.conv2(torch.tensor(x_pt1))).detach().cpu().numpy()
    x_pt2_nhwc = np.transpose(x_pt2, (0, 2, 3, 1))  # NCHW → NHWC
    x_jax2 = jax_model.conv_layer(x_jax1, layer_idx=1)
    diff2 = np.abs(x_pt2_nhwc - device_get(x_jax2))
    print(f"Conv2: max diff {diff2.max():.6f}, mean diff {diff2.mean():.6f}")

    # ---- Flatten ----
    x_pt_flat = x_pt2_nhwc.reshape(x_pt2_nhwc.shape[0], -1)
    x_jax_flat = device_get(x_jax2).reshape(x_jax2.shape[0], -1)
    diff_flat = np.abs(x_pt_flat - x_jax_flat)
    print(f"Flatten: max diff {diff_flat.max():.6f}, mean diff {diff_flat.mean():.6f}")

    # ---- FC layers ----
    x_pt_fc = x_pt_flat.copy()
    x_jax_fc = x_jax_flat.copy()
    for i, (w, b) in enumerate(zip(jax_model.fc_ws, jax_model.fc_bs)):
        w_np = device_get(w)
        b_np = device_get(b)

        # PyTorch equivalent in numpy
        x_pt_fc = x_pt_fc @ w_np + b_np
        if i < len(jax_model.fc_ws) - 1:
            x_pt_fc = np.maximum(x_pt_fc, 0.0)

        # JAX equivalent
        x_jax_fc = x_jax_fc @ w_np + b_np
        if i < len(jax_model.fc_ws) - 1:
            x_jax_fc = np.maximum(x_jax_fc, 0.0)

        diff_fc = np.abs(x_pt_fc - x_jax_fc)
        print(f"FC{i}: max diff {diff_fc.max():.6f}, mean diff {diff_fc.mean():.6f}")

    # ---- Final output ----
    print("Final PyTorch output:", x_pt_fc)
    print("Final JAX output:", x_jax_fc)
    print("Final diff:", np.abs(x_pt_fc - x_jax_fc).max())

if __name__ == "__main__":

    import numpy as np
    import torch
    import jax.numpy as jnp
    from jax import device_get
    import torch.nn.functional as F

    model_dir = "/home/nick/code/hjnnv/src/simulators/NASA_ULI_Xplane_Simulator/models/cnn64_taxinet"

    in_channels = 1
    H = 64
    W = 64

    # instantiate your PyTorch model exactly as used in inference
    pt_model = TaxiNetCNN(input_channels=in_channels, H=H, W=W)   # adjust to your model
    pt_model.eval()

    # load state dict you use in runtime
    sd = torch.load(model_dir + "/best_model.pt", map_location="cpu")
    pt_model.load_state_dict(sd)
    pt_model.eval()

    jax_model = JaxTaxiNetCNN(pt_model, use_valid_padding=True)

    # make a small random test input with same preprocessing pipeline result
    # NOTE: shape should be (C,H,W) or (1,C,H,W), depending how you call converter
    torch.manual_seed(42)
    test_torch_input = torch.randn(1, in_channels, H, W)  # batch
    # debug_cnn_conversion(pt_model, jax_model, test_torch_input)
    with torch.no_grad():
        pt_out = pt_model(test_torch_input).cpu().numpy()
        x_pt1 = F.relu(pt_model.conv1(test_torch_input)).cpu().numpy()
        print("PyTorch conv1:", x_pt1)

    # convert model
    jax_model = torch_to_jax_model(pt_model)

    # Feed the same data to jax model (convert to numpy/float32)
    test_jax_input = test_torch_input.detach().cpu().numpy()
    # jax_model accepts either (C,H,W) or (N,C,H,W). Our converter will detect NCHW and transpose.
    jax_out = device_get(jax_model(test_jax_input))

    print("pt_out:", pt_out.shape, pt_out)
    print("jax_out:", jax_out.shape, jax_out)

    diff = np.abs(pt_out - jax_out)
    print("max abs diff:", diff.max(), "mean diff:", diff.mean())