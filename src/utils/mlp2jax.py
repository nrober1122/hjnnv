import torch
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def torch_mlp2jax(model: torch.nn.Module):
    """Convert a PyTorch model (Conv2d + Linear + ReLU) to a JAX function.
    Expects PyTorch convs trained on NHWC inputs when converted.
    Returns a JAX function taking NHWC (batch,H,W,C) or (H,W,C).
    """
    layers = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            w = np.asarray(layer.weight.detach().cpu())  # (out_ch, in_ch, kH, kW)
            b = np.asarray(layer.bias.detach().cpu()).reshape(-1)  # (out_ch,)
            w = jnp.array(w.transpose(2, 3, 1, 0))  # -> HWIO
            b = jnp.array(b)

            # Handle padding robustly
            padding = layer.padding
            if isinstance(padding, int):
                pad_h = (padding, padding)
                pad_w = (padding, padding)
                padding_lax = (pad_h, pad_w)
            elif isinstance(padding, tuple) and all(isinstance(p, int) for p in padding):
                pad_h = (padding[0], padding[0])
                pad_w = (padding[1], padding[1])
                padding_lax = (pad_h, pad_w)
            elif isinstance(padding, str):
                padding_lax = padding.upper()
            else:
                raise ValueError(f"Unsupported padding type: {padding}")

            strides = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
            dilation = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation)

            layers.append(("conv2d", w, b, dict(strides=strides,
                                               padding=padding_lax,
                                               dilation=dilation)))

        elif isinstance(layer, torch.nn.Linear):
            w = np.asarray(layer.weight.detach().cpu())  # (out, in)
            b = np.asarray(layer.bias.detach().cpu()).reshape(-1)
            layers.append(("linear", jnp.array(w), jnp.array(b)))

    def jax_model(x):
        x = jnp.asarray(x)
        added_batch = False
        if x.ndim == 3:
            x = x[None, ...]
            added_batch = True

        x = x.astype(jnp.float32)

        saw_linear = False
        for i, item in enumerate(layers):
            kind = item[0]

            if kind == "conv2d":
                w, b, params = item[1], item[2], item[3]
                x = lax.conv_general_dilated(
                    x,
                    w,
                    window_strides=params["strides"],
                    padding=params["padding"],
                    rhs_dilation=params["dilation"],
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                )
                x = x + b.reshape(1, 1, 1, -1)
                x = jnp.maximum(x, 0.0)

            elif kind == "linear":
                if not saw_linear and x.ndim > 2:
                    x = x.reshape((x.shape[0], -1))
                    saw_linear = True

                w, b = item[1], item[2]
                x = jnp.dot(x, w.T) + b.reshape(1, -1)

                # ReLU for all but last linear
                is_last_linear = not any(l[0] == "linear" for l in layers[i + 1 :])
                if not is_last_linear:
                    x = jnp.maximum(x, 0.0)

        return x[0] if added_batch else x

    return jax_model


# def torch_mlp2jax(model: torch.nn.Module):
    # """Convert a PyTorch model (Conv2d + Linear + ReLU) to a JAX function.
    # Assumes the PyTorch convs were trained on inputs in NHWC when converted,
    # and that the PyTorch forward used `torch.flatten(x, 1)` before linears.
    # The returned jax_model expects inputs in NHWC (batch, H, W, C) or (H, W, C).
    # """
    # layers = []
    # # Collect layers in order encountered by model.modules(); we will use type checks
    # # Note: model.modules() yields modules in a parent-first traversal. For simple
    # # models like conv->conv->fc... this will work; if you have nn.Sequential you
    # # could also iterate through that directly.
    # for layer in model.modules():
    #     if isinstance(layer, torch.nn.Conv2d):
    #         w = np.asarray(layer.weight.detach().cpu())  # (out_ch, in_ch, kH, kW)
    #         b = np.asarray(layer.bias.detach().cpu()).reshape(-1)  # (out_ch,)
    #         # Reorder PyTorch (out_ch, in_ch, kH, kW) -> JAX HWIO (kH, kW, in_ch, out_ch)
    #         w = jnp.array(w.transpose(2, 3, 1, 0))
    #         b = jnp.array(b)
    #         conv_params = {
    #             "strides": tuple(layer.stride) if isinstance(layer.stride, tuple) else (layer.stride, layer.stride),
    #             "padding": layer.padding,
    #             "dilation": tuple(layer.dilation) if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation)
    #         }
    #         layers.append(("conv2d", w, b, conv_params))

    #     elif isinstance(layer, torch.nn.Linear):
    #         w = np.asarray(layer.weight.detach().cpu())  # (out, in)
    #         b = np.asarray(layer.bias.detach().cpu()).reshape(-1)  # (out,)
    #         layers.append(("linear", jnp.array(w), jnp.array(b)))

    #     # ignore other module types (ReLU handled in-line)

    # def jax_model(x):
    #     # Accept x as (H,W,C) or (batch,H,W,C); normalize to (batch,H,W,C)
    #     x = jnp.asarray(x)
    #     added_batch = False
    #     if x.ndim == 3:
    #         x = x[None, ...]  # add batch dim
    #         added_batch = True

    #     # Ensure dtype float32
    #     x = x.astype(jnp.float32)

    #     # Process layers in collected order. We will:
    #     # - apply convs with NHWC input and HWIO filters
    #     # - apply ReLU after convs (we assume convs are followed by ReLU in torch)
    #     # - before the first linear, flatten spatial dims to (batch, -1)
    #     saw_linear = False
    #     for i, item in enumerate(layers):
    #         kind = item[0]

    #         if kind == "conv2d":
    #             w, b, params = item[1], item[2], item[3]
    #             # Prepare padding for lax.conv_general_dilated:
    #             padding = params["padding"]
    #             if isinstance(padding, tuple) and all(isinstance(p, int) for p in padding):
    #                 # PyTorch e.g. padding=(0,0) -> ((0,0),(0,0))
    #                 pad_h = (padding[0], padding[0])
    #                 pad_w = (padding[1], padding[1])
    #                 padding_lax = (pad_h, pad_w)
    #             else:
    #                 # If padding is already something else (rare), try uppercase string
    #                 padding_lax = padding.upper() if isinstance(padding, str) else padding

    #             strides = params["strides"]
    #             dilation = params["dilation"]

    #             # Apply convolution (NHWC input, HWIO kernel)
    #             x = lax.conv_general_dilated(
    #                 x,
    #                 w,
    #                 window_strides=strides,
    #                 padding=padding_lax,
    #                 rhs_dilation=dilation,
    #                 dimension_numbers=("NHWC", "HWIO", "NHWC")
    #             )
    #             # add bias (b shape (out_ch,), broadcast over batch,H,W)
    #             x = x + b.reshape(1, 1, 1, -1)
    #             # ReLU
    #             x = jnp.maximum(x, 0.0)

    #         elif kind == "linear":
    #             # Before the first linear: flatten (batch, H, W, C) -> (batch, -1)
    #             if not saw_linear:
    #                 if x.ndim > 2:
    #                     x = x.reshape((x.shape[0], -1))
    #                 saw_linear = True

    #             w, b = item[1], item[2]  # w: (out, in), b: (out,)
    #             # We want (batch, in) @ (in, out) -> (batch, out)
    #             x = jnp.dot(x, w.T) + b.reshape(1, -1)
    #             # Apply ReLU for all but last linear (we can't check "last" easily here,
    #             # so we will apply ReLU for all linear layers except the final one:
    #             # detect if this is the last linear by looking ahead
    #             is_last_linear = True
    #             for j in range(i+1, len(layers)):
    #                 if layers[j][0] == "linear":
    #                     is_last_linear = False
    #                     break
    #             if not is_last_linear:
    #                 x = jnp.maximum(x, 0.0)

    #     # If caller passed single sample without batch, return squeezed result
    #     if added_batch:
    #         return x[0]
    #     return x

    # return jax_model

# import torch
# import jax
# import jax.numpy as jnp
# from jax import lax


# def torch_mlp2jax(model: torch.nn.Module):
#     """Convert a PyTorch model to a JAX function."""
#     layers = []
#     # import ipdb; ipdb.set_trace()
#     for layer in model.modules():
#         if isinstance(layer, torch.nn.Linear):
#             w = layer.weight.detach().numpy()  # shape: (out_features, in_features)
#             b = layer.bias.detach().numpy().reshape(-1, 1)
#             layers.append(("linear", jnp.array(w), jnp.array(b)))

#         elif isinstance(layer, torch.nn.Conv2d):
#             w = layer.weight.detach().numpy()  # (out_ch, in_ch, kH, kW)
#             b = layer.bias.detach().numpy().reshape(-1, 1)
#             # Rearrange to JAX's default: (kH, kW, in_ch, out_ch)
#             w = jnp.array(w.transpose(2, 3, 1, 0))
#             b = jnp.array(b)

#             conv_params = {
#                 "strides": layer.stride,
#                 "padding": layer.padding,  # careful: PyTorch gives tuple, need to map to "SAME"/"VALID" or pad manually
#                 "dilation": layer.dilation
#             }
#             layers.append(("conv2d", w, b, conv_params))

#     def jax_model(x):
#         for i in range(len(layers)):
#             if layers[i][0] == "linear":
#                 w, b = layers[i][1], layers[i][2]  # w: (out, in), b: (out, 1)
#                 x = jnp.dot(x, w.T) + b.T  # (batch, in) @ (in, out) + (batch, out)
#                 if i < len(layers) - 1:
#                     x = jnp.maximum(x, 0)

#             elif layers[i][0] == "conv2d":
#                 w, b, params = layers[i][1], layers[i][2], layers[i][3]
#                 # JAX expects NCHW or NHWC depending on dimension_numbers
#                 # We'll assume NHWC for input
#                 padding = params["padding"]
#                 if isinstance(padding, tuple):
#                     # Convert PyTorch's "pad by N pixels" to JAX ((pad_top, pad_bottom), (pad_left, pad_right))
#                     pad_h = (padding[0], padding[0])
#                     pad_w = (padding[1], padding[1])
#                     padding = (pad_h, pad_w)
#                 else:
#                     padding = padding.upper()  # e.g. 'SAME', 'VALID'

#                 x = lax.conv_general_dilated(
#                     x,
#                     w,
#                     window_strides=params["strides"],
#                     padding=padding,
#                     rhs_dilation=params["dilation"],
#                     dimension_numbers=("NHWC", "HWIO", "NHWC")
#                 )
#                 # Add bias (broadcast over H and W)
#                 x = x + b.reshape(1, 1, 1, -1)
#                 x = jnp.maximum(x, 0)  # ReLU

#         return x

#     return jax_model




# def torch2jax(model: torch.nn.Module) -> callable:
#     """Convert a PyTorch model to a JAX function."""
#     weights = []
#     biases = []

#     for layer in model.modules():
#         if isinstance(layer, torch.nn.Linear):
#             weights.append(layer.weight.detach().numpy())
#             biases.append(layer.bias.detach().numpy().reshape(-1, 1))

#     jax_weights = [jnp.array(w) for w in weights]
#     jax_biases = [jnp.array(b) for b in biases]
#     # import ipdb; ipdb.set_trace()

#     def jax_model(x):
#         for i in range(len(jax_weights) - 1):
#             x = jnp.dot(jax_weights[i], x) + jax_biases[i]
#             x = jnp.maximum(x, 0)  # ReLU
#         x = jnp.dot(jax_weights[-1], x) + jax_biases[-1]
#         return x

#     # import ipdb; ipdb.set_trace()
#     return jax_model
