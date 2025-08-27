import torch
import jax
import jax.numpy as jnp
from jax import lax


def torch_mlp2jax(model: torch.nn.Module):
    """Convert a PyTorch model to a JAX function."""
    layers = []
    # import ipdb; ipdb.set_trace()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().numpy()  # shape: (out_features, in_features)
            b = layer.bias.detach().numpy().reshape(-1, 1)
            layers.append(("linear", jnp.array(w), jnp.array(b)))

        elif isinstance(layer, torch.nn.Conv2d):
            w = layer.weight.detach().numpy()  # (out_ch, in_ch, kH, kW)
            b = layer.bias.detach().numpy().reshape(-1, 1)
            # Rearrange to JAX's default: (kH, kW, in_ch, out_ch)
            w = jnp.array(w.transpose(2, 3, 1, 0))
            b = jnp.array(b)

            conv_params = {
                "strides": layer.stride,
                "padding": layer.padding,  # careful: PyTorch gives tuple, need to map to "SAME"/"VALID" or pad manually
                "dilation": layer.dilation
            }
            layers.append(("conv2d", w, b, conv_params))

    def jax_model(x):
        for i in range(len(layers)):
            if layers[i][0] == "linear":
                w, b = layers[i][1], layers[i][2]  # w: (out, in), b: (out, 1)
                x = jnp.dot(x, w.T) + b.T  # (batch, in) @ (in, out) + (batch, out)
                if i < len(layers) - 1:
                    x = jnp.maximum(x, 0)

            elif layers[i][0] == "conv2d":
                w, b, params = layers[i][1], layers[i][2], layers[i][3]
                # JAX expects NCHW or NHWC depending on dimension_numbers
                # We'll assume NHWC for input
                padding = params["padding"]
                if isinstance(padding, tuple):
                    # Convert PyTorch's "pad by N pixels" to JAX ((pad_top, pad_bottom), (pad_left, pad_right))
                    pad_h = (padding[0], padding[0])
                    pad_w = (padding[1], padding[1])
                    padding = (pad_h, pad_w)
                else:
                    padding = padding.upper()  # e.g. 'SAME', 'VALID'

                x = lax.conv_general_dilated(
                    x,
                    w,
                    window_strides=params["strides"],
                    padding=padding,
                    rhs_dilation=params["dilation"],
                    dimension_numbers=("NHWC", "HWIO", "NHWC")
                )
                # Add bias (broadcast over H and W)
                x = x + b.reshape(1, 1, 1, -1)
                x = jnp.maximum(x, 0)  # ReLU

        return x

    return jax_model




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
