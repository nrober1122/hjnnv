import jax
import jax.numpy as jnp
import functools


# @functools.partial(jax.jit, static_argnames=['model', 'loss_fn', 'epsilon'])
def fgsm(
    model,
    target,
    observation,
    epsilon=0.1,
    loss_fn=lambda x, y: jnp.mean((x - y) ** 2),
    key=jax.random.PRNGKey(0),
    perturbation_prev=None
):
    grad_fn = jax.grad(lambda obs: loss_fn(model(obs), target))
    grad = grad_fn(observation)
    perturbation = - epsilon * jnp.sign(grad)
    adv_observation = observation + perturbation
    return adv_observation, loss_fn(model(adv_observation), target), perturbation


@functools.partial(jax.jit, static_argnames=['model', 'loss_fn', 'iters', 'alpha', 'epsilon'])
def pgd(
    model,
    target,
    observation,
    epsilon=0.1,
    alpha=0.01,
    iters=40,
    loss_fn=lambda x, y: jnp.mean((x - y) ** 2),
    key=jax.random.PRNGKey(0),
    perturbation_prev=None
):
    # Initialize adversarial example
    if perturbation_prev is None:
        adv = observation.copy()
        adv = adv + jax.random.uniform(
            key,
            shape=adv.shape,
            minval=-epsilon,
            maxval=epsilon
        )
    else:
        adv = observation.copy()
        adv = adv + perturbation_prev + jax.random.uniform(
            key,
            shape=adv.shape,
            minval=-alpha,
            maxval=alpha
        )

    for _ in range(iters):
        grad = jax.grad(lambda o: loss_fn(model(o), target))(adv)
        adv = adv - alpha * jnp.sign(grad)   # take a small FGSM step
        adv = jnp.clip(adv, observation - epsilon, observation + epsilon)  # project into ε-ball

    return adv, loss_fn(model(adv), target), adv - observation
