import jax.numpy as jnp
import jax
import numpy as np

import hj_reachability as hj
import torch

from dynamic_models.dynamics import HJNNVDynamics
from learned_models.beacon.estimators.simple_estimator_gpt.estimator_mlp import MLP
from utils.mlp2jax import torch_mlp2jax

class BeaconDynamics(HJNNVDynamics):

    def __init__(self,
                 dt=0.1,
                 max_input=1.0,
                 max_position_disturbance=0.01,
                 max_vel_disturbance=0.005,
                 range_disturbance=0.02,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None,
                 model_name="simple_estimator_3",
                 random_seed=0
                 ):

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_input, -max_input]),
                                        jnp.array([max_input, max_input]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(
                jnp.array([
                    -max_position_disturbance,
                    -max_position_disturbance,
                    -max_vel_disturbance,
                    -max_vel_disturbance
                ]),
                jnp.array([
                    max_position_disturbance,
                    max_position_disturbance,
                    max_vel_disturbance,
                    max_vel_disturbance
                ])
            )

        # self.beacons = jnp.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [4.0, 4.0]])
        self.range_disturbance = range_disturbance
        self.load_estimator(model_name=model_name)
        self.previous_observations = jnp.array([0., 0.])
        self.state_hat = jnp.array([5., 5., 0., 0.])

        super().__init__(
            dt,
            control_mode,
            disturbance_mode,
            control_space,
            disturbance_space,
            random_seed
        )

    def open_loop_dynamics(self, state, time):
        _, _, vx, vy = state
        return jnp.array([vx, vy, 0., 0.])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.5 * self.dt, 0.],
            [0., 0.5 * self.dt],
            [1., 0.],
            [0., 1.]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.eye(4)

    def get_observation(self, state, time):
        self.key, subkey = jax.random.split(self.key)
        pos = jnp.array([state[0], state[1]])
        d = jnp.linalg.norm(self.beacons - pos, axis=1)  # (3,)
        d_noisy = d + jax.random.uniform(
            subkey,
            shape=d.shape,
            minval=-self.range_disturbance,
            maxval=self.range_disturbance
        )
        if time == 0:
            self.previous_observations = jnp.concatenate([d_noisy, d_noisy], axis=0)
            # import ipdb; ipdb.set_trace()
            return jnp.concatenate([d_noisy, d_noisy, d_noisy], axis=0)
        elif time == 1:
            # import ipdb; ipdb.set_trace()
            self.previous_observations = jnp.concatenate(
                [self.previous_observations[4:], d_noisy],
                axis=0
            )
            return jnp.concatenate([self.previous_observations, d_noisy], axis=0)
        else:
            d_noisy_seq = jnp.concatenate([self.previous_observations, d_noisy])
            # import ipdb; ipdb.set_trace()
            self.previous_observations = d_noisy_seq[4:]

        return d_noisy_seq

    def load_estimator(self, model_name="simple_estimator_3"):
        
        checkpoint = torch.load(
            "/home/nick/code/hjnnv/src/learned_models/beacon/estimators/"
            + model_name
            + "/best_model.pt",
            map_location="cpu",
        )

        config = checkpoint.get("config_dict", None)
        self.estimator = MLP(in_dim=12, out_dim=4, hidden=config["hidden"])

        # load just the weights
        self.estimator.load_state_dict(checkpoint["model_state_dict"])
        self.estimator.eval()

        self.estimator = torch_mlp2jax(self.estimator)

        # optional: keep normalization params if you need them later
        self.in_mean = checkpoint.get("in_mean", None)
        self.in_std = checkpoint.get("in_std", None)
        self.beacons = jnp.array(config["beacons"])

    def get_state_estimate(self, obs):
        # obs = torch.tensor(np.array(obs)).unsqueeze(0)
        obs = (obs - self.in_mean) / self.in_std
        state_hat = self.estimator(obs)
        return state_hat
