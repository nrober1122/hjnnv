import jax.numpy as jnp
import jax
import numpy as np

import hj_reachability as hj
import torch

from dynamic_models.dynamics import HJNNVDynamics
from learned_models.beacon.estimators import MLP, CNN

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
                 model_name="simple_estimator_3t",
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

    def get_observation(self, state, time, mode="distances"):
        self.key, subkey = jax.random.split(self.key)

        if mode == "distance":
            obs_noisy = self.observe_distance(state, subkey)  # shape (num_beacons,)
            num_beacons = obs_noisy.shape[0]
            history_len = 3
            concatenate_axis = 0

        elif mode == "images":
            focal_length = 0.05
            max_theta = jnp.pi / 3
            lm_h, lm_w = 1.618, 1.0
            image_width, image_height = 256, 256
            obs_noisy = self.observe_image(
                state,
                image_width,
                image_height,
                lm_h,
                lm_w,
                focal_length,
                max_theta
            )  # shape (H, W)
            obs_noisy = obs_noisy[..., None]  # (H, W, 1)
            history_len = 3
            concatenate_axis = -1

        # Initialize history buffer
        if time == 0:
            self.previous_observations = jnp.concatenate(
                [obs_noisy] * history_len, axis=concatenate_axis
            )
            return self.previous_observations

        # Update history
        if mode == "distances":
            # drop oldest block of beacons, append newest
            self.previous_observations = jnp.concatenate(
                [self.previous_observations[num_beacons:], obs_noisy],
                axis=concatenate_axis
            )

        elif mode == "images":
            # drop oldest frame, append newest
            self.previous_observations = jnp.concatenate(
                [self.previous_observations[..., 1:], obs_noisy],
                axis=concatenate_axis
            )

        return self.previous_observations

    # def get_observation(self, state, time, mode="distance"):
    #     self.key, subkey = jax.random.split(self.key)
    #     if mode == "distance":
    #         obs_noisy = self.observe_distance(state, subkey)
    #         concatenate_axis = 0
    #     elif mode == "images":
    #         focal_length = 0.05
    #         max_theta = jnp.pi / 3
    #         lm_h, lm_w = 1.618, 1.0
    #         image_width, image_height = 256, 256
    #         obs_noisy = self.observe_image(
    #             state,
    #             image_width,
    #             image_height,
    #             lm_h,
    #             lm_w,
    #             focal_length,
    #             max_theta
    #         )
    #         concatenate_axis = -1
    #     if time == 0:
    #         self.previous_observations = jnp.concatenate([obs_noisy, obs_noisy], axis=0)
    #         # import ipdb; ipdb.set_trace()
    #         return jnp.concatenate([obs_noisy, obs_noisy, obs_noisy], axis=0)
    #     elif time == 1:
    #         # import ipdb; ipdb.set_trace()
    #         self.previous_observations = jnp.concatenate(
    #             [self.previous_observations[4:], obs_noisy],
    #             axis=0
    #         )
    #         return jnp.concatenate([self.previous_observations, obs_noisy], axis=0)
    #     else:
    #         d_noisy_seq = jnp.concatenate([self.previous_observations, obs_noisy])
    #         # import ipdb; ipdb.set_trace()
    #         self.previous_observations = d_noisy_seq[4:]

    #     return d_noisy_seq

    def observe_distance(self,
                         state,
                         subkey
                         ):
        pos = jnp.array([state[0], state[1]])
        d = jnp.linalg.norm(self.beacons - pos, axis=1)  # (3,)
        d_noisy = d + jax.random.uniform(
            subkey,
            shape=d.shape,
            minval=-self.range_disturbance,
            maxval=self.range_disturbance
        )
        return d_noisy

    def observe_image(self,
                      state,
                      image_width,
                      image_height,
                      lm_h,
                      lm_w,
                      focal_length=1.0,
                      max_theta=jnp.pi/3
                      ):
        """
        Draw rectangular landmarks into a panoramic image.
        Panoramic seam is at -pi/2, center is +pi/2 (positive y-axis).
        """
        def angle2pixel_centered(angle, image_width):
            """
            Map [-pi, pi] angle into pixel coordinate [0, image_width),
            with pi/2 in the center and -pi/2 at the seam.
            """
            shifted = angle + jnp.pi/2.0         # shift so -pi/2 -> 0
            shifted = shifted % (2 * jnp.pi)     # wrap to [0, 2pi)
            x = image_width - shifted / (2 * jnp.pi) * image_width
            return x.astype(int)

        image = jnp.zeros((image_height, image_width, 1))

        beacon_arr = jnp.array(self.beacons)

        # relative vectors and polar info
        range_vectors = beacon_arr[:, :2] - state[:2]
        angles = jnp.arctan2(range_vectors[:, 1], range_vectors[:, 0])
        distances = jnp.linalg.norm(range_vectors, axis=1)

        # angular half-extent for each landmark (clamped)
        theta_h = jnp.minimum(jnp.arctan2(lm_h / 2.0, distances), max_theta)
        theta_w = jnp.minimum(jnp.arctan2(lm_w / 2.0, distances), max_theta)

        # observed size in "object space"
        # h_obs = 2.0 * jnp.tan(theta_h) * focal_length
        # w_obs = 2.0 * jnp.tan(theta_w) * focal_length

        # # scale into pixels
        # h_image = 2.0 * focal_length * jnp.tan(max_theta)
        # w_image = h_image  # same FOV assumption
        # h_pixels = (h_obs / h_image) * image_height
        # w_pixels = (w_obs / w_image) * image_width

        # Full angular extent
        phi_w = 2 * theta_w
        phi_h = 2 * theta_h

        # Convert angular extent to pixel extent
        w_pixels = (phi_w / (2 * max_theta)) * image_width
        h_pixels = (phi_h / (2 * max_theta)) * image_height

        # Clamp: cannot exceed half the image
        w_pixels = jnp.minimum(w_pixels, image_width // 2)
        h_pixels = jnp.minimum(h_pixels, image_height // 2)

        # pixel centers
        x_centers = angle2pixel_centered(angles, image_width)
        y_center = image_height // 2

        # painter’s algorithm: far → near
        order = jnp.argsort(distances)[::-1]

        for idx in order:
            wc = int(jnp.round(w_pixels[idx]).item())
            hc = int(jnp.round(h_pixels[idx]).item())
            if wc <= 0 or hc <= 0:
                continue

            x0 = int(x_centers[idx] - wc // 2)
            x1 = x0 + wc

            # wrap handling in x
            x_ranges = []
            if wc >= image_width:
                x_ranges = [(0, image_width)]
            else:
                x0_mod = x0 % image_width
                x1_mod = x1 % image_width
                if x0_mod < x1_mod:
                    x_ranges = [(x0_mod, x1_mod)]
                else:
                    x_ranges = [(x0_mod, image_width), (0, x1_mod)]

            # vertical extent
            y0 = int(max(0, y_center - hc // 2))
            y1 = int(min(image_height, y_center + (hc - hc // 2)))

            for xa, xb in x_ranges:
                if xa < xb and y0 < y1:
                    fade = max(1.0 - 0.05 * distances[idx], 0.2)  # simple fade with distance
                    image = image.at[y0:y1, xa:xb, 0].set(fade)

        return image

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

    def get_smoothed_state_estimate(self, obs, alpha=0.7):
        prev_state_hat = obs[:4]
        prev_input = obs[jnp.array([4, 5])]
        obs = obs[6:]

        # print("prev_state_hat:", prev_state_hat)
        # print("prev_input:", prev_input)

        state_hat_nn = self.get_state_estimate(obs)
        state_hat_dyn = self.step(
            prev_state_hat,
            prev_input,
            jnp.array([0., 0., 0., 0.])
        )
        state_hat = alpha * state_hat_nn + (1-alpha) * state_hat_dyn
        return state_hat
