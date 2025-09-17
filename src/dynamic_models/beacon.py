import jax.numpy as jnp
import jax
import numpy as np

import hj_reachability as hj
import torch

from dynamic_models.dynamics import HJNNVDynamics
from learned_models.beacon.estimators import MLP, CNN

from utils.mlp2jax import torch_mlp2jax
from utils.torch2jaxmodel import torch_to_jax_model


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
                 obs_type="distances",
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
        self.obs_type = obs_type
        self.image_size = (128, 16)  # (width, height)
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

        if self.obs_type == "distances":
            obs_noisy = self.observe_distance(state, subkey)  # shape (num_beacons,)
            num_beacons = obs_noisy.shape[0]
            history_len = 3
            concatenate_axis = 0

        elif self.obs_type == "images":
            focal_length = 0.05
            max_theta = jnp.pi / 2
            lm_h, lm_w = 1.618*4, 1.0*0.5
            # image_width, image_height = 128, 128
            image_width, image_height = self.image_size
            # image_width, image_height = 64, 64
            obs_noisy = self.observe_image(
                state,
                image_width,
                image_height,
                lm_h,
                lm_w,
                subkey,
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
        if self.obs_type == "distances":
            # drop oldest block of beacons, append newest
            self.previous_observations = jnp.concatenate(
                [self.previous_observations[num_beacons:], obs_noisy],
                axis=concatenate_axis
            )

        elif self.obs_type == "images":
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
                      subkey,
                      focal_length=1.0,
                      max_theta=jnp.pi/3,
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

        # image = jnp.zeros((image_height, image_width, 1))
        image = jax.random.normal(subkey, (image_height, image_width, 1)) * self.range_disturbance
        image = jnp.clip(image, 0.0, 1.0)

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
        # h_pixels = jnp.minimum(h_pixels, image_height // 2)

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
                    fade = max(1.0 - 0.05 * distances[idx], 0.5)  # simple fade with distance
                    # fade = 1.0
                    image = image.at[y0:y1, xa:xb, 0].set(fade)

        return image
    
    def observe_image_realistic(
        self,
        state,
        image_width,
        image_height,
        lm_h,
        lm_w,
        focal_length=1.0,
        max_theta=jnp.pi/3,
        noise_sigma=0.02,
        interior_gray=0.6,
        base_sigma=1.0,
        fade_strength=0.0,  # set to 0.0 for no fading; small 0.1-0.2 is optional
    ):
        """
        Draw landmarks into a panoramic image with:
        - 360° panoramic mapping
        - Solid white edges
        - Slightly gray interior
        - Distance-dependent Gaussian blur
        - Optional subtle fading with distance
        - Background noise
        - Occlusion (nearer beacons overwrite farther)
        """
        def angle2pixel_centered(angle, image_width):
            """Map [-pi, pi] angle into pixel coordinate [0, image_width)."""
            shifted = angle + jnp.pi / 2.0
            shifted = shifted % (2 * jnp.pi)
            x = image_width - shifted / (2 * jnp.pi) * image_width
            return x.astype(int)

        # Initialize image with background noise
        key = jax.random.PRNGKey(0)
        image = jax.random.normal(key, (image_height, image_width, 1)) * noise_sigma
        image = jnp.clip(image, 0.0, 1.0)

        beacon_arr = jnp.array(self.beacons)

        # Relative vectors and polar coordinates
        range_vectors = beacon_arr[:, :2] - state[:2]
        angles = jnp.arctan2(range_vectors[:, 1], range_vectors[:, 0])
        distances = jnp.linalg.norm(range_vectors, axis=1)

        # Angular half-extent clamped
        theta_h = jnp.minimum(jnp.arctan2(lm_h / 2.0, distances), max_theta)
        theta_w = jnp.minimum(jnp.arctan2(lm_w / 2.0, distances), max_theta)

        # Pixel size
        phi_w = 2 * theta_w
        phi_h = 2 * theta_h
        w_pixels = (phi_w / (2 * max_theta)) * image_width
        h_pixels = (phi_h / (2 * max_theta)) * image_height
        w_pixels = jnp.minimum(w_pixels, image_width // 2)
        h_pixels = jnp.minimum(h_pixels, image_height // 2)

        # Pixel centers
        x_centers = angle2pixel_centered(angles, image_width)
        y_center = image_height // 2

        # Painter’s algorithm: far → near
        order = jnp.argsort(distances)[::-1]

        for idx in order:
            wc = int(jnp.round(w_pixels[idx]).item())
            hc = int(jnp.round(h_pixels[idx]).item())
            if wc <= 0 or hc <= 0:
                continue

            x0 = int(x_centers[idx] - wc // 2)
            x1 = x0 + wc
            y0 = int(max(0, y_center - hc // 2))
            y1 = int(min(image_height, y_center + (hc - hc // 2)))

            # Wrap handling in x
            if wc >= image_width:
                x_ranges = [(0, image_width)]
            else:
                x0_mod = x0 % image_width
                x1_mod = x1 % image_width
                if x0_mod < x1_mod:
                    x_ranges = [(x0_mod, x1_mod)]
                else:
                    x_ranges = [(x0_mod, image_width), (0, x1_mod)]

            # Fade factor with distance (optional subtle fade)
            fade = jnp.clip(1.0 - fade_strength * distances[idx], 0.2, 1.0)

            for xa, xb in x_ranges:
                xs = jnp.arange(xb - xa)
                ys = jnp.arange(y1 - y0)
                X, Y = jnp.meshgrid(xs, ys)

                # Rectangular edge + gray interior
                blob = jnp.full((y1 - y0, xb - xa), interior_gray * fade)
                edge_thick = max(1, int(0.1 * min(wc, hc)))
                blob = blob.at[:edge_thick, :].set(fade)         # top
                blob = blob.at[-edge_thick:, :].set(fade)        # bottom
                blob = blob.at[:, :edge_thick].set(fade)         # left
                blob = blob.at[:, -edge_thick:].set(fade)        # right

                # Distance-dependent Gaussian blur
                sigma_x = base_sigma * (1.0 + 0.05 * distances[idx])
                sigma_y = base_sigma * (1.0 + 0.05 * distances[idx])
                dx = X - (xb - xa) / 2
                dy = Y - (y1 - y0) / 2
                blur = jnp.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))
                blob = jnp.maximum(blob, blob * blur)

                # Merge with image (occlusion)
                img_slice = image[y0:y1, xa:xb, 0]
                updated_slice = jnp.maximum(img_slice, blob)
                image = image.at[y0:y1, xa:xb, 0].set(updated_slice)

        # Final clip
        image = jnp.clip(image, 0.0, 1.0)
        return image

    # def observe_image_realistic(self,
    #                             state,
    #                             image_width,
    #                             image_height,
    #                             lm_h,
    #                             lm_w,
    #                             focal_length=1.0,
    #                             max_theta=jnp.pi/3,
    #                             noise_sigma=0.02):
    #     """
    #     Draw landmarks into a panoramic image with:
    #     - Distance-dependent Gaussian blur
    #     - Beacon edges
    #     - Background noise
    #     Panoramic seam at -pi/2, center at +pi/2 (positive y-axis)
    #     """
    #     def angle2pixel_centered(angle, image_width):
    #         shifted = angle + jnp.pi / 2.0
    #         shifted = shifted % (2 * jnp.pi)
    #         x = image_width - shifted / (2 * jnp.pi) * image_width
    #         return x.astype(int)

    #     key = jax.random.PRNGKey(0)  # optional: make deterministic or pass as arg
    #     image = jax.random.normal(key, (image_height, image_width, 1)) * noise_sigma
    #     image = jnp.clip(image, 0.0, 1.0)  # initial background noise

    #     beacon_arr = jnp.array(self.beacons)

    #     # Relative vectors and polar info
    #     range_vectors = beacon_arr[:, :2] - state[:2]
    #     angles = jnp.arctan2(range_vectors[:, 1], range_vectors[:, 0])
    #     distances = jnp.linalg.norm(range_vectors, axis=1)

    #     # Angular half-extent for each landmark (clamped)
    #     theta_h = jnp.minimum(jnp.arctan2(lm_h / 2.0, distances), max_theta)
    #     theta_w = jnp.minimum(jnp.arctan2(lm_w / 2.0, distances), max_theta)

    #     # Convert angular extent to pixel extent
    #     phi_w = 2 * theta_w
    #     phi_h = 2 * theta_h
    #     w_pixels = (phi_w / (2 * max_theta)) * image_width
    #     h_pixels = (phi_h / (2 * max_theta)) * image_height

    #     # Clamp
    #     w_pixels = jnp.minimum(w_pixels, image_width // 2)
    #     h_pixels = jnp.minimum(h_pixels, image_height // 2)

    #     # Pixel centers
    #     x_centers = angle2pixel_centered(angles, image_width)
    #     y_center = image_height // 2

    #     # Painter’s algorithm: far → near (so nearest occludes farther)
    #     order = jnp.argsort(distances)[::-1]

    #     for idx in order:
    #         wc = int(jnp.round(w_pixels[idx]).item())
    #         hc = int(jnp.round(h_pixels[idx]).item())
    #         if wc <= 0 or hc <= 0:
    #             continue

    #         # Compute vertical and horizontal ranges
    #         x0 = int(x_centers[idx] - wc // 2)
    #         x1 = x0 + wc
    #         y0 = int(max(0, y_center - hc // 2))
    #         y1 = int(min(image_height, y_center + (hc - hc // 2)))

    #         # Wrap handling in x
    #         if wc >= image_width:
    #             x_ranges = [(0, image_width)]
    #         else:
    #             x0_mod = x0 % image_width
    #             x1_mod = x1 % image_width
    #             if x0_mod < x1_mod:
    #                 x_ranges = [(x0_mod, x1_mod)]
    #             else:
    #                 x_ranges = [(x0_mod, image_width), (0, x1_mod)]

    #         # Distance-dependent intensity fade
    #         fade = jnp.clip(1.0 / (1.0 + 0.1 * distances[idx]), 0.2, 1.0)

    #         # Create Gaussian blur for the beacon
    #         sigma_x = wc / 2.0
    #         sigma_y = hc / 2.0

    #         for xa, xb in x_ranges:
    #             xs = jnp.arange(xa, xb)
    #             ys = jnp.arange(y0, y1)
    #             X, Y = jnp.meshgrid(xs, ys)
    #             dx = (X - x_centers[idx]) / sigma_x
    #             dy = (Y - y_center) / sigma_y
    #             blob = fade * jnp.exp(-0.5 * (dx**2 + dy**2))

    #             # # Draw edges: add a thin ring
    #             # # Create the base blob
    #             # blob = fade * jnp.exp(-0.5 * (dx**2 + dy**2))

    #             # Base rectangular edge
    #             edge_thickness = max(1, int(0.1 * min(wc, hc)))  # 10% of beacon size
    #             blob = jnp.zeros((y1-y0, xb-xa))

    #             # Draw sharp rectangle
    #             blob = blob.at[:edge_thickness, :].set(1.0)   # top
    #             blob = blob.at[-edge_thickness:, :].set(1.0)  # bottom
    #             blob = blob.at[:, :edge_thickness].set(1.0)   # left
    #             blob = blob.at[:, -edge_thickness:].set(1.0)  # right

    #             # Add glow: distance-dependent blur
    #             # sigma grows with distance
    #             sigma_glow = max(1.0, distances[idx] * 0.05)
    #             xx, yy = jnp.meshgrid(jnp.arange(xb-xa), jnp.arange(y1-y0))
    #             cx, cy = (xb-xa)/2, (y1-y0)/2
    #             dx = xx - cx
    #             dy = yy - cy
    #             glow = jnp.exp(-0.5 * (dx**2 + dy**2) / (sigma_glow**2))
    #             glow = glow * (1.0 / (1.0 + 0.1 * distances[idx]))  # fade with distance

    #             # Combine edge + glow
    #             blob = jnp.maximum(blob, glow)

    #             # Merge with image slice using occlusion
    #             image = image.at[y0:y1, xa:xb, 0].set(jnp.maximum(image[y0:y1, xa:xb, 0], blob))

    #             ###### This was decent
    #             # # Add rectangular edge
    #             # edge_thickness = max(1, int(0.1 * min(wc, hc)))  # 10% of beacon size or at least 1px
    #             # blob = blob.at[:edge_thickness, :].set(1.0)         # top edge
    #             # blob = blob.at[-edge_thickness:, :].set(1.0)        # bottom edge
    #             # blob = blob.at[:, :edge_thickness].set(1.0)         # left edge
    #             # blob = blob.at[:, -edge_thickness:].set(1.0)        # right edge

    #             # # Occlusion: nearest beacon overwrites farther ones
    #             # # slice of the image we want to update
    #             # img_slice = image[y0:y1, xa:xb, 0]

    #             # # compute max between blob and existing slice
    #             # updated_slice = jnp.maximum(img_slice, blob)

    #             # # assign back
    #             # image = image.at[y0:y1, xa:xb, 0].set(updated_slice)
    #             ######


    #     # Clip final image to [0, 1]
    #     image = jnp.clip(image, 0.0, 1.0)
    #     return image

    def load_estimator(self, model_name="simple_estimator_3t"):

        checkpoint = torch.load(
            "/home/nick/code/hjnnv/src/learned_models/beacon/estimators/"
            + model_name
            + "/best_model.pt",
            map_location="cpu",
        )

        config = checkpoint.get("config_dict", None)

        if self.obs_type == "distances" or True:
            self.estimator = MLP(in_dim=12, out_dim=4, hidden=config["hidden"])
        elif self.obs_type == "images":
            w, h = self.image_size
            self.estimator = CNN(input_channels=3, out_dim=4, hidden1=config["hidden"], hidden2=config["hidden"], H=h, W=w)

        # load just the weights
        self.estimator.load_state_dict(checkpoint["model_state_dict"])
        self.estimator.eval()

        # self.estimator = torch_mlp2jax(self.estimator)
        self.estimator = torch_to_jax_model(self.estimator)

        # optional: keep normalization params if you need them later
        self.in_mean = checkpoint.get("in_mean", None)
        self.in_std = checkpoint.get("in_std", None)
        self.beacons = jnp.array(config["beacons"])

    def get_state_estimate(self, obs):
        # obs = torch.tensor(np.array(obs)).unsqueeze(0)
        if self.obs_type == "distances":
            obs = (obs - self.in_mean) / self.in_std
        elif self.obs_type == "images":
            # obs = obs.reshape((3, 64, 128))
            obs = jnp.transpose(obs, (2, 0, 1))
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
