import abc
import numpy as np
import jax.numpy as jnp
import jax

import hj_reachability as hj
import torch


class HJNNVDynamics(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 dt=0.1,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None,
                 random_seed=0):
        self.dt = dt
        self.key = jax.random.PRNGKey(random_seed)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time):
        """Implements the open loop dynamics `f(x, t)`."""

    @abc.abstractmethod
    def control_jacobian(self, state, time):
        """Implements the control Jacobian `G_u(x, t)`."""

    @abc.abstractmethod
    def disturbance_jacobian(self, state, time):
        """Implements the disturbance Jacobian `G_d(x, t)`."""

    @abc.abstractmethod
    def get_observation(self, state, time):
        """Implements the observation function `h(x, t)`."""

    def step(self, state, control, disturbance, time=0.):
        dxdt = self(state, control, disturbance, time)
        next_state = state + dxdt * self.dt
        return next_state

    def get_random_disturbance(self):
        if isinstance(self.disturbance_space, hj.sets.Box):
            self.key, subkey = jax.random.split(self.key)
            lo = self.disturbance_space.lo
            hi = self.disturbance_space.hi
            return jax.random.uniform(subkey, shape=lo.shape, minval=lo, maxval=hi)
        else:
            raise NotImplementedError(
                "Only Box disturbance space is supported for random sampling."
            )

    def set_random_seed(self, random_seed):
        self.key = jax.random.PRNGKey(random_seed)

    def simulate_trajectory(self, initial_state, control_policy, num_steps, use_observations=False):
        state = initial_state
        trajectory = [(state, self.get_observation(state, time=0))]
        for t in range(num_steps):
            if use_observations:
                obs = self.get_observation(state, time=t)
                state_hat = self.get_state_estimate(obs)
            else:
                state_hat = state
            control = control_policy(state_hat)
            disturbance = self.get_random_disturbance()
            print( "Time step:", t, "State:", state, "Control:", control, "Disturbance:", disturbance)
            state = self.step(state, control, disturbance, time=t+1)
            observation = self.get_observation(state, time=t+1)
            trajectory.append((state, observation))
        return trajectory
