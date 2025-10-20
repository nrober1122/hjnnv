import jax.numpy as jnp
import jax

import hj_reachability as hj


class TaxiNetDynamics(hj.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 dt=0.1,
                 velocity=5.,
                 L=5.,
                 max_rudder=jnp.tan(jnp.deg2rad(7.)),
                 max_position_disturbance=0.02,
                 max_yaw_disturbance=0.01,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.v = velocity
        self.L = L
        self.dt = dt

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_rudder]),
                                        jnp.array([max_rudder]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(
                jnp.array([
                    -max_position_disturbance,
                    -max_yaw_disturbance
                ]),
                jnp.array([
                    max_position_disturbance,
                    max_yaw_disturbance
                ])
            )

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        # Handle batch or single input
        theta = state[..., 1]
        dxdt = jnp.stack([self.v * jnp.sin(theta),
                          jnp.zeros_like(theta)], axis=-1)
        return dxdt

    def control_jacobian(self, state, time):
        # Shape: (..., 2, 1)
        return jnp.broadcast_to(
            jnp.array([[0.], [self.v/self.L]]),
            state.shape[:-1] + (2, 1)
        )

    def disturbance_jacobian(self, state, time):
        # Shape: (..., 2, 2)
        return jnp.broadcast_to(
            jnp.array([[1., 0.], [0., 1.]]),
            state.shape[:-1] + (2, 2)
        )

    def get_observation(self, state, time):
        return state

    def step(self, state, control, disturbance, time=0.):
        # If batched, vectorize automatically
        if state.ndim > 1:
            return jax.vmap(lambda s, u, d: self.step(s, u, d, time))(state, control, disturbance)
        
        dxdt = self(state, control, disturbance, time)
        return state + dxdt * self.dt

    # def open_loop_dynamics(self, state, time):
    #     _, theta = state
    #     return jnp.array([self.v * jnp.sin(theta), 0])

    # def control_jacobian(self, state, time):
    #     return jnp.array([
    #         [0.],
    #         [self.v/self.L],
    #     ])

    # def disturbance_jacobian(self, state, time):
    #     return jnp.array([
    #         [1., 0.],
    #         [0., 1.]
    #     ])

    # def get_observation(self, state, time):
    #     """Implements the observation function `h(x, t)`."""
    #     return state

    # def step(self, state, control, disturbance, time=0.):
    #     dxdt = self(state, control, disturbance, time)
    #     next_state = state + dxdt * self.dt
    #     return next_state
