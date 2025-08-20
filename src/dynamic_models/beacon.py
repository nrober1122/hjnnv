import jax.numpy as jnp
import jax

import hj_reachability as hj

from dynamic_models.dynamics import HJNNVDynamics


class BeaconDynamics(HJNNVDynamics):

    def __init__(self,
                 dt=0.1,
                 velocity=5.,
                 L=5.,
                 max_input=1.0,
                 max_position_disturbance=0.01,
                 max_vel_disturbance=0.005,
                 range_disturbance=0.02,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.v = velocity
        self.L = L
        self.dt = dt

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_input, -max_input]),
                                        jnp.array([max_input, max_input]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(
                jnp.array([
                    -max_position_disturbance,
                    -max_vel_disturbance,
                    -max_position_disturbance,
                    -max_vel_disturbance
                ]),
                jnp.array([
                    max_position_disturbance,
                    max_vel_disturbance,
                    max_position_disturbance,
                    max_vel_disturbance
                ])
            )

        self.beacons = ((0.0, 0.0), (6.0, 0.0), (0.0, 6.0))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, vx, _, vy = state
        return jnp.array([vx, 0, vy, 0])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [1., 0.],
            [0., 0.],
            [0., 1.]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.eye(4)

    def get_observation(self, state, time):
        pos = jnp.array([state[0], state[1]])
        d = jnp.linalg.norm(self.beacons - pos, axis=1)  # (3,)
        d_noisy = d + jnp.random.randn(3) * self.range_disturbance

        return d_noisy

    def step(self, state, control, disturbance, time=0.):
        dxdt = self(state, control, disturbance, time)
        next_state = state + dxdt * self.dt
        return next_state
