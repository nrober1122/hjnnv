import jax.numpy as jnp
import jax
import numpy as np

import matplotlib.pyplot as plt

import hj_reachability as hj
from dynamic_models.beacon import BeaconDynamics
from simulators import Simulator
from hjnnv import hjnnvUncertaintyAwareFilter


class BeaconSimulator(Simulator):
    def __init__(self, dt=0.1):
        self.dynamics = BeaconDynamics(dt=dt)

        self.experiments = {
            1: {
                "run": self.run_expt_1,
                "plots": {
                    "trajectory": self.plot_expt_1_trajectory,
                    "velocity": self.plot_expt_1_velocity,
                },
            },
            2: {
                "run": self.run_expt_2,
                "plots": {
                    "whatever": self.plot_expt_2,
                },
            }
        }

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                np.array([-5., -5., -5., -5.]),
                np.array([15., 15., 5., 5.])),
            (50, 50, 50, 50),
        )

        # # Set up the uncertainty-aware filter
        # hjnnv_filter = hjnnvUncertaintyAwareFilter(
        #     dynamics=self.dynamics,
        #     pred_model=self.dynamics.get_state_estimate,
        #     grid=grid,
        #     num_controls=30,
        #     num_disturbances=15,
        # )

        # # Use random calculations to jit compile things before the main loop
        # v_star, u_star, worst_val, val_filter = hjnnv_filter.ua_filter(
        #         jnp.array(np.tan(np.deg2rad(0))),
        #         hj.sets.Box(
        #             jnp.array([-0.1, -0.1]),
        #             jnp.array([0.1, 0.1])
        #         ),
        #         num_states=10,
        # )
        # if settings.STATE_ESTIMATOR == 'tiny_taxinet':
        #     dummy_input = jnp.zeros((128, 1))
        # elif settings.STATE_ESTIMATOR == 'dnn':
        #     dummy_input = jnp.zeros((1, 3, 224, 224))

        # state_bounds = hjnnv_filter.nnv_state_bounds(
        #     dummy_input,
        #     0.03
        # )

        super().__init__(dt=dt)

    def run(self,
            num_steps=100,
            filtering=False,
            bounding_method=None,
            experiment_number=1,
            ):
        exp = self.experiments.get(experiment_number)
        if exp is None:
            raise ValueError(f"Unknown experiment number: {experiment_number}")
        return exp["run"](num_steps, filtering, bounding_method)

    def plot(self,
             results,
             experiment_number,
             plot_type="trajectory",
             **kwargs,
             ):
        exp = self.experiments.get(experiment_number)
        if exp is None:
            raise ValueError(f"Unknown experiment number: {experiment_number}")
        plot_func = exp["plots"].get(plot_type)
        if plot_func is None:
            raise ValueError(f"Unknown plot type: {plot_type}")
        return plot_func(results, **kwargs)

    def run_expt_1(self, num_steps, filtering, bounding_method):
        def control_policy(state):
            state = state.flatten()
            return jnp.array([-2*(state[0]-1)-2*state[2], -2*(state[1]-1)-2*state[3]])

        # Generate 10 randomized initial states in the specified range
        # key = jnp.array([0, 0], dtype=jnp.uint32)
        low = jnp.array([-3, -3, -2, -2])
        high = jnp.array([12, 12, 2, 2])
        keys = jax.random.split(jax.random.PRNGKey(1), 3)
        x0s = [jax.random.uniform(k, shape=(4,), minval=low, maxval=high) for k in keys]
        trajectories = []
        for x0 in x0s:
            trajectory = self.dynamics.simulate_trajectory(
                initial_state=x0,
                control_policy=control_policy,
                num_steps=100,
                use_observations=True,
            )
            trajectories.append(trajectory)

        results_dir = {"trajectory": trajectories}

        return results_dir

    def plot_expt_1_trajectory(self, results, show=True, save=False):
        trajectories = results["trajectory"]
        for i, trajectory in enumerate(trajectories):
            states = jnp.stack([s for s, _ in trajectory])
            plt.plot(states[:, 0], states[:, 1], marker='o', label=f"Trial {i}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Trajectory")
        if save:
            plt.savefig("traj.png")
        if show:
            plt.show()

    def plot_expt_1_velocity(self, results, show=True, save=False):
        trajectory = results["trajectory"]
        states = jnp.stack([s for s, _ in trajectory])
        plt.plot(states[:, 2], label="vx")
        plt.plot(states[:, 3], label="vy")
        plt.legend()
        plt.title("Velocities")
        if save:
            plt.savefig("vel.png")
        if show:
            plt.show()

    def run_expt_2(self, num_steps, filtering, bounding_method):
        pass

    def plot_expt_2(self, results, show=True, save=False):
        pass
