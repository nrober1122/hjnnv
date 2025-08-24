import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from dynamic_models.beacon import BeaconDynamics
from simulators import Simulator


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
        rng = np.random.default_rng(seed=0)
        low = np.array([-3, -3, -2, -2])
        high = np.array([12, 12, 2, 2])
        x0s = [jnp.array(rng.uniform(low, high)) for _ in range(10)]

        for x0 in x0s:
            trajectory = self.dynamics.simulate_trajectory(
                initial_state=x0,
                control_policy=control_policy,
                num_steps=100,
                use_observations=True,
            )

        results_dir = {"trajectory": trajectory}

        return results_dir

    def plot_expt_1_trajectory(self, results, show=True, save=False):
        trajectory = results["trajectory"]
        states = jnp.stack([s for s, _ in trajectory])
        plt.plot(states[:, 0], states[:, 1], marker='o')
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
