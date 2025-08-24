import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

from dynamic_models.beacon import BeaconDynamics
from simulators import Simulator


class BeaconSimulator(Simulator):
    def __init__(self, dt=0.1):
        self.dynamics = BeaconDynamics(dt=dt)

        super().__init__(dt=dt)

    def run(self, num_steps, filtering, bounding_method, experiment_number):
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

    def plot(self, results_dir, experiment_number, show, save):
        
        if experiment_number == 1:
            self.plot_expt_1(results_dir)
        elif experiment_number == 2:
            self.plot_expt_2(results_dir)

    def plot_expt_1(self, results_dir):
        trajectory = results_dir["trajectory"]

        states = [s for s, _ in trajectory]
        states = jnp.stack(states)
        plt.plot(states[:, 0], states[:, 1], marker='o')

        plt.xlabel('State 0')
        plt.ylabel('State 1')
        plt.title('Trajectory: State 0 vs State 1')
        plt.show()

    def plot_expt_2(results_dir):
        pass