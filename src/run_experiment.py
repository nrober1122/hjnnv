import jax
import jax.numpy as jnp
import numpy as np
import functools

from IPython.display import HTML
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import ipdb

import hj_reachability as hj
from hj_reachability.finite_differences import upwind_first
jax.config.update('jax_platform_name', 'cpu')

from hjnnv import hjnnvUncertaintyAwareFilter
from dynamic_models import BeaconDynamics
from train_beacons import Config


def main():
    dynamics = BeaconDynamics()

    def control_policy(state):
        state = state.flatten()
        return jnp.array([-2*(state[0]-1)-2*state[2], -2*(state[1]-1)-2*state[3]])
    
    # Generate 10 randomized initial states in the specified range
    rng = np.random.default_rng(seed=0)
    low = np.array([-3, -3, -2, -2])
    high = np.array([12, 12, 2, 2])
    x0s = [jnp.array(rng.uniform(low, high)) for _ in range(10)]
    # x0s = [jnp.array([2,])]
    
    for x0 in x0s:
        trajectory = dynamics.simulate_trajectory(
            initial_state=x0,
            control_policy=control_policy,
            num_steps=100,
            use_observations=True,
        )

        states = [s for s, _ in trajectory]
        states = jnp.stack(states)
        plt.plot(states[:, 0], states[:, 1], marker='o')

    plt.xlabel('State 0')
    plt.ylabel('State 1')
    plt.title('Trajectory: State 0 vs State 1')
    plt.show()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()