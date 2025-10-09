import jax.numpy as jnp
import jax
import numpy as np

import warnings
import functools
import time

import matplotlib.pyplot as plt

import hj_reachability as hj
# from dynamic_models.beacon import BeaconDynamics
from simulators import Simulator
from hjnnv import hjnnvUncertaintyAwareFilter
from simulators.beacon_env.desired_trajectories import DesiredTrajectory


class BeaconSimulator(Simulator):
    def __init__(self, dynamics, epsilon=0.1, smoothing=True, smoothing_alpha=0.7):
        self.dynamics = dynamics
        self.smoothing = smoothing
        self.smoothing_alpha = smoothing_alpha

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
        if dynamics.range_disturbance is not epsilon:
            warnings.warn("dynamics.max_range_disturbance not epsilon: check your logic or configuration.", UserWarning)
        self.epsilon = epsilon

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
                    hj.sets.Box(
                        np.array([-5., -5., -5., -5.]),
                        np.array([15., 15., 5., 5.])),
                    (30, 30, 20, 20),
                )
        values_ = jnp.stack(
            [jnp.abs(grid.states[..., 0] - 5.),
             jnp.abs(grid.states[..., 1] - 5.)],
            axis=-1
        )
        values = -jnp.max(values_, axis=-1) + 5.

        if smoothing:
            estimation_function = functools.partial(
                dynamics.get_smoothed_state_estimate,
                alpha=smoothing_alpha
            )
        else:
            estimation_function = dynamics.get_state_estimate

        self.hjnnv_filter = hjnnvUncertaintyAwareFilter(
            dynamics=dynamics,
            pred_model=estimation_function,
            grid=grid,
            initial_values=values,
            num_controls=30,
            num_disturbances=2
        )

        super().__init__(dt=dynamics.dt)

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

    def run_expt_1(
            self,
            num_steps,
            filtering,
            bounding_method,
            seed=1,
            traj_type="spiral",
            ):
        results_dict = {
            "trajectory": None,
            "state_history": None,
            "state_hat_history": None,
            "state_bounds_history": None,
            "state_sample_history": None,
            "state_des_history": None,
        }

        def rov_controller(state, desired_traj, time):
            spline_x, spline_y = desired_traj.generate_trajectory()
            ax_des = spline_x(time, 2)
            ay_des = spline_y(time, 2)
            vx_des = spline_x(time, 1)
            vy_des = spline_y(time, 1)
            x_des = spline_x(time)
            y_des = spline_y(time)

            kp = 3.0  # Control gain
            kd = 1.0  # Derivative gain

            ux = ax_des + kp * (x_des - state[0]) + kd * (vx_des - state[2])
            uy = ay_des + kp * (y_des - state[1]) + kd * (vy_des - state[3])

            u = jnp.array([ux, uy])
            u_clipped = jnp.clip(u, -1.0, 1.0)
            return u_clipped

        state_history = []
        state_hat_history = []
        state_bounds_history = []
        state_sample_history = []
        state_des_history = []

        key = jax.random.PRNGKey(seed)
        num_samples = 5000
        self.dynamics.set_random_seed(seed)

        if traj_type == "lawnmower":
            state = jnp.array([0.5, 0.5, 0.0, 0.0])  # initial state
            num_steps = 1650
        elif traj_type == "spiral":
            state = jnp.array([5., 5., 0.0, 0.0])
            # num_steps = 650
            num_steps = 700
        elif traj_type == "circle":
            state = jnp.array([8., 5., 0.0, 0.0])
            num_steps = 650
        else:
            state = jnp.array([5.0, 5.0, 0.0, 0.0])  # initial state
            num_steps = 650

        desired_traj = DesiredTrajectory(traj_type)
        state_hat = state
        u = rov_controller(state_hat, desired_traj, 0)

        for step in range(num_steps):
            key, subkey = jax.random.split(key)
            t = step * self.dynamics.dt
            # Simulate the system dynamics
            if self.smoothing:
                obs = self.dynamics.get_observation(state, time=step)
                est_obs = jnp.concatenate([state_hat, u, obs])  # append previous state estimate and input
                state_hat = self.dynamics.get_smoothed_state_estimate(est_obs).flatten()
            else:
                obs = self.dynamics.get_observation(state, time=step)
                state_hat = self.dynamics.get_state_estimate(obs).flatten()

            state_bounds = self.hjnnv_filter.state_bounds_from_gt(
                jnp.array(state_hat),
                state
            )

            time_start = time.time()
            if self.smoothing:
                bound_epsilon = jnp.concatenate([1e-8*jnp.ones(6,), self.epsilon*jnp.ones(12,)])
                state_bounds = self.hjnnv_filter.nnv_state_bounds(
                    obs=est_obs,
                    eps=bound_epsilon
                )
            else:
                state_bounds = self.hjnnv_filter.nnv_state_bounds(
                    obs=obs,
                    eps=self.epsilon
                )

            u_nominal = rov_controller(state_hat, desired_traj, t)
            # v_star, u_star, worst_val, val_filter, min_vals, distances = hjnnv_filter.ua_filter_max_val(
            #     u_nominal,
            #     state_bounds,
            #     num_states=2,
            #     eps=1e-6
            # )
            v_star, u_star, worst_val, val_filter, min_vals, distances, _, _, = self.hjnnv_filter.ua_filter_best_u(
                u_nominal,
                state_bounds,
                num_states=2,
                delta=0.01
            )

            if val_filter < 0.0:
                u = u_star
            else:
                u = u_nominal
            time_end = time.time()

            obs_perturbations = jax.random.uniform(
                subkey, shape=(num_samples, obs.shape[0]),
                minval=-self.epsilon, maxval=self.epsilon
            )
            obs_samples = obs + obs_perturbations
            state_samples_0 = jax.vmap(lambda o: self.dynamics.get_state_estimate(o).flatten())(obs_samples)

            # state_samples_0 = jax.random.uniform(
            #     subkey, shape=(num_samples, state.shape[0]),
            #     minval=state_bounds.lo, maxval=state_bounds.hi
            # )

            disturbances = jax.vmap(lambda k: self.dynamics.get_random_disturbance())(
                jax.random.split(subkey, num_samples)
            )

            state_samples = jax.vmap(
                lambda s, d: self.dynamics.step(s, u_nominal, d, time=step)
            )(state_samples_0, disturbances)

            state_sample_history.append(state_samples)

            # print(f"Step {step}: State: {state}")
            # print(f"Step {step}: Estimated State: {state_hat}")

            # print(f"Step {step}: u_star={u}, val_filter={val_filter}")
            print(f"Step {step}: Filter execution time: {time_end - time_start:.5f} seconds")

            # Store the state
            # if step == 0:
            state_history.append(state)
            state_hat_history.append(state_hat)
            state_bounds_history.append(state_bounds)
            spline_x, spline_y = desired_traj.generate_trajectory()
            state_des_history.append(jnp.array([spline_x(t, 0), spline_y(t, 0), spline_x(t, 1), spline_y(t, 1)]))

            state = self.dynamics.step(state, u, self.dynamics.get_random_disturbance(), time=step)
        # def control_policy(state):
        #     state = state.flatten()
        #     return jnp.array([-2*(state[0]-1)-2*state[2], -2*(state[1]-1)-2*state[3]])

        # # Generate 10 randomized initial states in the specified range
        # # key = jnp.array([0, 0], dtype=jnp.uint32)
        # low = jnp.array([-3, -3, -2, -2])
        # high = jnp.array([12, 12, 2, 2])
        # keys = jax.random.split(jax.random.PRNGKey(1), 3)
        # x0s = [jax.random.uniform(k, shape=(4,), minval=low, maxval=high) for k in keys]
        # trajectories = []
        # for x0 in x0s:
        #     trajectory = self.dynamics.simulate_trajectory(
        #         initial_state=x0,
        #         control_policy=control_policy,
        #         num_steps=100,
        #         use_observations=True,
        #     )
        #     trajectories.append(trajectory)

        # results_dir = {"trajectory": trajectories}
        results_dict["state_history"] = state_history
        results_dict["state_hat_history"] = state_hat_history
        results_dict["state_bounds_history"] = state_bounds_history
        results_dict["state_des_history"] = state_des_history
        results_dict["state_sample_history"] = state_sample_history
        results_dict["desired_trajectory"] = desired_traj

        return results_dict

    def plot_expt_1_trajectory(self, results, show=True, save=False):
        state_history = results["state_history"]
        state_hat_history = results["state_hat_history"]
        state_bounds_history = results["state_bounds_history"]
        desired_traj = results["desired_trajectory"]

        state_history = jnp.stack(state_history)
        state_hat_history = jnp.stack(state_hat_history)

        num_steps = state_history.shape[0]
        spline_x, spline_y = desired_traj.generate_trajectory()

        t_samples = np.linspace(0, num_steps * self.dynamics.dt, num_steps)
        x_des_history = spline_x(t_samples, 0)
        y_des_history = spline_y(t_samples, 0)



        plt.figure(figsize=(10, 8))
        plt.plot(state_history[:, 0], state_history[:, 1], label="True State Trajectory")
        plt.plot(state_hat_history[:, 0], state_hat_history[:, 1], label="Estimated State Trajectory")
        plt.plot(x_des_history, y_des_history, label="Desired State Trajectory", linestyle='--')
        plt.scatter(self.dynamics.beacons[:, 0], self.dynamics.beacons[:, 1], c='red', marker='o', label="Beacons")
        for i in range(len(state_bounds_history)):
            bounds = state_bounds_history[i]
            lo = bounds.lo
            hi = bounds.hi
            rect = plt.Rectangle(
                (lo[0], lo[1]),
                hi[0] - lo[0],
                hi[1] - lo[1],
                linewidth=1,
                edgecolor='b',
                facecolor='none',
                alpha=0.5
            )
            plt.gca().add_patch(rect)
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.title("Trajectory: True State vs Estimated State")
        plt.legend(loc="upper right")
        plt.xlim(-1, 11)
        plt.ylim(-1, 11)
        plt.grid(True)
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
