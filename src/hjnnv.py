import functools
import jax
import jax.numpy as jnp
import numpy as np
import time

import hj_reachability as hj
import jax_verify
from jax_verify.src import bound_propagation
# import dynamic_models


class hjnnvUncertaintyAwareFilter:
    def __init__(self, dynamics, pred_model, grid, num_controls=10, num_disturbances=5):
        """
        Initializes the class with the provided system dynamics, grid, and discretization parameters.
        This constructor sets up the value function, solver settings, and discretizes the control and disturbance spaces
        for use in reachability analysis or optimal control computations.
        Args:
            dynamics: An object representing the system dynamics, expected to have `disturbance_space` and `control_space` attributes.
            grid: The computational grid object, expected to have a `states` attribute.
            num_controls (int, optional): Number of discrete control values to sample. Defaults to 10.
            num_disturbances (int, optional): Number of discrete disturbance values to sample per dimension. Defaults to 5.
            num_states (int, optional): Number of state dimensions (currently unused). Defaults to 5.
        Attributes:
            grid: Stores the provided grid object.
            dynamics: Stores the provided dynamics object.
            target_values: The computed value function at the target time.
            disturbance_vals: Array of discretized disturbance values.
            control_vals: Array of discretized control values.
        # This constructor prepares the class for reachability or control computations by
        # initializing the value function, solver settings, and discretizing the control/disturbance spaces.
        """

        values = -jnp.abs(grid.states[..., 0]) + 10
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        )

        final_time = 0.
        target_time = -10.0
        target_values = hj.step(solver_settings, dynamics, grid, final_time, values, target_time)

        self.dynamics = dynamics
        self.pred_model = pred_model
        self.grid = grid

        self.target_values = target_values

        self.disturbance_vals = self.make_grid(
            dynamics.disturbance_space.lo,
            dynamics.disturbance_space.hi,
            num_disturbances
        )

        self.control_vals = self.make_grid(
            dynamics.control_space.lo,
            dynamics.control_space.hi,
            num_controls
        )


    def make_grid(self, lo, hi, num_points):
        """
        Make a uniform grid over a box [lo, hi] in arbitrary dimensions.
        """
        d = len(lo)
        if isinstance(num_points, int):
            num_points = [num_points] * d

        axes = [jnp.linspace(lo[i], hi[i], num_points[i]) for i in range(d)]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        return jnp.stack(mesh, axis=-1).reshape(-1, d)

    def state_bounds_from_gt(self, prediction, ground_truth):
        lo = jnp.min(jnp.vstack((prediction, ground_truth)), axis=0)
        hi = jnp.max(jnp.vstack((prediction, ground_truth)), axis=0)
        return hj.sets.Box(lo, hi)

    @functools.partial(jax.jit, static_argnames=("self",))
    def nnv_state_bounds(self, image, eps):
        interval_bounds = jax_verify.IntervalBound(
            jnp.array(image - eps),
            jnp.array(image + eps)
        )
        jittable_input_bounds = interval_bounds.to_jittable()

        def bound_prop_fun(inp_bound):
            (inp_bound,) = bound_propagation.unjit_inputs(inp_bound)

            bounds = jax_verify.backward_crown_bound_propagation(self.pred_model, inp_bound)
            lower = bounds.lower
            upper = bounds.upper

            return jnp.array(lower), jnp.array(upper)

        time_start = time.time()
        out_lb, out_ub = bound_prop_fun(jittable_input_bounds)
        time_finish = time.time()
        print(f"Execution time NNV bound prop: {time_finish - time_start:.5f} seconds")

        return hj.sets.Box(out_lb.flatten(), out_ub.flatten())

    @functools.partial(jax.jit, static_argnames=("self", "num_states"))
    def ua_filter(self, u_control, state_bounds, num_states=5):
        """
        Evaluates and filters control actions based on their worst-case performance over a grid of states and disturbances.
        This method constructs a grid of states within the provided bounds, and for each candidate control input,
        it computes the minimum value of a target function (typically a value function or safety metric) over all
        possible disturbances and states. It identifies the control input that maximizes this minimum value (i.e., the
        most robust control), as well as the worst-case value among all candidate controls. It also evaluates the
        nominal control input provided by the controller.
        Args:
            u_control: The nominal control input to be evaluated (scalar or array-like).
            state_bounds: An object with 'lo' and 'hi' attributes specifying the lower and upper bounds for each state dimension.
            num_controls (int, optional): Number of candidate control inputs to evaluate. Default is 10.
            num_disturbances (int, optional): Number of disturbance samples to consider. Default is 5.
            num_states (int, optional): Number of grid points per state dimension. Default is 5.
        Returns:
            best_val: The maximum of the minimum target values across all candidate controls (float).
            best_u: The control input corresponding to 'best_val'.
            worst_val: The minimum of the minimum target values across all candidate controls (float).
            val_filter: The minimum target value for the nominal control input 'u_control' (float).
        """

        # Construct grids
        axes = [jnp.linspace(lo, hi, num_states) for lo, hi in zip(state_bounds.lo, state_bounds.hi)]

        # Create a full grid
        mesh = jnp.meshgrid(*axes, indexing="ij")

        # Stack and flatten into shape
        states = jnp.stack(mesh, axis=-1).reshape(-1, len(axes))

        num_d = self.disturbance_vals.shape[0]

        def evaluate_u(u):
            # Compute f(x, u, d) for all x and d
            u = jnp.atleast_1d(u)

            def eval_x_d(x):
                def eval_d(d):
                    d = jnp.atleast_1d(d)
                    next_state = self.dynamics.step(x, u, d, 0.)
                    val = self.grid.interpolate(self.target_values, next_state)
                    return val
                return jax.vmap(eval_d)(self.disturbance_vals)

            vals = jax.vmap(eval_x_d)(states)  # shape (num_x, num_d)
            vals = vals.reshape(-1)
            min_idx = jnp.argmin(vals)
            min_val = vals[min_idx]

            # Recover the x and d that caused this minimum
            min_x_idx = min_idx // num_d
            min_d_idx = min_idx % num_d
            return min_val, states[min_x_idx], self.disturbance_vals[min_d_idx]

        # Check nominal u value from controller
        val_filter, _, _ = evaluate_u(u_control)

        # Vectorize over all controls
        min_vals, min_states, min_disturbances = jax.vmap(evaluate_u)(self.control_vals)
        best_u_idx = jnp.argmax(min_vals)
        worst_val = jnp.min(min_vals)
        best_val = min_vals[best_u_idx]
        best_u = self.control_vals[best_u_idx]

        return best_val, best_u, worst_val, val_filter
