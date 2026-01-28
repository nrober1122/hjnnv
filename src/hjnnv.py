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
    def __init__(self,
                 dynamics,
                 pred_model,
                 grid,
                 initial_values,
                 num_controls=10,
                 num_disturbances=5,
                 time_horizon=10.0,
                 ):
        """
        Initializes the class with the provided system dynamics, grid, and discretization parameters.
        This constructor sets up the value function, solver settings, and discretizes the control and disturbance spaces
        for use in reachability analysis or optimal control computations.
        Args:
            dynamics: An object representing the system dynamics, expected to have `disturbance_space` and `control_space` attributes.
            pred_model: A neural network model used for prediction and verification.
            grid: The computational grid object, expected to have a `states` attribute.
            initial_values: The initial state values for the system.
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

        # values = -jnp.abs(grid.states[..., 0]) + 10
        solver_settings = hj.SolverSettings.with_accuracy(
            "very_high",
            hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
        )

        final_time = 0.
        target_time = -time_horizon
        target_values = hj.step(
            solver_settings,
            dynamics,
            grid,
            final_time,
            initial_values,
            target_time
        )

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
        num_points can be an int or a jnp/numpy array specifying the number of points per dimension.
        """
        d = len(lo)
        if isinstance(num_points, int):
            num_points = [num_points] * d
        elif isinstance(num_points, (jnp.ndarray, np.ndarray)):
            num_points = list(np.array(num_points).tolist())
        axes = [jnp.linspace(lo[i], hi[i], int(num_points[i])) for i in range(d)]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        return jnp.stack(mesh, axis=-1).reshape(-1, d)

        # axes = [jnp.linspace(lo[i], hi[i], num_points[i]) for i in range(d)]
        # mesh = jnp.meshgrid(*axes, indexing="ij")
        # return jnp.stack(mesh, axis=-1).reshape(-1, d)

    def state_bounds_from_gt(self, prediction, ground_truth):
        dist = jnp.abs(prediction - ground_truth)
        lo = prediction - dist
        hi = prediction + dist
        # lo = jnp.min(jnp.vstack((prediction, ground_truth)), axis=0)
        # hi = jnp.max(jnp.vstack((prediction, ground_truth)), axis=0)
        return hj.sets.Box(lo, hi)

    @functools.partial(jax.jit, static_argnames=("self",))
    def nnv_state_bounds(self, obs, eps):
        interval_bounds = jax_verify.IntervalBound(
            jnp.array(obs - eps),
            jnp.array(obs + eps)
        )
        jittable_input_bounds = interval_bounds.to_jittable()

        def bound_prop_fun(inp_bound):
            (inp_bound,) = bound_propagation.unjit_inputs(inp_bound)

            bounds = jax_verify.backward_crown_bound_propagation(self.pred_model, inp_bound)
            lower = bounds.lower
            upper = bounds.upper

            return jnp.array(lower), jnp.array(upper)

        out_lb, out_ub = bound_prop_fun(jittable_input_bounds)

        return hj.sets.Box(out_lb.flatten(), out_ub.flatten())

    @functools.partial(jax.jit, static_argnames=("self",))
    def nnv_dynamics_state_bounds(self, state, control, state_eps):
        disturbance = jnp.zeros_like(state)
        eps = jnp.concatenate([state_eps, jnp.zeros_like(control), self.dynamics.disturbance_space.hi])
        inp = jnp.concatenate([state, control, disturbance])

        interval_bounds = jax_verify.IntervalBound(
            jnp.array(inp - eps),
            jnp.array(inp + eps)
        )
        jittable_input_bounds = interval_bounds.to_jittable()

        def dynamics_bound_prop_fun(inp):
            state_propagation = inp[0:state.shape[0]]
            control_propagation = inp[state.shape[0]:state.shape[0]+control.shape[0]]
            disturbance_propagation = inp[state.shape[0]+control.shape[0]:]

            next_state = self.dynamics.step(
                state_propagation,
                control_propagation,
                disturbance_propagation,
                time=0.
            )
            return next_state

        def bound_prop_fun(inp_bound):
            (inp_bound,) = bound_propagation.unjit_inputs(inp_bound)

            bounds = jax_verify.backward_crown_bound_propagation(dynamics_bound_prop_fun, inp_bound)
            lower = bounds.lower
            upper = bounds.upper

            return jnp.array(lower), jnp.array(upper)

        out_lb, out_ub = bound_prop_fun(jittable_input_bounds)

        return hj.sets.Box(out_lb.flatten(), out_ub.flatten())

    @functools.partial(jax.jit, static_argnames=("self", "num_states"))
    def ua_filter_max_val(self, u_control, state_bounds, num_states=5, eps=1e-7):
        """
        Evaluates and filters control actions based on their worst-case performance over a grid of states and disturbances.
        This method constructs a grid of states within the provided bounds, and for each candidate control input,
        it computes the minimum value of a target function (typically a value function or safety metric) over all
        possible disturbances and states. It identifies the control input that maximizes this minimum value (i.e., the
        most robust control), as well as the worst-case value among all candidate controls. It also evaluates the
        nominal control input provided by the controller, with tie-breaking for multiple equivalent best controls
        in favor of the one closest to the nominal input.
        Args:
            u_control: The nominal control input to be evaluated (scalar or array-like).
            state_bounds: An object with 'lo' and 'hi' attributes specifying the lower and upper bounds for each state dimension.
            num_states (int, optional): Number of grid points per state dimension. Default is 5.
        Returns:
            best_val: The maximum of the minimum target values across all candidate controls (float).
            best_u: The control input corresponding to 'best_val', tie-broken to be closest to u_control (scalar or array-like).
            worst_val: The minimum of the minimum target values across all candidate controls (float).
            val_filter: The minimum target value for the nominal control input 'u_control' (float).
        """

        # Construct state grid
        axes = [jnp.linspace(lo, hi, num_states) for lo, hi in zip(state_bounds.lo, state_bounds.hi)]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        states = jnp.stack(mesh, axis=-1).reshape(-1, len(axes))

        def evaluate_u(u):
            u = jnp.atleast_1d(u)

            def eval_x_d(x):
                def eval_d(d):
                    d = jnp.atleast_1d(d)
                    next_state = self.dynamics.step(x, u, d, 0.)
                    val = self.grid.interpolate(self.target_values, next_state)
                    return val
                return jax.vmap(eval_d)(self.disturbance_vals)

            vals = jax.vmap(eval_x_d)(states)  # shape (num_x, num_d)
            min_val = jnp.min(vals)
            return min_val

        # Evaluate nominal control
        val_filter = evaluate_u(u_control)

        # Evaluate all candidate controls
        min_vals = jax.vmap(evaluate_u)(self.control_vals)
        worst_val = jnp.min(min_vals)
        best_val = jnp.max(min_vals)

        # Tie-breaking: pick best_u closest to nominal u_control
        distances = jnp.linalg.norm(self.control_vals - u_control, axis=-1) \
            if self.control_vals.ndim > 1 else jnp.abs(self.control_vals - u_control)

        # # Stack min_vals and distances for lexicographic sorting: (-value first, distance second)
        # best_idx = jnp.lexsort((distances, -min_vals))[0]
        # best_u = self.control_vals[best_idx]
        
        mask = min_vals >= (best_val - eps)  # candidates within tolerance
        # among candidates, pick closest to nominal u_control
        best_idx = jnp.argmin(jnp.where(mask, distances, jnp.inf))
        best_u = self.control_vals[best_idx]

        return best_val, best_u, worst_val, val_filter, min_vals, distances
    
    @functools.partial(jax.jit, static_argnames=("self", "num_states"))
    def ua_filter_best_u(self, u_control, state_bounds, num_states=5, delta=0.0):
        """
        Evaluates and filters control actions based on their worst-case performance over a grid of states and disturbances.
        Instead of only returning the argmax, this function finds all control inputs whose performance is within `delta`
        of the best, and then selects the one closest to the nominal control `u_control`.

        Args:
            u_control: The nominal control input to be evaluated (scalar or array-like).
            state_bounds: An object with 'lo' and 'hi' attributes specifying the lower and upper bounds for each state dimension.
            num_states (int, optional): Number of grid points per state dimension. Default is 5.
            delta (float, optional): Acceptable suboptimality tolerance. Controls within delta of the best are considered.

        Returns:
            best_val: The maximum of the minimum target values across all candidate controls (float).
            best_u: A control input within `delta` of best_val, chosen to be closest to u_control (scalar or array-like).
            worst_val: The minimum of the minimum target values across all candidate controls (float).
            val_filter: The minimum target value for the nominal control input 'u_control' (float).
        """

        # Construct state grid
        axes = [jnp.linspace(lo, hi, num_states) for lo, hi in zip(state_bounds.lo, state_bounds.hi)]
        mesh = jnp.meshgrid(*axes, indexing="ij")
        states = jnp.stack(mesh, axis=-1).reshape(-1, len(axes))

        def evaluate_u(u):
            u = jnp.atleast_1d(u)

            def eval_x_d(x):
                def eval_d(d):
                    d = jnp.atleast_1d(d)
                    next_state = self.dynamics.step(x, u, d, 0.0)
                    val = self.grid.interpolate(self.target_values, next_state)
                    return val
                return jnp.min(jax.vmap(eval_d)(self.disturbance_vals))
            return jnp.min(jax.vmap(eval_x_d)(states))

        # Evaluate nominal control
        val_filter = evaluate_u(u_control)

        # Evaluate all candidate controls
        min_vals = jax.vmap(evaluate_u)(self.control_vals)
        worst_val = jnp.min(min_vals)
        best_val = jnp.max(min_vals)

        # Compute distances to nominal
        distances = (
            jnp.linalg.norm(self.control_vals - u_control, axis=-1)
            if self.control_vals.ndim > 1
            else jnp.abs(self.control_vals - u_control)
        )

        # Mask for candidates above delta
        mask = min_vals >= delta
        any_above = jnp.any(mask)

        # If any candidate meets delta, pick closest to u_control; else pick argmax
        best_idx = jnp.where(
            any_above,
            jnp.argmin(jnp.where(mask, distances, jnp.inf)),
            jnp.argmax(min_vals)
        )

        best_u = self.control_vals[best_idx]

        return best_val, best_u, worst_val, val_filter, min_vals, distances, any_above, best_idx


    # @functools.partial(jax.jit, static_argnames=("self", "num_states"))
    # def ua_filter2(self, u_control, state_bounds, num_states=5):
    #     """
    #     Evaluates and filters control actions based on their worst-case performance over a grid of states and disturbances.
    #     This method constructs a grid of states within the provided bounds, and for each candidate control input,
    #     it computes the minimum value of a target function (typically a value function or safety metric) over all
    #     possible disturbances and states. It identifies the control input that maximizes this minimum value (i.e., the
    #     most robust control), as well as the worst-case value among all candidate controls. It also evaluates the
    #     nominal control input provided by the controller.
    #     Args:
    #         u_control: The nominal control input to be evaluated (scalar or array-like).
    #         state_bounds: An object with 'lo' and 'hi' attributes specifying the lower and upper bounds for each state dimension.
    #         num_controls (int, optional): Number of candidate control inputs to evaluate. Default is 10.
    #         num_disturbances (int, optional): Number of disturbance samples to consider. Default is 5.
    #         num_states (int, optional): Number of grid points per state dimension. Default is 5.
    #     Returns:
    #         best_val: The maximum of the minimum target values across all candidate controls (float).
    #         best_u: The control input corresponding to 'best_val'.
    #         worst_val: The minimum of the minimum target values across all candidate controls (float).
    #         val_filter: The minimum target value for the nominal control input 'u_control' (float).
    #     """

    #     # Construct grids
    #     axes = [jnp.linspace(lo, hi, num_states) for lo, hi in zip(state_bounds.lo, state_bounds.hi)]

    #     # Create a full grid
    #     mesh = jnp.meshgrid(*axes, indexing="ij")

    #     # Stack and flatten into shape
    #     states = jnp.stack(mesh, axis=-1).reshape(-1, len(axes))

    #     num_d = self.disturbance_vals.shape[0]

    #     def evaluate_u(u):
    #         # Compute f(x, u, d) for all x and d
    #         u = jnp.atleast_1d(u)

    #         def eval_x_d(x):
    #             def eval_d(d):
    #                 d = jnp.atleast_1d(d)
    #                 next_state = self.dynamics.step(x, u, d, 0.)
    #                 val = self.grid.interpolate(self.target_values, next_state)
    #                 return val
    #             return jax.vmap(eval_d)(self.disturbance_vals)

    #         vals = jax.vmap(eval_x_d)(states)  # shape (num_x, num_d)
    #         vals = vals.reshape(-1)
    #         min_idx = jnp.argmin(vals)
    #         min_val = vals[min_idx]

    #         # Recover the x and d that caused this minimum
    #         min_x_idx = min_idx // num_d
    #         min_d_idx = min_idx % num_d
    #         return min_val, states[min_x_idx], self.disturbance_vals[min_d_idx]

    #     # Check nominal u value from controller
    #     val_filter, _, _ = evaluate_u(u_control)

    #     # Vectorize over all controls
    #     min_vals, min_states, min_disturbances = jax.vmap(evaluate_u)(self.control_vals)
    #     # best_u_idx = jnp.argmax(min_vals)
    #     worst_val = jnp.min(min_vals)
    #     # best_val = min_vals[best_u_idx]
    #     # best_u = self.control_vals[best_u_idx]
    #     best_val = jnp.max(min_vals)

    #     # Find all indices with value == best_val (within tolerance)
    #     tol = 1e-6
    #     mask = jnp.isclose(min_vals, best_val, atol=tol)

    #     # Candidate controls that achieve best_val
    #     candidate_controls = self.control_vals[mask]

    #     # Distances to nominal control
    #     # distances = jnp.linalg.norm(candidate_controls - u_control, axis=-1)
    #     distances = jnp.linalg.norm(candidate_controls - u_control, axis=-1) \
    #         if candidate_controls.ndim > 1 else jnp.abs(candidate_controls - u_control)

    #     # Pick the closest
    #     closest_idx = jnp.argmin(distances)
    #     best_u = candidate_controls[closest_idx]

    #     return best_val, best_u, worst_val, val_filter
