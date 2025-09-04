import numpy as np
from scipy.interpolate import CubicSpline


class DesiredTrajectory:
    def __init__(self, traj_type='spiral'):
        self.traj_type = traj_type

    def generate_trajectory(self):
        if self.traj_type == "lawnmower":
            return self._generate_lawnmower_trajectory()
        elif self.traj_type == "spiral":
            return self._generate_spiral_trajectory()
        elif self.traj_type == "circle":
            return self._generate_circle_trajectory()
        else:
            raise ValueError(f"Unknown trajectory type: {self.traj_type}")

    def _generate_lawnmower_trajectory(self):
        square_size = 9
        lane_width = 1.0
        num_lanes = int(square_size / lane_width) + 1
        points_per_lane = 50
        T_lane = 15.0  # seconds per lane

        waypoints_x = []
        waypoints_y = []
        time_stamps = []

        t = 0.0
        for i in range(num_lanes):
            y = i * lane_width
            if i % 2 == 0:
                x_start, x_end = 0, square_size
            else:
                x_start, x_end = square_size, 0
            
            # Only append the start of the lane if this is the first lane or first point
            if i == 0:
                waypoints_x.append(x_start)
                waypoints_y.append(y)
                time_stamps.append(t)
            
            # Append end of the lane
            t += T_lane
            waypoints_x.append(x_end)
            waypoints_y.append(y)
            time_stamps.append(t)

            if i == num_lanes - 1:
                waypoints_x.append(x_start)
                waypoints_y.append(y)
                time_stamps.append(t + T_lane)

        waypoints_x = np.array(waypoints_x)
        waypoints_y = np.array(waypoints_y)
        time_stamps = np.array(time_stamps)

        # Cubic splines with zero velocities at start and end
        spline_x = CubicSpline(time_stamps, waypoints_x + 0.5, bc_type=((1,0.0),(1,0.0)))
        spline_y = CubicSpline(time_stamps, waypoints_y + 0.5, bc_type=((1,0.0),(1,0.0)))

        return spline_x, spline_y

    def _generate_spiral_trajectory(self):
        # geometry + speed params
        center = 5.0
        radius = 5.0
        v_max = 1.0
        circle_start_angle = np.pi/6  # change as you like

        cx = cy = center
        theta = circle_start_angle
        s = np.sin(theta)

        # fillet geometry (tangent to horizontal line y=cy and big circle)
        r_f = radius * s / (1.0 + s + 1e-12)
        Cf_x = cx + (radius - r_f) * np.cos(theta)
        Cf_y = cy + (radius - r_f) * np.sin(theta)

        # join point between horizontal line and the fillet
        Sx, Sy = Cf_x, cy

        # containers
        waypoints_x = []
        waypoints_y = []
        time_stamps = []
        s_total = 0.0

        # --- smooth accel profile: constant acceleration up to s_accel then cruise ---
        s_accel = radius * 0.5              # distance over which we accelerate
        a = 0.5 * v_max**2 / (s_accel + 1e-12)
        t_accel = v_max / a

        def time_from_s(s_local):
            if s_local <= 0.0:
                return 0.0
            if s_local < s_accel:
                return np.sqrt(2.0 * s_local / a)
            else:
                return t_accel + (s_local - s_accel) / v_max

        def append_path_points(xs, ys):
            # nonlocal_vars = {}
            nonlocal s_total, waypoints_x, waypoints_y, time_stamps
            # global s_total
            for (x, y) in zip(xs, ys):
                if not waypoints_x:
                    waypoints_x.append(x)
                    waypoints_y.append(y)
                    time_stamps.append(0.0)
                    continue
                dx = x - waypoints_x[-1]
                dy = y - waypoints_y[-1]
                ds = np.hypot(dx, dy)
                s_prev = s_total
                s_total += ds
                t_new = time_from_s(s_total)
                waypoints_x.append(x)
                waypoints_y.append(y)
                time_stamps.append(t_new)

        # --- Step 1: line from center to S ---
        num_line_points = 40
        if Sx > cx + 1e-12:
            xs = np.linspace(cx, Sx, num_line_points)
            ys = np.full_like(xs, cy)
        else:
            xs = np.array([Sx])
            ys = np.array([Sy])
        append_path_points(xs, ys)

        # --- Step 2: fillet arc ---
        num_arc_points = 80
        phis = np.linspace(-np.pi/2, theta, num_arc_points + 1)[1:]
        xs = Cf_x + r_f * np.cos(phis)
        ys = Cf_y + r_f * np.sin(phis)
        append_path_points(xs, ys)

        # --- Step 3: big circle ---
        num_circle_points = 200
        thetas = np.linspace(theta, theta + 2*np.pi, num_circle_points + 1)[1:]
        xs = cx + radius * np.cos(thetas)
        ys = cy + radius * np.sin(thetas)
        append_path_points(xs, ys)

        # convert to numpy arrays
        waypoints_x = np.array(waypoints_x)
        waypoints_y = np.array(waypoints_y)
        time_stamps = np.array(time_stamps)

        # --- spline ---
        spline_x = CubicSpline(time_stamps, waypoints_x, bc_type="clamped")
        spline_y = CubicSpline(time_stamps, waypoints_y, bc_type="clamped")

        return spline_x, spline_y

    def _generate_circle_trajectory(self):
        # Circle trajectory
        center = 5.0
        radius = 5.0
        vel = 1.0
        
        def circle_trajectory(t, deriv=0):
            omega = vel / radius
            if deriv == 0:
                x = center + radius * np.cos(omega * t)
                y = center + radius * np.sin(omega * t)
            elif deriv == 1:
                x = -radius * omega * np.sin(omega * t)
                y = radius * omega * np.cos(omega * t)
            elif deriv == 2:
                x = -radius * omega**2 * np.cos(omega * t)
                y = -radius * omega**2 * np.sin(omega * t)
            return x, y

        def spline_x(t, deriv=0):
            x, _ = circle_trajectory(t, deriv)
            return x

        def spline_y(t, deriv=0):
            _, y = circle_trajectory(t, deriv)
            return y

        return spline_x, spline_y

# # Parameters
# traj_type = "spiral"  # "lawnmower" or "spiral"

# if traj_type == "lawnmower":
#     square_size = 9
#     lane_width = 1.0
#     num_lanes = int(square_size / lane_width) + 1
#     points_per_lane = 50
#     T_lane = 15.0  # seconds per lane

#     waypoints_x = []
#     waypoints_y = []
#     time_stamps = []

#     t = 0.0
#     for i in range(num_lanes):
#         y = i * lane_width
#         if i % 2 == 0:
#             x_start, x_end = 0, square_size
#         else:
#             x_start, x_end = square_size, 0
        
#         # Only append the start of the lane if this is the first lane or first point
#         if i == 0:
#             waypoints_x.append(x_start)
#             waypoints_y.append(y)
#             time_stamps.append(t)
        
#         # Append end of the lane
#         t += T_lane
#         waypoints_x.append(x_end)
#         waypoints_y.append(y)
#         time_stamps.append(t)

#         if i == num_lanes - 1:
#             waypoints_x.append(x_start)
#             waypoints_y.append(y)
#             time_stamps.append(t + T_lane)

#     waypoints_x = np.array(waypoints_x)
#     waypoints_y = np.array(waypoints_y)
#     time_stamps = np.array(time_stamps)

#     # Cubic splines with zero velocities at start and end
#     spline_x = CubicSpline(time_stamps, waypoints_x + 0.5, bc_type=((1,0.0),(1,0.0)))
#     spline_y = CubicSpline(time_stamps, waypoints_y + 0.5, bc_type=((1,0.0),(1,0.0)))

#     t_samples = np.linspace(0, time_stamps[-1], num_lanes * points_per_lane)
#     x_traj = spline_x(t_samples)
#     y_traj = spline_y(t_samples)
#     vx_traj = spline_x(t_samples, 1)
#     vy_traj = spline_y(t_samples, 1)
#     ax_traj = spline_x(t_samples, 2)
#     ay_traj = spline_y(t_samples, 2)
    
# elif traj_type == "spiral":
#     # geometry + speed params
#     center = 5.0
#     radius = 5.0
#     v_max = 1.0
#     circle_start_angle = np.pi/6  # change as you like

#     cx = cy = center
#     theta = circle_start_angle
#     s = np.sin(theta)

#     # fillet geometry (tangent to horizontal line y=cy and big circle)
#     r_f = radius * s / (1.0 + s + 1e-12)
#     Cf_x = cx + (radius - r_f) * np.cos(theta)
#     Cf_y = cy + (radius - r_f) * np.sin(theta)

#     # join point between horizontal line and the fillet
#     Sx, Sy = Cf_x, cy

#     # containers
#     waypoints_x = []
#     waypoints_y = []
#     time_stamps = []
#     s_total = 0.0

#     # --- smooth accel profile: constant acceleration up to s_accel then cruise ---
#     s_accel = radius * 0.5              # distance over which we accelerate
#     a = 0.5 * v_max**2 / (s_accel + 1e-12)
#     t_accel = v_max / a

#     def time_from_s(s_local):
#         if s_local <= 0.0:
#             return 0.0
#         if s_local < s_accel:
#             return np.sqrt(2.0 * s_local / a)
#         else:
#             return t_accel + (s_local - s_accel) / v_max

#     def append_path_points(xs, ys):
#         nonlocal_vars = {}
#         global s_total
#         for (x, y) in zip(xs, ys):
#             if not waypoints_x:
#                 waypoints_x.append(x)
#                 waypoints_y.append(y)
#                 time_stamps.append(0.0)
#                 continue
#             dx = x - waypoints_x[-1]
#             dy = y - waypoints_y[-1]
#             ds = np.hypot(dx, dy)
#             s_prev = s_total
#             s_total += ds
#             t_new = time_from_s(s_total)
#             waypoints_x.append(x)
#             waypoints_y.append(y)
#             time_stamps.append(t_new)

#     # --- Step 1: line from center to S ---
#     num_line_points = 40
#     if Sx > cx + 1e-12:
#         xs = np.linspace(cx, Sx, num_line_points)
#         ys = np.full_like(xs, cy)
#     else:
#         xs = np.array([Sx])
#         ys = np.array([Sy])
#     append_path_points(xs, ys)

#     # --- Step 2: fillet arc ---
#     num_arc_points = 80
#     phis = np.linspace(-np.pi/2, theta, num_arc_points + 1)[1:]
#     xs = Cf_x + r_f * np.cos(phis)
#     ys = Cf_y + r_f * np.sin(phis)
#     append_path_points(xs, ys)

#     # --- Step 3: big circle ---
#     num_circle_points = 200
#     thetas = np.linspace(theta, theta + 2*np.pi, num_circle_points + 1)[1:]
#     xs = cx + radius * np.cos(thetas)
#     ys = cy + radius * np.sin(thetas)
#     append_path_points(xs, ys)

#     # convert to numpy arrays
#     waypoints_x = np.array(waypoints_x)
#     waypoints_y = np.array(waypoints_y)
#     time_stamps = np.array(time_stamps)

#     # --- spline ---
#     spline_x = CubicSpline(time_stamps, waypoints_x, bc_type="clamped")
#     spline_y = CubicSpline(time_stamps, waypoints_y, bc_type="clamped")

#     t_samples = np.linspace(0.0, time_stamps[-1], 500)
#     x_traj = spline_x(t_samples)
#     y_traj = spline_y(t_samples)
#     vx_traj = spline_x(t_samples, 1)
#     vy_traj = spline_y(t_samples, 1)
#     ax_traj = spline_x(t_samples, 2)
#     ay_traj = spline_y(t_samples, 2)
#     # center = 5.0
#     # radius = 5.0
#     # vel = 1.0
#     # circle_start_angle = np.pi/6  # e.g., π/6; works up to near π (see note)

#     # cx = cy = center
#     # theta = circle_start_angle
#     # s = np.sin(theta)

#     # # Fillet radius (tangent to line y=c and the big circle of radius R)
#     # # Add tiny eps to avoid division by zero when theta ~ 0
#     # r_f = radius * s / (1.0 + s + 1e-12)

#     # # Fillet center
#     # Cf_x = cx + (radius - r_f) * np.cos(theta)
#     # Cf_y = cy + (radius - r_f) * np.sin(theta)

#     # # Start of fillet (tangent to the horizontal line y=cy)
#     # Sx, Sy = Cf_x, cy

#     # waypoints_x, waypoints_y, time_stamps = [], [], []
#     # t = 0.0

#     # # --- Straight line from center -> S ---
#     # num_line_points = 40
#     # if Sx > cx + 1e-12:  # positive-length line; if zero, we just skip
#     #     xs = np.linspace(cx, Sx, num_line_points)
#     #     ys = np.full_like(xs, cy)
#     #     for i in range(len(xs)):
#     #         line_vel = i/len(xs) * vel
#     #         waypoints_x.append(xs[i])
#     #         waypoints_y.append(ys[i])
#     #         if i > 0:
#     #             dx = xs[i] - xs[i-1]
#     #             dy = ys[i] - ys[i-1]
#     #             t += np.hypot(dx, dy) / line_vel
#     #         time_stamps.append(t)
#     # else:
#     #     # Start directly at S if the straight segment collapses (e.g., theta=π/2)
#     #     waypoints_x.append(Sx); waypoints_y.append(Sy); time_stamps.append(t)

#     # # --- Fillet arc from φ=-π/2 (at S) to φ=theta (merge point on big circle) ---
#     # num_arc_points = 80
#     # phis = np.linspace(-np.pi/2, theta, num_arc_points+1)[1:]  # skip duplicate S
#     # for φ in phis:
#     #     x = Cf_x + r_f * np.cos(φ)
#     #     y = Cf_y + r_f * np.sin(φ)
#     #     dx = x - waypoints_x[-1]
#     #     dy = y - waypoints_y[-1]
#     #     t += np.hypot(dx, dy) / vel
#     #     waypoints_x.append(x); waypoints_y.append(y); time_stamps.append(t)

#     # # --- Big circle from theta onward (one lap) ---
#     # num_circle_points = 200
#     # thetas = np.linspace(theta, theta + 2*np.pi, num_circle_points+1)[1:]
#     # for th in thetas:
#     #     x = cx + radius * np.cos(th)
#     #     y = cy + radius * np.sin(th)
#     #     dx = x - waypoints_x[-1]
#     #     dy = y - waypoints_y[-1]
#     #     t += np.hypot(dx, dy) / vel
#     #     waypoints_x.append(x); waypoints_y.append(y); time_stamps.append(t)

#     # waypoints_x = np.array(waypoints_x)
#     # waypoints_y = np.array(waypoints_y)
#     # time_stamps = np.array(time_stamps)

#     # # Spline the time-parameterized path (keep endpoints clamped if you want zero start/end vel)
#     # spline_x = CubicSpline(time_stamps, waypoints_x, bc_type="clamped")
#     # spline_y = CubicSpline(time_stamps, waypoints_y, bc_type="clamped")

#     # t_samples = np.linspace(0, time_stamps[-1], 500)
#     # x_traj = spline_x(t_samples)
#     # y_traj = spline_y(t_samples)
#     # vx_traj = spline_x(t_samples, 1)
#     # vy_traj = spline_y(t_samples, 1)
#     # ax_traj = spline_x(t_samples, 2)
#     # ay_traj = spline_y(t_samples, 2)

# elif traj_type == "circle":
#     # Circle trajectory
#     center = 5.0
#     radius = 5.0
#     vel = 1.0
    
#     def circle_trajectory(t, deriv=0):
#         omega = vel / radius
#         if deriv == 0:
#             x = center + radius * np.cos(omega * t)
#             y = center + radius * np.sin(omega * t)
#         elif deriv == 1:
#             x = -radius * omega * np.sin(omega * t)
#             y = radius * omega * np.cos(omega * t)
#         elif deriv == 2:
#             x = -radius * omega**2 * np.cos(omega * t)
#             y = -radius * omega**2 * np.sin(omega * t)
#         return x, y

#     def spline_x(t, deriv=0):
#         x, _ = circle_trajectory(t, deriv)
#         return x

#     def spline_y(t, deriv=0):
#         _, y = circle_trajectory(t, deriv)
#         return y