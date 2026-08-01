import numpy as np
from scipy.integrate import solve_ivp
from manim import *


# Constants
G = 1
c = 1
M = 0.5
rs = 2 * G * M / (c**2)

# Coordinates
theta0 = np.pi / 3
phi0 = 0
t0 = 0
r0 = 10
eps = -1

# Parameters
L = 2
E = 0.97

time_end = 1000
number_of_points = 6000


v_t0 = E / (c**2) * 1 / (1 - rs / r0)
v_p0 = L / (r0**2)
v_th0 = 0.003
inside_sqrt = (eps + v_t0**2 * (1 - rs / r0) - r0**2 * (v_th0**2 + np.sin(theta0)**2 * v_p0**2)) * (1 - rs / r0)
v_r0 = -np.sqrt(max(0, inside_sqrt))


def christoffel(a, b, c, r, theta):
    if a == 1 and b == 0 and c == 0:
        return M * (r - 2 * M) / r**3
    if a == 1 and b == 1 and c == 1:
        return -M / (r * (r - 2 * M))
    if a == 1 and b == 2 and c == 2:
        return -(r - 2 * M)
    if a == 1 and b == 3 and c == 3:
        return -(r - 2 * M) * np.sin(theta)**2
    if a == 0 and b == 1 and c == 0:
        return M / (r * (r - 2 * M))
    if a == 2 and b == 1 and c == 2:
        return 1 / r
    if a == 2 and b == 3 and c == 3:
        return -np.sin(theta) * np.cos(theta)
    if a == 3 and b == 1 and c == 3:
        return 1 / r
    if a == 3 and b == 2 and c == 3:
        return np.cos(theta) / np.sin(theta)
    return 0


def solve(l, values):
    t, r, theta, phi, v_t, v_r, v_th, v_p = values

    f_t = v_t
    f_r = v_r
    f_th = v_th
    f_p = v_p

    f_v_t = -2 * christoffel(0, 1, 0, r, theta) * v_r * v_t
    f_v_r = -(christoffel(1, 0, 0, r, theta) * v_t**2 + christoffel(1, 1, 1, r, theta) * v_r**2 + christoffel(1, 2, 2, r, theta) * v_th**2 + christoffel(1, 3, 3, r, theta) * v_p**2)
    f_v_th = -(2 * christoffel(2, 1, 2, r, theta) * v_r * v_th + christoffel(2, 3, 3, r, theta) * v_p**2)
    f_v_p = -(2 * christoffel(3, 1, 3, r, theta) * v_r * v_p + 2 * christoffel(3, 2, 3, r, theta) * v_th * v_p)

    return f_t, f_r, f_th, f_p, f_v_t, f_v_r, f_v_th, f_v_p


sol = solve_ivp(
    solve, (0, time_end), (t0, r0, theta0, phi0, v_t0, v_r0, v_th0, v_p0),
    t_eval=np.linspace(0, time_end, number_of_points), rtol=1e-8, atol=1e-10,
)


class Schwarzchild2D(Scene):
    def construct(self):
        x = sol.y[1] * np.cos(sol.y[3])
        y = sol.y[1] * np.sin(sol.y[3])
        points = np.column_stack((x, y, np.zeros(len(x))))

        scale = max(np.max(np.abs(x)), np.max(np.abs(y)), 3) / 5
        points = points / scale
        event_horizon = Circle(radius=rs / scale, color=RED, fill_opacity=0.8)
        axes = Axes(x_range=[-6, 6, 2], y_range=[-4, 4, 2], x_length=10, y_length=6, axis_config={"stroke_opacity": 0.35})
        path = VMobject(color=YELLOW, stroke_width=3)
        path.set_points_as_corners(points[:2])
        particle = Dot(points[0], color=WHITE, radius=0.08)
        time = ValueTracker(sol.t[0])

        def move_particle(dot):
            i = np.argmin(np.abs(sol.t - time.get_value()))
            dot.move_to(points[i])

        def grow_path(curve):
            i = max(2, np.argmin(np.abs(sol.t - time.get_value())))
            curve.set_points_as_corners(points[:i])

        particle.add_updater(move_particle)
        path.add_updater(grow_path)
        self.add(axes, event_horizon, Dot(ORIGIN, color=BLACK, radius=0.07), path, particle)
        self.play(time.animate.set_value(sol.t[-1]), run_time=5, rate_func=linear)
        self.wait(1)


class Schwarzchild3D(ThreeDScene):
    def construct(self):
        x = sol.y[1] * np.sin(sol.y[2]) * np.cos(sol.y[3])
        y = sol.y[1] * np.sin(sol.y[2]) * np.sin(sol.y[3])
        z = sol.y[1] * np.cos(sol.y[2])
        points = np.column_stack((x, y, z))

        scale = max(np.max(np.abs(points)), 3) / 4
        points = points / scale
        axes = ThreeDAxes(x_range=[-5, 5, 1], y_range=[-5, 5, 1], z_range=[-5, 5, 1], x_length=7, y_length=7, z_length=5)
        event_horizon = Sphere(radius=rs/scale, fill_color=RED, stroke_width=0, checkerboard_colors=False)
        event_horizon.set_opacity(0.75)
        start_point = Sphere(radius = 0.01, fill_color=RED, stroke_width=0, checkerboard_colors=False)
        start_point.move_to(points[0])
        start_point.set_opacity(1)
        path = VMobject(color=YELLOW, stroke_width=3)
        path.set_points_as_corners(points[:2])
        particle = Dot(points[0], color=WHITE, radius = 0.01)
        time = ValueTracker(sol.t[0])

        def move_particle(dot):
            i = np.argmin(np.abs(sol.t - time.get_value()))
            dot.move_to(points[i])

        def grow_path(curve):
            i = max(2, np.argmin(np.abs(sol.t - time.get_value())))
            curve.set_points_as_corners(points[:i])

        particle.add_updater(move_particle)
        path.add_updater(grow_path)
        self.set_camera_orientation(phi=65 * DEGREES, theta=70 * DEGREES, zoom=2.5)
        self.add(axes, event_horizon, path, particle)
        self.play(time.animate.set_value(sol.t[-1]), run_time=5, rate_func=linear)
        self.wait(1)
