import sympy
from sympy import symbols, diff
from sympy.matrices import Matrix
import numpy as np
from scipy.integrate import solve_ivp
from manim import *


# Constants
G, M, J, c = 1, 0.5, 0.2, 1
a = J / (M * c)
rs = 2 * G * M / (c**2)

# Coordinates
theta0 = np.pi / 2
phi0 = 0
t0 = 0
r0 = 10
eps = 0

# Parameters
L = 1.619
E = 1

time_end = 45
number_of_points = 3000


coords = [symbols("t"), symbols("r"), symbols("theta"), symbols("phi")]
delta = coords[1]**2 - rs * coords[1] + a**2
sigma = coords[1]**2 + a**2 * sympy.cos(coords[2])**2
delta0 = r0**2 - r0 * rs + a**2
sigma0 = r0**2 + a**2 * np.cos(theta0)**2


v_t0 = 1 / delta0 * (((r0**2 + a**2 + 2 * M * a**2 * r0 * np.sin(theta0)**2 / sigma0) * E) - 2 * M * a * r0 * L / sigma0)
v_p0 = 1 / delta0 * (2 * M * a * r0 * E / sigma0 + (1 - 2 * M * r0 / sigma0) * L / np.sin(theta0)**2)
v_th0 = 0.005
v_r0 = -np.sqrt((eps - (-(1 - rs * r0 / sigma0) * c**2 * v_t0**2 - 2 * rs * r0 * a * np.sin(theta0)**2 / sigma0 * c * v_t0 * v_p0 + sigma0 * v_th0**2 + (r0**2 + a**2 + rs * r0 * a**2 / sigma0 * np.sin(theta0)**2) * np.sin(theta0)**2 * v_p0**2)) * delta0 / sigma0)


kerr = sympy.Matrix(4, 4, [
    -(1 - rs * coords[1] / sigma) * c**2, 0, 0, -rs * coords[1] * a * sympy.sin(coords[2])**2 / sigma * c,
    0, sigma / delta, 0, 0,
    0, 0, sigma, 0,
    -rs * coords[1] * a * sympy.sin(coords[2])**2 / sigma * c, 0, 0, (coords[1]**2 + a**2 + rs * coords[1] * a**2 / sigma * sympy.sin(coords[2])**2) * sympy.sin(coords[2])**2,
])
kerr_inv = kerr.inv()


k_christoffel = []
for i in range(4):
    list1 = []
    for j in range(4):
        list2 = []
        for k in range(4):
            christoffel = 0
            for alpha in range(4):
                christoffel += 1 / 2 * kerr_inv[i, alpha] * (diff(kerr[alpha, k], coords[j]) + diff(kerr[alpha, j], coords[k]) - diff(kerr[j, k], coords[alpha]))
            list2.append(christoffel)
        list1.append(list2)
    k_christoffel.append(list1)


velocities = [symbols("v_t"), symbols("v_r"), symbols("v_th"), symbols("v_p")]
de0, de1, de2, de3 = 0, 0, 0, 0
for alpha in range(4):
    for beta in range(4):
        de0 -= k_christoffel[0][alpha][beta] * velocities[alpha] * velocities[beta]
        de1 -= k_christoffel[1][alpha][beta] * velocities[alpha] * velocities[beta]
        de2 -= k_christoffel[2][alpha][beta] * velocities[alpha] * velocities[beta]
        de3 -= k_christoffel[3][alpha][beta] * velocities[alpha] * velocities[beta]


f_de0 = sympy.lambdify([coords[1], coords[2], velocities[0], velocities[1], velocities[2], velocities[3]], de0, "numpy")
f_de1 = sympy.lambdify([coords[1], coords[2], velocities[0], velocities[1], velocities[2], velocities[3]], de1, "numpy")
f_de2 = sympy.lambdify([coords[1], coords[2], velocities[0], velocities[1], velocities[2], velocities[3]], de2, "numpy")
f_de3 = sympy.lambdify([coords[1], coords[2], velocities[0], velocities[1], velocities[2], velocities[3]], de3, "numpy")


def solve(l, values):
    t, r, theta, phi, v_t, v_r, v_th, v_p = values

    f_t = v_t
    f_r = v_r
    f_th = v_th
    f_p = v_p

    f_v_t = f_de0(r, theta, v_t, v_r, v_th, v_p)
    f_v_r = f_de1(r, theta, v_t, v_r, v_th, v_p)
    f_v_th = f_de2(r, theta, v_t, v_r, v_th, v_p)
    f_v_p = f_de3(r, theta, v_t, v_r, v_th, v_p)

    return f_t, f_r, f_th, f_p, f_v_t, f_v_r, f_v_th, f_v_p


sol = solve_ivp(
    solve, (0, time_end), (t0, r0, theta0, phi0, v_t0, v_r0, v_th0, v_p0),
    t_eval=np.linspace(0, time_end, number_of_points), rtol=1e-8, atol=1e-10, method="DOP853",
)


class Kerr2D(Scene):
    def construct(self):
        x = sol.y[1] * np.cos(sol.y[3])
        y = sol.y[1] * np.sin(sol.y[3])
        points = np.column_stack((x, y, np.zeros(len(x))))

        scale = max(np.max(np.abs(x)), np.max(np.abs(y)), 3) / 5
        points = points / scale
        outer_horizon = Circle(radius=(rs + np.sqrt(rs**2 - 4 * a**2)) / (2 * scale), color=RED, fill_opacity=0.35)
        inner_horizon = Circle(radius=(rs - np.sqrt(rs**2 - 4 * a**2)) / (2 * scale), color=MAROON_B)
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
        self.add(axes, outer_horizon, inner_horizon, Dot(ORIGIN, color=BLACK, radius=0.07), path, particle)
        self.play(time.animate.set_value(sol.t[-1]), run_time=5, rate_func=linear)
        self.wait(1)


class Kerr3D(ThreeDScene):
    def construct(self):
        x = sol.y[1] * np.sin(sol.y[2]) * np.cos(sol.y[3])
        y = sol.y[1] * np.sin(sol.y[2]) * np.sin(sol.y[3])
        z = sol.y[1] * np.cos(sol.y[2])
        points = np.column_stack((x, y, z))

        scale = max(np.max(np.abs(points)), 3) / 4
        points = points / scale
        axes = ThreeDAxes(x_range=[-10, 10, 1], y_range=[-10, 10, 1], z_range=[-5, 5, 1], x_length=7, y_length=7, z_length=5)
        outer_horizon_radius = (rs + np.sqrt(rs**2 - 4 * a**2)) / (2 * scale)
        inner_horizon_radius = (rs - np.sqrt(rs**2 - 4 * a**2)) / (2 * scale)
        outer_horizon = Sphere(radius=outer_horizon_radius, fill_color=RED, stroke_width=0, checkerboard_colors=False)
        inner_horizon = Sphere(radius=inner_horizon_radius, fill_color=MAROON_B, stroke_width=0, checkerboard_colors=False)
        outer_horizon.set_opacity(0.35)
        inner_horizon.set_opacity(0.75)
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
        self.add(axes, outer_horizon, inner_horizon, path, particle, start_point)
        self.play(time.animate.set_value(sol.t[-1]), run_time=5, rate_func=linear)
        self.wait(1)
