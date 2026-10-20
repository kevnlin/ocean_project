"""117: the kernel's long axis, its alignment with SSH isolines, and the SSH gradient on the grid."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "align117", Path(__file__).resolve().parents[1] / "experiments/synthetic/117_kernel_alignment.py")
align = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align)


def rotated(major, minor, angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.array([[c, -s], [s, c]]) @ np.diag([major ** 2, minor ** 2]) @ np.array([[c, s], [-s, c]])
    return np.array([m[0, 0], m[0, 1], m[1, 1]])


def test_the_long_axis_and_semi_axes_of_a_length_scale_matrix():
    theta, major, minor = align.axes(np.stack([rotated(300., 100., 0.), rotated(300., 100., np.pi / 4)]))
    assert theta == pytest.approx([0., np.pi / 4]) and major == pytest.approx([300., 300.])
    assert minor == pytest.approx([100., 100.])


def test_alignment_is_one_along_the_isolines_and_zero_across():
    east_axis = np.zeros(2)
    gradient = np.array([[0., 1.], [1., 0.]])                 # northward: isolines run east; eastward: they run north
    assert align.alignment(east_axis, gradient) == pytest.approx([1., 0.], abs=1e-12)
    assert align.alignment(np.array([np.pi]), np.array([[0., -2.]])) == pytest.approx([1.])


def test_random_orientations_average_two_over_pi():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, 200000)
    gradient = rng.normal(size=(200000, 2))
    assert align.alignment(theta, gradient).mean() == pytest.approx(2 / np.pi, abs=.005)


def test_the_gradient_of_a_field_rising_northward():
    grid_lat, grid_lon = np.arange(-89.5, 90, 1.), np.arange(.5, 360, 1.)
    field = np.repeat(grid_lat[:, None], len(grid_lon), 1) * 2.
    gradient = align.ssh_gradient(field, np.array([10.2, -40.]), np.array([-60., 200.3]), grid_lat, grid_lon)
    assert gradient[:, 1] == pytest.approx(2. / align.KM_PER_DEGREE) and gradient[:, 0] == pytest.approx(0.)


def test_tilt_correlation_separates_turning_with_the_flow_from_zonal_elongation():
    rng = np.random.default_rng(1)
    phi = rng.uniform(-np.pi / 3, np.pi / 3, 50000)                  # isoline angles from east
    gradient = np.stack([np.sin(phi), -np.cos(phi)], -1)             # gradients whose isolines lie at phi
    assert align.tilt_correlation(phi, gradient) == pytest.approx(1., abs=1e-9)
    assert abs(align.tilt_correlation(np.zeros_like(phi) + rng.normal(scale=.05, size=phi.shape), gradient)) < .02
    block = align.block(np.zeros(4), np.full(4, 300.), np.full(4, 100.), np.array([[0., 1.], [1., 0.], [0., 1.], [1., 0.]]))
    assert block["always_east"] == pytest.approx(block["alignment"]) and block["zonal_orientation"] == pytest.approx(1.)
