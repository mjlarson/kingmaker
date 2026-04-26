"""
Unit tests for kingmaker.utils.

Covers _interp2d (bilinear interpolation, boundary clamping, grid-exact
recovery), map2nside, angular_distance (known angles, symmetry, self-distance),
and meshgrid2d (shape, values, dtype preservation).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kingmaker.utils import _interp2d, angular_distance, map2nside, meshgrid2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_grid(nx=5, ny=7):
    """Grid where z(x,y) = x + y — bilinear interpolation is exact."""
    xp = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    yp = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    z = xp[:, None] + yp[None, :]
    return xp, yp, z


def _make_product_grid(nx=6, ny=8):
    """Grid where z(x,y) = x * y — tests non-trivial bilinear blending."""
    xp = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    yp = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    z = xp[:, None] * yp[None, :]
    return xp, yp, z


# ---------------------------------------------------------------------------
# _interp2d — interior interpolation
# ---------------------------------------------------------------------------


class TestInterp2DInterior:
    def test_exact_at_grid_points(self):
        """Query at every grid point should return the stored value exactly."""
        xp, yp, z = _make_linear_grid()
        for i, xi in enumerate(xp):
            for j, yj in enumerate(yp):
                result = _interp2d(xi, yj, xp, yp, z)
                assert_allclose(result, z[i, j], rtol=1e-12)

    def test_midpoint_linear_grid(self):
        """Midpoint of a linear-z grid: bilinear interpolation is exact."""
        xp, yp, z = _make_linear_grid(nx=5, ny=5)
        x_mid = (xp[1] + xp[2]) / 2
        y_mid = (yp[1] + yp[2]) / 2
        expected = x_mid + y_mid
        assert_allclose(_interp2d(x_mid, y_mid, xp, yp, z), expected, rtol=1e-12)

    def test_product_grid_midpoint(self):
        """Midpoint of a product grid: bilinear approximation matches analytic."""
        xp, yp, z = _make_product_grid()
        x_mid = (xp[2] + xp[3]) / 2
        y_mid = (yp[3] + yp[4]) / 2
        # Bilinear interpolation of x*y is exact at midpoints of grid cells.
        expected = x_mid * y_mid
        assert_allclose(_interp2d(x_mid, y_mid, xp, yp, z), expected, rtol=1e-10)

    def test_array_input_shape(self):
        """Vectorised call: output shape matches input shape."""
        xp, yp, z = _make_linear_grid()
        xs = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        ys = np.array([0.2, 0.4, 0.8], dtype=np.float64)
        out = _interp2d(xs, ys, xp, yp, z)
        assert out.shape == (3,)

    def test_array_values_match_scalar(self):
        """Vectorised and scalar calls produce identical results."""
        xp, yp, z = _make_linear_grid()
        xs = np.array([0.1, 0.3, 0.7], dtype=np.float64)
        ys = np.array([0.2, 0.5, 0.9], dtype=np.float64)
        batch = _interp2d(xs, ys, xp, yp, z)
        scalar = np.array([_interp2d(xi, yi, xp, yp, z) for xi, yi in zip(xs, ys)])
        assert_allclose(batch, scalar, rtol=1e-12)

    def test_float32_supported(self):
        """float32 arrays produce finite results without error."""
        xp = np.linspace(0, 1, 5, dtype=np.float32)
        yp = np.linspace(0, 1, 5, dtype=np.float32)
        z = (xp[:, None] + yp[None, :]).astype(np.float32)
        result = _interp2d(np.float32(0.5), np.float32(0.5), xp, yp, z)
        assert np.isfinite(result)
        assert_allclose(float(result), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# _interp2d — boundary clamping
# ---------------------------------------------------------------------------


class TestInterp2DBoundary:
    def test_clamp_below_x(self):
        """Query below xp[0] should return the x=xp[0] edge value."""
        xp, yp, z = _make_linear_grid()
        assert_allclose(_interp2d(-1.0, yp[2], xp, yp, z), z[0, 2], rtol=1e-12)

    def test_clamp_above_x(self):
        """Query above xp[-1] should return the x=xp[-1] edge value."""
        xp, yp, z = _make_linear_grid()
        assert_allclose(_interp2d(2.0, yp[2], xp, yp, z), z[-1, 2], rtol=1e-12)

    def test_clamp_below_y(self):
        """Query below yp[0] should return the y=yp[0] edge value."""
        xp, yp, z = _make_linear_grid()
        assert_allclose(_interp2d(xp[2], -1.0, xp, yp, z), z[2, 0], rtol=1e-12)

    def test_clamp_above_y(self):
        """Query above yp[-1] should return the y=yp[-1] edge value."""
        xp, yp, z = _make_linear_grid()
        assert_allclose(_interp2d(xp[2], 2.0, xp, yp, z), z[2, -1], rtol=1e-12)

    def test_corner_bottom_left(self):
        """Query at (xp[0], yp[0]) returns z[0, 0], not z[1, 1]."""
        xp, yp, z = _make_product_grid()
        assert_allclose(_interp2d(xp[0], yp[0], xp, yp, z), z[0, 0], rtol=1e-12)

    def test_corner_top_right(self):
        """Query at (xp[-1], yp[-1]) returns z[-1, -1]."""
        xp, yp, z = _make_product_grid()
        assert_allclose(_interp2d(xp[-1], yp[-1], xp, yp, z), z[-1, -1], rtol=1e-12)

    def test_corner_bottom_right(self):
        """Query at (xp[-1], yp[0]) returns z[-1, 0]."""
        xp, yp, z = _make_product_grid()
        assert_allclose(_interp2d(xp[-1], yp[0], xp, yp, z), z[-1, 0], rtol=1e-12)

    def test_corner_top_left(self):
        """Query at (xp[0], yp[-1]) returns z[0, -1]."""
        xp, yp, z = _make_product_grid()
        assert_allclose(_interp2d(xp[0], yp[-1], xp, yp, z), z[0, -1], rtol=1e-12)


# ---------------------------------------------------------------------------
# map2nside
# ---------------------------------------------------------------------------


class TestMap2Nside:
    @pytest.mark.parametrize("nside", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    def test_round_trip(self, nside):
        npix = 12 * nside**2
        skymap = np.ones(npix)
        assert map2nside(skymap) == nside

    def test_returns_int(self):
        assert isinstance(map2nside(np.ones(12)), int)


# ---------------------------------------------------------------------------
# angular_distance
# ---------------------------------------------------------------------------


class TestAngularDistance:
    def test_self_distance_is_zero(self):
        """Distance from a point to itself is 0."""
        ra, dec = 1.2, 0.5
        assert_allclose(angular_distance(ra, dec, ra, dec), 0.0, atol=1e-12)

    def test_antipodal_distance_is_pi(self):
        """Distance between antipodal points is π."""
        ra, dec = 0.0, np.pi / 2
        assert_allclose(angular_distance(ra, dec, ra + np.pi, -dec), np.pi, rtol=1e-12)

    def test_equatorial_90deg(self):
        """Two points separated by 90° of RA on the equator."""
        assert_allclose(
            angular_distance(0.0, 0.0, np.pi / 2, 0.0),
            np.pi / 2,
            rtol=1e-12,
        )

    def test_symmetry(self):
        """d(A, B) == d(B, A)."""
        ra1, dec1 = 0.3, 0.7
        ra2, dec2 = 1.1, -0.4
        d1 = angular_distance(ra1, dec1, ra2, dec2)
        d2 = angular_distance(ra2, dec2, ra1, dec1)
        assert_allclose(d1, d2, rtol=1e-12)

    def test_pole_to_pole(self):
        """North pole to south pole is π."""
        assert_allclose(
            angular_distance(0.0, np.pi / 2, 0.0, -np.pi / 2),
            np.pi,
            rtol=1e-12,
        )

    def test_nonnegative(self):
        """All angular distances are non-negative."""
        rng = np.random.default_rng(0)
        ras = rng.uniform(0, 2 * np.pi, 50)
        decs = rng.uniform(-np.pi / 2, np.pi / 2, 50)
        dists = angular_distance(ras[0], decs[0], ras, decs)
        assert np.all(dists >= 0)

    def test_output_in_range(self):
        """Angular distance is always in [0, π]."""
        rng = np.random.default_rng(1)
        ras = rng.uniform(0, 2 * np.pi, 100)
        decs = rng.uniform(-np.pi / 2, np.pi / 2, 100)
        dists = angular_distance(ras[0], decs[0], ras, decs)
        assert np.all(dists <= np.pi + 1e-12)


# ---------------------------------------------------------------------------
# meshgrid2d
# ---------------------------------------------------------------------------


class TestMeshgrid2d:
    def test_output_shape(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0])
        ga, gb = meshgrid2d(a, b)
        assert ga.shape == (len(b), len(a))
        assert gb.shape == (len(b), len(a))

    def test_a_values_constant_along_rows(self):
        """Each row of ga should contain the same a value."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        ga, _ = meshgrid2d(a, b)
        for row in ga:
            assert_allclose(row, a)

    def test_b_values_constant_along_columns(self):
        """Each column of gb should contain the same b value."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        _, gb = meshgrid2d(a, b)
        for col in gb.T:
            assert_allclose(col, b)

    def test_matches_numpy_meshgrid(self):
        """Result should equal np.meshgrid(a, b) in 'xy' indexing."""
        a = np.linspace(0, 1, 4)
        b = np.linspace(0, 2, 5)
        ga, gb = meshgrid2d(a, b)
        np_a, np_b = np.meshgrid(a, b)
        assert_allclose(ga, np_a)
        assert_allclose(gb, np_b)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        ga, gb = meshgrid2d(a, b)
        assert ga.dtype == np.float32
        assert gb.dtype == np.float32
