"""
Unit tests for InterpolatedKingPDF.

Focuses on what InterpolatedKingPDF adds over KingPDF: the precomputed
normalization grid, the interpolated norm() override, and out-of-bounds
error handling. Inherited PDF/CDF/sample behaviour is tested against the
base class to confirm the interpolation doesn't break correctness.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from kingmaker.pdf import KingPDF, InterpolatedKingPDF


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PARAM_CASES = [
    pytest.param(np.radians(0.5), 2.0, id="narrow-moderate"),
    pytest.param(np.radians(2.0), 2.0, id="wide-moderate"),
    pytest.param(np.radians(1.0), 5.0, id="moderate-heavy"),
    pytest.param(np.radians(1.0), 9.0, id="near-grid-edge"),
]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInterpolatedKingPDFInit:
    def test_default_grid_shapes(self):
        king = InterpolatedKingPDF()
        assert len(king.points_alpha) == 200
        assert len(king.points_beta) == 200

    def test_default_cutoff(self):
        king = InterpolatedKingPDF()
        assert king.angular_cutoff == np.pi

    def test_custom_cutoff_stored(self):
        king = InterpolatedKingPDF(angular_cutoff=np.pi / 2)
        assert king.angular_cutoff == np.pi / 2

    def test_custom_grid_stored(self):
        alphas = np.logspace(-3, 0, 50)
        betas = np.logspace(0, 1, 50)
        king = InterpolatedKingPDF(points_alpha=alphas, points_beta=betas)
        assert_allclose(king.points_alpha, alphas)
        assert_allclose(king.points_beta, betas)

    def test_log10_grids_match_points(self):
        king = InterpolatedKingPDF()
        assert_allclose(king.log10_points_alpha, np.log10(king.points_alpha))
        assert_allclose(king.log10_points_beta, np.log10(king.points_beta))

    def test_norm_grid_shape(self):
        alphas = np.logspace(-3, 0, 30)
        betas = np.logspace(0, 1, 40)
        king = InterpolatedKingPDF(points_alpha=alphas, points_beta=betas)
        assert king.log10_grid_norms.shape == (30, 40)

    def test_norm_grid_finite(self):
        king = InterpolatedKingPDF()
        assert np.all(np.isfinite(king.log10_grid_norms))

    def test_norm_grid_positive(self):
        """All precomputed normalizations must be positive (log10 of positive)."""
        king = InterpolatedKingPDF()
        assert np.all(10**king.log10_grid_norms > 0)


# ---------------------------------------------------------------------------
# Norm accuracy vs base class
# ---------------------------------------------------------------------------


class TestInterpolatedKingPDFNorm:
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_norm_close_to_base(self, alpha, beta):
        """Interpolated norm should agree with the exact norm to < 0.1%."""
        base = KingPDF()
        interp = InterpolatedKingPDF()
        assert_allclose(interp.norm(alpha, beta), base.norm(alpha, beta), rtol=1e-3)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_norm_positive(self, alpha, beta):
        king = InterpolatedKingPDF()
        assert king.norm(alpha, beta) > 0

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_norm_finite(self, alpha, beta):
        king = InterpolatedKingPDF()
        assert np.isfinite(king.norm(alpha, beta))

    def test_norm_array_input(self):
        king = InterpolatedKingPDF()
        alphas = np.radians([0.5, 1.0, 2.0])
        norms = king.norm(alphas, 2.0)
        assert norms.shape == alphas.shape
        assert np.all(norms > 0)

    def test_norm_array_matches_scalar(self):
        king = InterpolatedKingPDF()
        alphas = np.radians([0.5, 1.0, 2.0])
        scalar_norms = np.array([king.norm(a, 2.0) for a in alphas])
        assert_allclose(king.norm(alphas, 2.0), scalar_norms, rtol=1e-6)

    # Out-of-bounds errors
    def test_raises_alpha_too_small(self):
        king = InterpolatedKingPDF()
        with pytest.raises(ValueError, match="[Aa]lpha"):
            king.norm(king.points_alpha[0] * 0.5, 2.0)

    def test_raises_alpha_too_large(self):
        king = InterpolatedKingPDF()
        with pytest.raises(ValueError, match="[Aa]lpha"):
            king.norm(king.points_alpha[-1] * 2.0, 2.0)

    def test_raises_beta_too_small(self):
        king = InterpolatedKingPDF()
        with pytest.raises(ValueError, match="[Bb]eta"):
            king.norm(np.radians(1.0), king.points_beta[0] * 0.5)

    def test_raises_beta_too_large(self):
        king = InterpolatedKingPDF()
        with pytest.raises(ValueError, match="[Bb]eta"):
            king.norm(np.radians(1.0), king.points_beta[-1] * 2.0)


# ---------------------------------------------------------------------------
# PDF correctness (inherits KingPDF.pdf but uses interpolated norm)
# ---------------------------------------------------------------------------


class TestInterpolatedKingPDFPDF:
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_pdf_close_to_base(self, alpha, beta):
        """PDF values from interpolated norm should match the base class."""
        theta = np.linspace(0, np.radians(5), 100)
        base = KingPDF()
        interp = InterpolatedKingPDF()
        assert_allclose(interp.pdf(theta, alpha, beta), base.pdf(theta, alpha, beta), rtol=1e-3)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_normalizes_to_one(self, alpha, beta):
        king = InterpolatedKingPDF()
        theta = np.linspace(0, king.angular_cutoff, 5000)
        integral = 2 * np.pi * np.trapezoid(king.pdf(theta, alpha, beta) * np.sin(theta), theta)
        assert_allclose(integral, 1.0, rtol=1e-3)

    @pytest.mark.parametrize("cutoff", [np.pi / 2, np.pi])
    def test_zero_beyond_cutoff(self, cutoff):
        king = InterpolatedKingPDF(angular_cutoff=cutoff)
        beyond = np.array([cutoff + 0.01, cutoff + 0.2])
        beyond = beyond[beyond <= np.pi]
        vals = king.pdf(beyond, np.radians(1.0), 2.0)
        assert np.all(vals == 0)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_pdf_positive_inside_cutoff(self, alpha, beta):
        king = InterpolatedKingPDF()
        theta = np.linspace(0, np.radians(5), 50)
        assert np.all(king.pdf(theta, alpha, beta) >= 0)


# ---------------------------------------------------------------------------
# CDF correctness
# ---------------------------------------------------------------------------


class TestInterpolatedKingPDFCDF:
    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_cdf_close_to_base(self, alpha, beta):
        theta = np.linspace(0, np.radians(5), 50)
        base = KingPDF()
        interp = InterpolatedKingPDF()
        assert_allclose(interp.cdf(theta, alpha, beta), base.cdf(theta, alpha, beta), rtol=1e-3)

    @pytest.mark.parametrize("alpha, beta", PARAM_CASES)
    def test_cdf_at_cutoff_is_one(self, alpha, beta):
        king = InterpolatedKingPDF()
        assert_allclose(king.cdf(king.angular_cutoff, alpha, beta), 1.0, rtol=1e-3)


# ---------------------------------------------------------------------------
# Sampling (inherited, but confirm it still works with interpolated norm)
# ---------------------------------------------------------------------------


class TestInterpolatedKingPDFSample:
    def test_sample_length(self):
        king = InterpolatedKingPDF()
        assert len(king.sample(200, np.radians(1.0), 2.0)) == 200

    def test_samples_within_cutoff(self):
        cutoff = np.pi / 2
        king = InterpolatedKingPDF(angular_cutoff=cutoff)
        samples = king.sample(500, np.radians(1.0), 2.0)
        assert np.all(samples <= cutoff + 1e-9)

    def test_sample_reproducible(self):
        king = InterpolatedKingPDF()
        s1 = king.sample(100, np.radians(1.0), 2.0, rng=np.random.default_rng(7))
        s2 = king.sample(100, np.radians(1.0), 2.0, rng=np.random.default_rng(7))
        assert_allclose(s1, s2)
