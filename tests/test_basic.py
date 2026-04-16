"""
Placeholder test module for kingmaker.

Add your tests here following pytest conventions.
"""

import numpy as np
import pytest


def test_import():
    """Test that the package can be imported."""
    from kingmaker.pdf import KingPDF, InterpolatedKingPDF
    from kingmaker.fitting import KingPSFFitter

    assert KingPDF is not None
    assert InterpolatedKingPDF is not None
    assert KingPSFFitter is not None


def test_basic_king_pdf():
    """Basic smoke test for KingPDF."""
    from kingmaker.pdf import KingPDF

    king = KingPDF(angular_cutoff=np.pi)

    # Test basic evaluation
    alpha = np.radians(1.0)
    beta = 2.0
    x = np.radians(0.5)

    pdf_val = king.pdf(x, alpha, beta)
    assert pdf_val > 0
    assert np.isfinite(pdf_val)

    # Test CDF
    cdf_val = king.cdf(x, alpha, beta)
    assert 0 <= cdf_val <= 1
    assert np.isfinite(cdf_val)


def test_interpolated_king_pdf():
    """Basic smoke test for InterpolatedKingPDF."""
    from kingmaker.pdf import InterpolatedKingPDF

    king = InterpolatedKingPDF(angular_cutoff=np.pi)

    # Test basic evaluation
    alpha = np.radians(1.0)
    beta = 2.0
    x = np.radians(0.5)

    pdf_val = king.pdf(x, alpha, beta)
    assert pdf_val > 0
    assert np.isfinite(pdf_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
