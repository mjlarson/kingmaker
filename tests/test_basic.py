"""Smoke tests: verify all public classes can be imported."""

from kingmaker.pdf import KingPDF, InterpolatedKingPDF, TemplateSmearedKingPDF
from kingmaker.fitting import KingPSFFitter
from kingmaker.wrapper import KingSpatialLikelihood


def test_imports():
    assert KingPDF is not None
    assert InterpolatedKingPDF is not None
    assert TemplateSmearedKingPDF is not None
    assert KingPSFFitter is not None
    assert KingSpatialLikelihood is not None
