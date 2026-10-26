"""Tests for read options (issues #9 and #10)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.options import GeometryDimension, LoadStrategy, ReadOptions


def test_coerce_accepts_strings_and_enums() -> None:
    assert ReadOptions.coerce(strategy="footprint").strategy is LoadStrategy.FOOTPRINT
    assert (
        ReadOptions.coerce(strategy=LoadStrategy.SEMANTIC_SURFACES).strategy
        is LoadStrategy.SEMANTIC_SURFACES
    )
    assert (
        ReadOptions.coerce(geometry_dimension="xy").geometry_dimension
        is GeometryDimension.XY
    )


def test_coerce_normalizes_lod_to_string() -> None:
    assert ReadOptions.coerce(lod=2).lod == "2"
    assert ReadOptions.coerce(lod=None).lod is None


def test_coerce_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        ReadOptions.coerce(strategy="nonsense")
    with pytest.raises(ValueError, match="unknown geometry_dimension"):
        ReadOptions.coerce(geometry_dimension="3d")


def test_defaults_are_semantic_surfaces_and_xyz() -> None:
    options = ReadOptions()
    assert options.strategy is LoadStrategy.SEMANTIC_SURFACES
    assert options.geometry_dimension is GeometryDimension.XYZ
    assert options.child_inherits_parent_attributes is False
