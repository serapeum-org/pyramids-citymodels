"""Tests for the shared intermediate representation (issue #5)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import (
    CityObject,
    crs_from_identifier,
    iter_surfaces,
    normalize_lod,
)


def test_crs_from_identifier_handles_urn_url_and_plain() -> None:
    assert crs_from_identifier("urn:ogc:def:crs:EPSG::7415") == "EPSG:7415"
    assert (
        crs_from_identifier("https://www.opengis.net/def/crs/EPSG/0/25832")
        == "EPSG:25832"
    )
    assert crs_from_identifier("EPSG:4326") == "EPSG:4326"
    assert crs_from_identifier(None) is None
    assert crs_from_identifier("not a crs") is None


def test_normalize_lod_makes_numeric_lods_comparable() -> None:
    assert normalize_lod("2") == normalize_lod(2) == "2.0"
    assert normalize_lod("2.2") == "2.2"
    assert normalize_lod(None) is None


def test_iter_surfaces_multisurface_pairs_each_surface_with_its_value() -> None:
    # A MultiSurface: boundaries is a list of surfaces, values a flat list.
    surface_a = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    surface_b = [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
    pairs = list(iter_surfaces([surface_a, surface_b], [0, 1]))
    assert [value for _surface, value in pairs] == [0, 1]
    assert pairs[0][0] is surface_a


def test_iter_surfaces_solid_descends_shells() -> None:
    # A Solid: boundaries is a list of shells (each a list of surfaces); values nests one level.
    surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    solid = [[surface, surface]]  # one shell, two surfaces
    pairs = list(iter_surfaces(solid, [[7, 9]]))
    assert [value for _surface, value in pairs] == [7, 9]


def test_city_object_parent_is_first_parent_or_none() -> None:
    assert CityObject(id="a", type="Building").parent is None
    assert CityObject(id="b", type="BuildingPart", parents=["a"]).parent == "a"
