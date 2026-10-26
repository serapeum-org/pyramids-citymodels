"""Tests for the CityGML LOD2 parser (issue #7)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

from pyramids_citymodels.readers.citygml import looks_like_citygml, parse_citygml

DATA = Path(__file__).parent / "data"
CITYGML = DATA / "minimal_building.gml"


def test_parse_reads_crs_and_building() -> None:
    model = parse_citygml(CITYGML)
    assert model.source_format == "CityGML"
    assert model.crs == "EPSG:25832"
    assert "B1" in model.objects
    assert model.objects["B1"].type == "Building"


def test_parse_reconstructs_boundary_surface_semantics() -> None:
    model = parse_citygml(CITYGML)
    geometry = model.objects["B1"].geometries[0]
    assert geometry.type == "MultiSurface"
    assert geometry.lod == "2"
    # Document order of boundedBy: Ground, Roof, Wall.
    assert [surface.type for surface in geometry.semantics] == [
        "GroundSurface",
        "RoofSurface",
        "WallSurface",
    ]


def test_parse_reads_poslist_coordinates_as_3d() -> None:
    model = parse_citygml(CITYGML)
    geometry = model.objects["B1"].geometries[0]
    ground_exterior = geometry.boundaries[0][0]
    assert ground_exterior[0] == (0.0, 0.0, 0.0)
    assert ground_exterior[2] == (1.0, 1.0, 0.0)
    assert len(ground_exterior[0]) == 3


def test_parse_lifts_building_attributes() -> None:
    model = parse_citygml(CITYGML)
    attributes = model.objects["B1"].attributes
    # measuredHeight is coerced to a number so it matches the CityJSON reader's JSON types (SF5).
    assert attributes["measuredHeight"] == 10.0
    assert isinstance(attributes["measuredHeight"], float)
    # Codelist values (function) and generic string attributes stay as strings.
    assert attributes["function"] == "1000"
    assert attributes["owner"] == "City"


def test_looks_like_citygml_accepts_citygml_and_declines_plain_xml() -> None:
    assert looks_like_citygml(CITYGML) is True
    assert looks_like_citygml("<root><child/></root>") is False
    assert looks_like_citygml("not xml") is False
