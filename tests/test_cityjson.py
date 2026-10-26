"""Tests for the CityJSON parser (issue #6)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import json
from pathlib import Path

from pyramids_citymodels.readers.cityjson import looks_like_cityjson, parse_cityjson

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"


def test_parse_reads_version_crs_and_objects() -> None:
    model = parse_cityjson(CITYJSON)
    assert model.source_format == "CityJSON"
    assert model.version == "2.0"
    assert model.crs == "EPSG:7415"
    assert set(model.objects) == {"bldg1", "bldg1_part"}


def test_parse_dereferences_and_transforms_coordinates() -> None:
    model = parse_cityjson(CITYJSON)
    building = model.objects["bldg1"]
    geometry = building.geometries[0]
    assert geometry.type == "MultiSurface"
    assert geometry.lod == "2"
    # First surface, first (exterior) ring, third vertex is quantized index 2 -> (1,1,0) * 0.001scale.
    ground_ring = geometry.boundaries[0][0]
    assert ground_ring[2] == (1.0, 1.0, 0.0)


def test_parse_keeps_semantic_surfaces_and_values() -> None:
    model = parse_cityjson(CITYJSON)
    geometry = model.objects["bldg1"].geometries[0]
    assert [surface.type for surface in geometry.semantics] == [
        "GroundSurface",
        "WallSurface",
        "RoofSurface",
    ]
    assert geometry.semantic_values == [0, 1, 2]
    assert geometry.semantic_for(2).type == "RoofSurface"


def test_parse_builds_parent_child_hierarchy() -> None:
    model = parse_cityjson(CITYJSON)
    assert model.objects["bldg1"].children == ["bldg1_part"]
    assert model.objects["bldg1_part"].parent == "bldg1"


def test_parse_accepts_a_preloaded_dict() -> None:
    document = json.loads(CITYJSON.read_text(encoding="utf-8"))
    model = parse_cityjson(document)
    assert "bldg1" in model.objects


def test_parse_accepts_a_json_content_string() -> None:
    text = CITYJSON.read_text(encoding="utf-8")
    model = parse_cityjson(text)
    assert "bldg1" in model.objects


def test_looks_like_cityjson_distinguishes_from_geojson() -> None:
    assert looks_like_cityjson(CITYJSON) is True
    geojson = '{"type": "FeatureCollection", "features": []}'
    assert looks_like_cityjson(geojson) is False
    assert looks_like_cityjson("not json at all") is False
    # A CityJSON content string (not a path) is also recognized.
    assert looks_like_cityjson(CITYJSON.read_text(encoding="utf-8")) is True
