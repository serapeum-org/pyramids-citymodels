"""Tests for the JSON-FG writer (issue #27) — a thin, one-way GIS-exchange projection."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.cityjson import parse_cityjson
from pyramids_citymodels.writers.jsonfg import write_jsonfg

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"


def test_each_object_becomes_one_feature_with_3d_place_geometry() -> None:
    document = write_jsonfg(parse_cityjson(CITYJSON))
    assert document["type"] == "FeatureCollection"
    assert document["conformsTo"][0].startswith("http://www.opengis.net/spec/json-fg")
    feature_types = {feature["featureType"] for feature in document["features"]}
    assert feature_types == {"Building", "BuildingPart"}
    building = next(f for f in document["features"] if f["id"] == "bldg1")
    assert (
        building["geometry"] is None
    )  # GeoJSON geometry stays null; 3D lives in `place`
    assert building["place"]["type"] == "MultiPolygon"
    assert len(building["place"]["coordinates"]) == 3  # three surfaces


def test_coord_ref_sys_comes_from_the_crs() -> None:
    document = write_jsonfg(parse_cityjson(CITYJSON))
    assert document["coordRefSys"].endswith("/7415")


def test_no_crs_omits_coord_ref_sys() -> None:
    document = write_jsonfg(
        CityModel(objects={"b": CityObject(id="b", type="Building")})
    )
    assert "coordRefSys" not in document


def test_rings_are_closed_in_output() -> None:
    document = write_jsonfg(parse_cityjson(CITYJSON))
    ring = document["features"][0]["place"]["coordinates"][0][0]
    assert ring[0] == ring[-1]  # first point repeated to close the ring


def test_properties_carry_attributes_and_parent_id() -> None:
    document = write_jsonfg(parse_cityjson(CITYJSON))
    building = next(f for f in document["features"] if f["id"] == "bldg1")
    part = next(f for f in document["features"] if f["id"] == "bldg1_part")
    assert building["properties"]["measuredHeight"] == 10.0
    assert part["properties"]["parent_id"] == "bldg1"


def test_object_without_geometry_has_null_place() -> None:
    model = CityModel(objects={"b": CityObject(id="b", type="Building", geometries=[])})
    assert write_jsonfg(model)["features"][0]["place"] is None


def test_degenerate_surface_is_dropped() -> None:
    """A ring with fewer than three points yields no polygon (and no place)."""
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]],
                    )
                ],
            )
        }
    )
    assert write_jsonfg(model)["features"][0]["place"] is None


def test_write_to_path_returns_none(tmp_path: Path) -> None:
    import json

    out = tmp_path / "out.fg.json"
    assert write_jsonfg(parse_cityjson(CITYJSON), out) is None
    assert json.loads(out.read_text(encoding="utf-8"))["type"] == "FeatureCollection"
