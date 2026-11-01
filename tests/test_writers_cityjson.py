"""Tests for the CityJSON writer and the vertex-pool rebuild (issues #24, #25, #27)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import json
from pathlib import Path

from pyramids_citymodels.ir import (
    CityModel,
    CityObject,
    Geometry,
    SemanticSurface,
)
from pyramids_citymodels.readers.cityjson import parse_cityjson
from pyramids_citymodels.writers.cityjson import write_cityjson, write_cityjsonseq

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"


def test_round_trip_reproduces_the_model() -> None:
    """parse → write → parse reproduces the IR exactly (the #24 round-trip contract)."""
    model = parse_cityjson(CITYJSON)
    assert parse_cityjson(write_cityjson(model)) == model


def test_write_rebuilds_vertex_pool_and_indexes_boundaries() -> None:
    model = parse_cityjson(CITYJSON)
    document = write_cityjson(model)
    assert document["type"] == "CityJSON"
    # Every boundary leaf is an int index into the rebuilt vertex pool, not a coordinate.
    boundaries = document["CityObjects"]["bldg1"]["geometry"][0]["boundaries"]
    assert all(isinstance(index, int) for ring in boundaries for index in ring[0])
    vertices = document["vertices"]
    assert all(0 <= index < len(vertices) for ring in boundaries for index in ring[0])


def test_vertex_pool_deduplicates_identical_coordinates() -> None:
    """Two surfaces sharing a coordinate intern it once — the pool is deduplicated."""
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]],
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]],
                        ],
                    )
                ],
            )
        }
    )
    document = write_cityjson(model)
    # Six coordinates referenced, but (0,0,0) and (1,0,0) are shared → four unique vertices.
    assert len(document["vertices"]) == 4


def test_write_preserves_semantics_values_and_surface_metadata() -> None:
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                        semantics=[
                            SemanticSurface(
                                type="RoofSurface",
                                attributes={"slope": 45.0},
                                parent=None,
                                children=[1],
                            )
                        ],
                        semantic_values=[0],
                    )
                ],
            )
        }
    )
    semantics = write_cityjson(model)["CityObjects"]["b"]["geometry"][0]["semantics"]
    assert semantics["values"] == [0]
    assert semantics["surfaces"][0] == {
        "type": "RoofSurface",
        "slope": 45.0,
        "children": [1],
    }


def test_write_omits_semantics_when_absent() -> None:
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod=None,
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                    )
                ],
            )
        }
    )
    geometry = write_cityjson(model)["CityObjects"]["b"]["geometry"][0]
    assert "semantics" not in geometry
    assert "lod" not in geometry  # a None lod is omitted


def test_write_passes_appearance_and_extensions_through() -> None:
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                        appearance={"material": [{"value": 0}]},
                    )
                ],
            )
        },
        appearances={"materials": [{"name": "brick"}]},
        extensions={"Noise": {"url": "http://example.com/noise.json"}},
    )
    document = write_cityjson(model)
    assert document["appearance"] == {"materials": [{"name": "brick"}]}
    assert document["extensions"] == {"Noise": {"url": "http://example.com/noise.json"}}
    assert document["CityObjects"]["b"]["geometry"][0]["material"] == [{"value": 0}]


def test_reference_system_is_filled_from_crs_when_metadata_lacks_one() -> None:
    """A CityGML-sourced model (CRS set, empty metadata) still writes its CRS out."""
    model = CityModel(objects={}, crs="EPSG:25832")
    document = write_cityjson(model)
    assert document["metadata"]["referenceSystem"].endswith("/25832")


def test_write_to_path_writes_valid_json_and_returns_none(tmp_path: Path) -> None:
    model = parse_cityjson(CITYJSON)
    out = tmp_path / "out.city.json"
    assert write_cityjson(model, out) is None
    assert json.loads(out.read_text(encoding="utf-8"))["type"] == "CityJSON"


def test_empty_model_writes_an_empty_document() -> None:
    document = write_cityjson(CityModel())
    assert document["CityObjects"] == {}
    assert document["vertices"] == []


def test_cityjsonseq_has_a_header_then_one_feature_per_root() -> None:
    model = parse_cityjson(CITYJSON)
    lines = [json.loads(line) for line in write_cityjsonseq(model).strip().splitlines()]
    assert lines[0]["type"] == "CityJSON"
    # bldg1 is the only root (bldg1_part has a parent) → one feature line, bundling the part.
    assert [line["type"] for line in lines[1:]] == ["CityJSONFeature"]
    feature = lines[1]
    assert feature["id"] == "bldg1"
    assert set(feature["CityObjects"]) == {"bldg1", "bldg1_part"}


def test_cityjsonseq_writes_to_path(tmp_path: Path) -> None:
    model = parse_cityjson(CITYJSON)
    out = tmp_path / "out.city.jsonl"
    assert write_cityjsonseq(model, out) is None
    assert out.read_text(encoding="utf-8").count("\n") == 2
