"""Tests for CityJSON `compress` — quantized integer vertices + a `transform` (issue #59).

`to_cityjson(compress=digits)` snaps coordinates to a `digits`-decimal grid, writes them as compact
integer `vertices`, and adds a CityJSON `transform` {scale, translate} the reader multiplies back —
smaller output, lossy to the chosen precision, and near-coincident points merge.
"""

import json

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.cityjson import parse_cityjson


def _model(n: int = 1) -> CityModel:
    objects = {}
    for i in range(n):
        x, y, z = 123456.789 + i * 3.14159, 654321.123 + i * 2.71828, 10.0 + i * 0.333
        surface = [
            [(x, y, z), (x + 5.5, y + 1.1, z), (x + 5.5, y + 6.6, z + 2.2), (x, y, z)]
        ]
        objects[f"o{i}"] = CityObject(
            id=f"o{i}",
            type="Building",
            geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])],
        )
    return CityModel(objects=objects)


def test_default_is_lossless_with_no_transform() -> None:
    doc = _model().to_cityjson()
    assert "transform" not in doc
    assert doc["vertices"][0] == [123456.789, 654321.123, 10.0]  # exact floats


def test_compress_writes_integer_vertices_and_a_transform() -> None:
    doc = _model().to_cityjson(compress=3)
    assert "transform" in doc
    assert doc["transform"]["scale"] == [0.001, 0.001, 0.001]
    assert all(
        isinstance(component, int) for vertex in doc["vertices"] for component in vertex
    )
    assert doc["vertices"][0] == [0, 0, 0]  # the min corner is the transform origin


def test_compress_round_trips_to_the_chosen_precision() -> None:
    model = _model()
    back = parse_cityjson(model.to_cityjson(compress=3))
    original = model.objects["o0"].geometries[0].boundaries[0][0][0]
    recovered = back.objects["o0"].geometries[0].boundaries[0][0][0]
    assert all(
        abs(a - b) <= 0.001 for a, b in zip(original, recovered)
    )  # within the quant step


def test_compress_shrinks_a_realistic_model() -> None:
    model = _model(200)
    exact = len(json.dumps(model.to_cityjson()))
    compressed = len(json.dumps(model.to_cityjson(compress=3)))
    assert compressed < exact  # the per-vertex integer saving dominates at scale


def test_compress_merges_points_that_quantize_together() -> None:
    # Two coordinates closer than the quant step collapse to one pooled vertex.
    near = [
        [(0.0, 0.0, 0.0), (0.0004, 0.0, 0.0), (1.0, 1.0, 0.0)]
    ]  # first two round to the same cell
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[near])],
            )
        }
    )
    doc = model.to_cityjson(compress=3)
    assert len(doc["vertices"]) == 2  # the two near points merged (3 distinct -> 2)
