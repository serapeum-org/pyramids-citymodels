"""Tests for 3D Tiles 1.1 glTF tile content (issue #56).

`to_3dtiles(..., content="gltf")` writes tile content as a plain `.glb` instead of a `b3dm` wrapper:
each city object is a glTF feature via `EXT_mesh_features` (a `_FEATURE_ID_0` vertex attribute) and
its id / type / attributes live in an `EXT_structural_metadata` property table (attributes as one
JSON-encoded STRING property). The output round-trips through this package's own 3D-Tiles reader in
every tiling mode (single tile, explicit quadtree, implicit quadtree/octree), and — unlike the
columnar `b3dm` batch table — any JSON-representable attribute round-trips (types preserved, nested
values kept, a sparse key kept only on the objects that carry it). The feature-id attribute is FLOAT
(glTF forbids an integer vertex attribute), and a non-finite or non-JSON attribute value raises a
clear error rather than emitting invalid JSON.
"""

import json
import struct
import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import numpy as np
import pytest

from pyramids_citymodels.errors import UnsupportedGeometryError
from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import (
    _decode_gltf,
    _maybe_json_object,
    _parse_glb,
    _parse_gltf_tile,
    looks_like_3dtiles,
    parse_3dtiles,
)
from pyramids_citymodels.writers.threedtiles import _encode_glb


def _square(x: float, y: float) -> list[list[tuple[float, float, float]]]:
    """A closed 10-unit square surface at ``(x, y)`` on the z=0 plane."""
    ring = [(x, y, 0.0), (x + 10, y, 0.0), (x + 10, y + 10, 0.0), (x, y + 10, 0.0)]
    return [ring + [ring[0]]]


def _model(count: int = 2) -> CityModel:
    """``count`` buildings on a diagonal, each with a distinctive attribute set."""
    return CityModel(
        objects={
            f"o{i}": CityObject(
                id=f"o{i}",
                type="Building" if i % 2 == 0 else "Bridge",
                attributes={"height": 3.5 + i, "label": f"obj-{i}", "floors": i},
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[_square(i * 50.0, i * 50.0)],
                    )
                ],
            )
            for i in range(count)
        }
    )


def _glb_document(glb: bytes) -> dict:
    """Parse the JSON chunk out of a ``.glb`` for structural assertions."""
    magic, _version, length = struct.unpack_from("<III", glb, 0)
    assert magic == 0x46546C67  # "glTF"
    assert length == len(glb)  # declared length matches the file
    json_length, json_type = struct.unpack_from("<II", glb, 12)
    assert json_type == 0x4E4F534A  # "JSON"
    return json.loads(glb[20 : 20 + json_length])


def test_gltf_content_writes_a_glb_tile_and_a_1_1_tileset() -> None:
    out = Path(tempfile.mkdtemp())
    tileset = _model().to_3dtiles(out, content="gltf")
    assert tileset["root"]["content"]["uri"] == "content.glb"
    assert tileset["asset"]["version"] == "1.1"  # glTF content is a 1.1 feature
    assert tileset["asset"]["gltfUpAxis"] == "Z"
    assert not (out / "content.b3dm").exists()  # no b3dm wrapper
    assert (out / "content.glb").read_bytes()[:4] == b"glTF"


def test_glb_declares_the_two_metadata_extensions() -> None:
    out = Path(tempfile.mkdtemp())
    _model().to_3dtiles(out, content="gltf")
    document = _glb_document((out / "content.glb").read_bytes())
    assert set(document["extensionsUsed"]) == {
        "EXT_mesh_features",
        "EXT_structural_metadata",
    }
    primitive = document["meshes"][0]["primitives"][0]
    # Per-feature IDs replace _BATCHID with a _FEATURE_ID_0 attribute + an EXT_mesh_features table.
    assert "_FEATURE_ID_0" in primitive["attributes"]
    assert "_BATCHID" not in primitive["attributes"]
    feature_ids = primitive["extensions"]["EXT_mesh_features"]["featureIds"][0]
    assert feature_ids["featureCount"] == 2
    assert feature_ids["propertyTable"] == 0
    # Per-object attributes live in an EXT_structural_metadata property table.
    table = document["extensions"]["EXT_structural_metadata"]["propertyTables"][0]
    assert table["count"] == 2
    assert set(table["properties"]) == {"id", "type", "attributes"}


def test_gltf_content_round_trips_geometry_and_identity() -> None:
    out = Path(tempfile.mkdtemp())
    _model().to_3dtiles(out, content="gltf")
    back = parse_3dtiles(out / "tileset.json")
    assert sorted(back.objects) == ["o0", "o1"]
    assert back.objects["o0"].type == "Building"
    assert back.objects["o1"].type == "Bridge"
    # Each feature recovered an areal footprint.
    for city_object in back.objects.values():
        assert city_object.geometries
        assert city_object.geometries[0].boundaries


def test_gltf_attribute_round_trip_is_lossless_and_typed() -> None:
    # Unlike the columnar b3dm batch table, the glTF path preserves attribute TYPES and keeps id/type
    # out of the attributes dict (they are their own metadata properties).
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                attributes={
                    "height": 12.5,
                    "floors": 3,
                    "listed": True,
                    "name": "Hall",
                },
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            )
        }
    )
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf")
    recovered = parse_3dtiles(out / "tileset.json").objects["b"]
    assert recovered.attributes == {
        "height": 12.5,
        "floors": 3,
        "listed": True,
        "name": "Hall",
    }
    assert isinstance(recovered.attributes["floors"], int)
    assert isinstance(recovered.attributes["listed"], bool)
    assert "id" not in recovered.attributes
    assert "type" not in recovered.attributes


def test_gltf_keeps_a_sparse_attribute_only_where_present() -> None:
    # A key on one object but not another must not be null-injected into the other (the b3dm batch
    # table null-fills; the glTF JSON-per-feature encoding does not).
    model = CityModel(
        objects={
            "a": CityObject(
                id="a",
                type="Building",
                attributes={"only_a": 1},
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            ),
            "b": CityObject(
                id="b",
                type="Building",
                attributes={},
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(50, 50)])
                ],
            ),
        }
    )
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf")
    back = parse_3dtiles(out / "tileset.json")
    assert back.objects["a"].attributes == {"only_a": 1}
    assert back.objects["b"].attributes == {}  # no spurious only_a=None


@pytest.mark.parametrize(
    "mode",
    [
        {"max_objects_per_tile": 1},  # explicit XY quadtree, one object per leaf
        {"implicit_tiling": True, "max_objects_per_tile": 1},  # implicit quadtree
        {"implicit_tiling": True, "scheme": "octree", "max_objects_per_tile": 1},
    ],
)
def test_gltf_content_round_trips_in_every_tiling_mode(mode: dict) -> None:
    model = _model(4)
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf", **mode)
    # Every content file is a .glb, never a .b3dm.
    glbs = list(out.rglob("*.glb"))
    assert glbs  # at least one glTF content file written
    assert not list(out.rglob("*.b3dm"))  # and never a legacy b3dm
    back = parse_3dtiles(out / "tileset.json")
    assert sorted(back.objects) == ["o0", "o1", "o2", "o3"]
    assert back.objects["o2"].attributes == {
        "height": 5.5,
        "label": "obj-2",
        "floors": 2,
    }


def test_bare_glb_tile_is_recognised_and_dispatched_to_the_gltf_decoder() -> None:
    # A single .glb (no tileset) is recognised as 3D-Tiles content and routed to the glTF decoder.
    # Like a bare b3dm, its footprint is rotated by the default y-up->z-up convention (there is no
    # tileset to declare gltfUpAxis="Z"), so correct plan-view geometry needs tileset.json — here we
    # only assert the glTF path is taken and does not raise.
    out = Path(tempfile.mkdtemp())
    _model(1).to_3dtiles(out, content="gltf")
    glb = out / "content.glb"
    assert looks_like_3dtiles(glb)
    assert parse_3dtiles(glb).source_format == "3DTiles"


def test_empty_model_gltf_content_is_a_trivial_tileset() -> None:
    out = Path(tempfile.mkdtemp())
    tileset = CityModel().to_3dtiles(out, content="gltf")
    assert "content" not in tileset["root"]  # nothing to write
    assert not list(out.rglob("*.glb"))


def test_unknown_content_format_raises() -> None:
    model = _model(1)
    out = tempfile.mkdtemp()
    with pytest.raises(ValueError, match="content must be 'b3dm' or 'gltf'"):
        model.to_3dtiles(out, content="obj")


def test_gltf_content_round_trips_non_ascii_strings() -> None:
    # The EXT_structural_metadata STRING pool is UTF-8; multibyte characters must survive the
    # values/stringOffsets byte-offset math in id, type and the attributes JSON.
    model = CityModel(
        objects={
            "café": CityObject(
                id="café",
                type="Bâtiment",
                attributes={"nom": "Hôtel de Ville", "ville": "Zürich"},
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            )
        }
    )
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf")
    recovered = parse_3dtiles(out / "tileset.json").objects["café"]
    assert recovered.type == "Bâtiment"
    assert recovered.attributes == {"nom": "Hôtel de Ville", "ville": "Zürich"}


def test_gltf_content_round_trips_nested_attribute_values() -> None:
    # Attributes ride as one JSON-encoded property, so nested dicts/lists round-trip losslessly —
    # something the flat, columnar b3dm batch table cannot represent.
    attrs = {
        "stories": [1, 2, 3],
        "roof": {"type": "gable", "pitch": 35.0},
        "tags": ["heritage", "listed"],
    }
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                attributes=attrs,
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            )
        }
    )
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf")
    assert parse_3dtiles(out / "tileset.json").objects["b"].attributes == attrs


def test_feature_id_attribute_uses_a_gltf_legal_component_type() -> None:
    # glTF 2.0 forbids an integer componentType for a vertex attribute; _FEATURE_ID_0 must be FLOAT
    # (5126), not UNSIGNED_INT (5125). Asserted at the glTF level, not via the package's own (lenient)
    # reader — a conformance check the round-trip tests cannot make.
    out = Path(tempfile.mkdtemp())
    _model(2).to_3dtiles(out, content="gltf")
    document = _glb_document((out / "content.glb").read_bytes())
    primitive = document["meshes"][0]["primitives"][0]
    feature_id_accessor = primitive["attributes"]["_FEATURE_ID_0"]
    assert document["accessors"][feature_id_accessor]["componentType"] == 5126  # FLOAT
    assert (
        document["accessors"][primitive["indices"]]["componentType"] == 5125
    )  # UINT indices


def test_non_finite_attribute_value_raises_a_clear_error() -> None:
    # NaN / Infinity would serialize to non-standard JSON tokens a strict client cannot parse; the
    # writer rejects them with a ValueError naming the object instead.
    model = CityModel(
        objects={
            "tower": CityObject(
                id="tower",
                type="Building",
                attributes={"height": float("inf")},
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            )
        }
    )
    out = Path(tempfile.mkdtemp())
    with pytest.raises(ValueError, match="object 'tower'"):
        model.to_3dtiles(out, content="gltf")
    assert not list(
        out.rglob("*.glb")
    )  # nothing written on the error (assert on the real target)


@pytest.mark.parametrize(
    "mode",
    [
        {},  # single tile
        {"max_objects_per_tile": 1},  # explicit quadtree (writes content per leaf)
        {
            "implicit_tiling": True,
            "max_objects_per_tile": 1,
        },  # implicit (writes content per cell)
    ],
)
def test_a_rejected_attribute_leaves_nothing_written_in_any_mode(mode: dict) -> None:
    # The multi-tile writers emit content incrementally; a later object's bad attribute must still
    # leave the output empty (validated up front), not orphan .glb tiles with no tileset.json.
    model = CityModel(
        objects={
            "ok": CityObject(
                id="ok",
                type="Building",
                attributes={"h": 1},
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            ),
            "bad": CityObject(
                id="bad",
                type="Building",
                attributes={"x": float("nan")},
                geometries=[
                    Geometry(
                        type="MultiSurface", lod="2", boundaries=[_square(100, 100)]
                    )
                ],
            ),
        }
    )
    out = Path(tempfile.mkdtemp())
    with pytest.raises(ValueError, match="object 'bad'"):
        model.to_3dtiles(out, content="gltf", **mode)
    assert list(out.rglob("*")) == []  # fail-fast: no content, no tileset.json


def test_structural_metadata_schema_has_a_valid_id() -> None:
    # EXT_structural_metadata requires a schema `id` matching ^[a-zA-Z_]\w*$; omitting it makes the
    # schema invalid for a conformant validator.
    import re

    out = Path(tempfile.mkdtemp())
    _model(1).to_3dtiles(out, content="gltf")
    document = _glb_document((out / "content.glb").read_bytes())
    schema = document["extensions"]["EXT_structural_metadata"]["schema"]
    assert re.fullmatch(r"[a-zA-Z_]\w*", schema["id"]), schema.get("id")


def test_too_many_features_in_one_tile_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # A FLOAT feature id is exact only up to 2**24; a single tile beyond that is rejected rather than
    # silently aliasing ids. Shrink the ceiling to exercise the guard without building 16M objects.
    from pyramids_citymodels.writers import threedtiles as writer

    monkeypatch.setattr(writer, "_MAX_FEATURES_PER_TILE", 1)
    model = _model(2)
    out = tempfile.mkdtemp()
    with pytest.raises(ValueError, match="features in one tile"):
        model.to_3dtiles(out, content="gltf")


def test_non_json_attribute_value_raises_a_clear_error() -> None:
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                attributes={"tags": {"a", "b"}},  # a set is not JSON-serializable
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            )
        }
    )
    out = tempfile.mkdtemp()
    with pytest.raises(ValueError, match="cannot encode the attributes of object 'b'"):
        model.to_3dtiles(out, content="gltf")


def test_json_coercions_are_documented_not_silent_data_loss() -> None:
    # JSON has no tuple or non-string-key dict, so those coerce (tuple -> list, int key -> str). This
    # pins the real, documented behavior rather than pretending it is byte-lossless.
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                attributes={"span": (1, 2), "levels": {1: "ground", 2: "first"}},
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])
                ],
            )
        }
    )
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf")
    recovered = parse_3dtiles(out / "tileset.json").objects["b"].attributes
    assert recovered == {"span": [1, 2], "levels": {"1": "ground", "2": "first"}}


def test_gltf_feature_spans_multiple_surfaces_of_one_object() -> None:
    # One object with several surfaces is a single feature — all its vertices share one
    # _FEATURE_ID_0 — so it round-trips to one object (not one per surface).
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                attributes={"h": 9},
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[_square(0, 0), _square(5, 0), _square(0, 5)],
                    )
                ],
            )
        }
    )
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf")
    back = parse_3dtiles(out / "tileset.json")
    assert list(back.objects) == ["b"]
    assert back.objects["b"].attributes == {"h": 9}


# --- robustness / interop with other tools' glTF content ------------------------------------------


def _glb_bytes(model: CityModel) -> bytes:
    """The single-tile glTF content bytes a model writes."""
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, content="gltf")
    return (out / "content.glb").read_bytes()


def _rebuild_glb(glb: bytes, mutate) -> bytes:
    """Re-encode a ``.glb`` after ``mutate`` edits its glTF JSON (the BIN buffer is reused as-is)."""
    document, binary = _parse_glb(glb)
    mutate(document)
    return _encode_glb(document, binary)


def test_maybe_json_object_distinguishes_objects_from_other_values() -> None:
    assert _maybe_json_object('{"a": 1}') == {"a": 1}
    assert _maybe_json_object("") == {}  # empty string -> empty dict
    assert _maybe_json_object("not json") is None  # plain string stays a string
    assert _maybe_json_object("[1, 2]") is None  # a JSON array is not an object


def test_malformed_glb_tile_is_wrapped_as_unsupported_geometry() -> None:
    # A truncated / corrupt glb must raise UnsupportedGeometryError (one catchable type), so a caller
    # walking a tileset can skip a bad tile — mirroring the b3dm path.
    truncated = _glb_bytes(_model(1))[:40]
    with pytest.raises(
        UnsupportedGeometryError, match="glTF tile payload is malformed"
    ):
        _parse_gltf_tile(truncated, np.eye(4), up_axis="Z")


def test_gltf_content_without_our_metadata_still_yields_features() -> None:
    # A glTF tile from another tool may carry _FEATURE_ID_0 but no EXT_structural_metadata; features
    # are still recovered, with generated ids (nothing to read id/type/attributes from).
    def drop_metadata(document: dict) -> None:
        document.get("extensions", {}).pop("EXT_structural_metadata", None)

    glb = _rebuild_glb(_glb_bytes(_model(1)), drop_metadata)
    objects = _decode_gltf(glb, np.eye(4), up_axis="Z", source_id="t")
    assert len(objects) == 1
    assert objects[0].id == "t/feature-0"  # generated, since no id property
    assert objects[0].attributes == {}


def test_compressed_gltf_content_is_rejected() -> None:
    # A Draco/meshopt-compressed glTF tile this reader cannot decode raises UnsupportedGeometryError
    # (the message points at the compressed-mesh follow-up), not a bare decode error.
    def require_draco(document: dict) -> None:
        document["extensionsRequired"] = ["KHR_draco_mesh_compression"]

    glb = _rebuild_glb(_glb_bytes(_model(1)), require_draco)
    with pytest.raises(UnsupportedGeometryError, match="mesh-compression"):
        _parse_gltf_tile(glb, np.eye(4), up_axis="Z")


def test_gltf_tile_with_no_geometry_yields_no_features() -> None:
    def drop_meshes(document: dict) -> None:
        document["meshes"] = []

    glb = _rebuild_glb(_glb_bytes(_model(1)), drop_meshes)
    assert _decode_gltf(glb, np.eye(4), up_axis="Z", source_id="t") == []


def test_reader_skips_non_string_and_reads_extra_string_metadata_properties() -> None:
    # A non-string property (no stringOffsets) is skipped, not mis-decoded; an extra STRING property
    # beyond id/type/attributes becomes a plain attribute of its own name.
    def tweak(document: dict) -> None:
        table = document["extensions"]["EXT_structural_metadata"]["propertyTables"][0]
        table["properties"]["elevation"] = {
            "values": 0
        }  # numeric-ish, no stringOffsets -> skipped
        table["properties"]["owner"] = dict(
            table["properties"]["id"]
        )  # reuse id's string bytes

    glb = _rebuild_glb(_glb_bytes(_model(1)), tweak)
    recovered = _decode_gltf(glb, np.eye(4), up_axis="Z", source_id="t")[0]
    assert "elevation" not in recovered.attributes  # the numeric property was skipped
    assert (
        recovered.attributes["owner"] == "o0"
    )  # the extra string property came through
