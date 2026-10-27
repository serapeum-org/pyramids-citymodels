"""Branch-level tests for the 3D Tiles reader internals (issue #16).

These exercise the decode helpers and edge branches directly — node/quaternion transforms, the
convex-hull footprint, accessor striding, RTC_CENTER, missing-POSITION primitives, nested tilesets
and the malformed-source guards — with small in-memory glTF/b3dm payloads, so the suite needs no
large committed binaries beyond the one happy-path fixture.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import json
import struct

import numpy as np
import pytest

from pyramids_citymodels.errors import UnsupportedGeometryError
from pyramids_citymodels.readers import threedtiles as tt


def _pack_glb(gltf: dict, binary: bytes) -> bytes:
    """Wrap a glTF JSON document and its binary buffer into a glb container."""
    json_chunk = json.dumps(gltf).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_chunk) + (8 + len(binary) if binary else 0)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    if binary:
        out += struct.pack("<II", len(binary), 0x004E4942) + binary
    return out


def _mesh_glb(
    positions: list[tuple[float, float, float]],
    batch_ids: list[int] | None = None,
    node: dict | None = None,
) -> bytes:
    """A glb with one POSITION (and optional _BATCHID) mesh, optional node transform."""
    position_bytes = b"".join(struct.pack("<3f", *point) for point in positions)
    binary = position_bytes
    attributes = {"POSITION": 0}
    buffer_views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)}]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": len(positions),
            "type": "VEC3",
        }
    ]
    if batch_ids is not None:
        batch_bytes = b"".join(struct.pack("<H", value) for value in batch_ids)
        attributes["_BATCHID"] = 1
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": len(position_bytes),
                "byteLength": len(batch_bytes),
            }
        )
        accessors.append(
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(batch_ids),
                "type": "SCALAR",
            }
        )
        binary = position_bytes + batch_bytes
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{**(node or {}), "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "mode": 4}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    return _pack_glb(gltf, binary)


def _b3dm(
    glb: bytes, feature_table: dict | None = None, batch_table: dict | None = None
) -> bytes:
    """Wrap a glb in a b3dm with the given feature and batch tables."""
    ft = json.dumps(feature_table or {}).encode("utf-8")
    ft += b" " * ((8 - len(ft) % 8) % 8)
    bt = json.dumps(batch_table or {}).encode("utf-8")
    bt += b" " * ((8 - len(bt) % 8) % 8)
    body = ft + bt + glb
    header = struct.pack(
        "<4sIIIIII", b"b3dm", 1, 28 + len(body), len(ft), 0, len(bt), 0
    )
    return header + body


_SQUARE = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]


class TestNodeMatrix:
    """``_node_matrix`` builds a transform from an explicit matrix or TRS components."""

    def test_explicit_matrix_is_transposed_from_column_major(self) -> None:
        node = {"matrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1]}
        matrix = tt._node_matrix(node)
        assert matrix[0, 3] == 5.0 and matrix[1, 3] == 6.0 and matrix[2, 3] == 7.0

    def test_translation_only(self) -> None:
        matrix = tt._node_matrix({"translation": [1.0, 2.0, 3.0]})
        assert list(matrix[:3, 3]) == [1.0, 2.0, 3.0]

    def test_scale_only(self) -> None:
        matrix = tt._node_matrix({"scale": [2.0, 3.0, 4.0]})
        assert matrix[0, 0] == 2.0 and matrix[1, 1] == 3.0 and matrix[2, 2] == 4.0

    def test_rotation_identity_quaternion_is_identity(self) -> None:
        matrix = tt._node_matrix({"rotation": [0.0, 0.0, 0.0, 1.0]})
        assert np.allclose(matrix, np.eye(4))

    def test_combined_trs(self) -> None:
        matrix = tt._node_matrix(
            {"translation": [1, 0, 0], "rotation": [0, 0, 0, 1], "scale": [2, 2, 2]}
        )
        assert matrix[0, 0] == 2.0 and matrix[0, 3] == 1.0


class TestQuaternionMatrix:
    """``_quaternion_matrix`` turns a glTF quaternion into a rotation matrix."""

    def test_identity(self) -> None:
        assert np.allclose(tt._quaternion_matrix([0, 0, 0, 1]), np.eye(4))

    def test_180_degrees_about_z_flips_x_and_y(self) -> None:
        matrix = tt._quaternion_matrix([0, 0, 1, 0])
        point = matrix[:3, :3] @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(point, [-1.0, 0.0, 0.0])


class TestConvexHullRing:
    """``_convex_hull_ring`` returns a closed ring, or None for a degenerate footprint."""

    def test_square_returns_closed_ring(self) -> None:
        ring = tt._convex_hull_ring(
            np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        )
        assert ring is not None and ring[0] == ring[-1] and len(ring) == 5

    def test_fewer_than_three_unique_points_is_none(self) -> None:
        assert tt._convex_hull_ring(np.array([[0, 0], [0, 0]], dtype=float)) is None

    def test_collinear_points_are_none(self) -> None:
        assert (
            tt._convex_hull_ring(np.array([[0, 0], [1, 1], [2, 2]], dtype=float))
            is None
        )


class TestAccessorArray:
    """``_accessor_array`` honours count and an interleaving byteStride."""

    def test_empty_count_returns_empty(self) -> None:
        gltf = {
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 0, "type": "VEC3"}
            ],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 0}],
        }
        array = tt._accessor_array(gltf, b"", 0)
        assert array.shape == (0, 3)

    def test_interleaved_byte_stride_is_respected(self) -> None:
        packed = struct.pack("<3f", 1, 2, 3) + b"\x00\x00\x00\x00"
        packed += struct.pack("<3f", 4, 5, 6) + b"\x00\x00\x00\x00"
        gltf = {
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 2, "type": "VEC3"}
            ],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteStride": 16}],
        }
        array = tt._accessor_array(gltf, packed, 0)
        assert np.allclose(array, [[1, 2, 3], [4, 5, 6]])


class TestMeshVertices:
    """``_mesh_vertices`` collects positions, skips POSITION-less primitives, flags _BATCHID."""

    def test_no_batchid_returns_none_batch(self) -> None:
        gltf, binary = tt._parse_glb(_mesh_glb(_SQUARE))
        points, batch = tt._mesh_vertices(gltf, binary)
        assert len(points) == 4 and batch is None

    def test_primitive_without_position_is_skipped(self) -> None:
        gltf, binary = tt._parse_glb(_mesh_glb(_SQUARE))
        gltf["meshes"][0]["primitives"].append({"attributes": {}})
        points, _ = tt._mesh_vertices(gltf, binary)
        assert len(points) == 4

    def test_no_meshes_is_empty(self) -> None:
        gltf, binary = tt._parse_glb(_mesh_glb(_SQUARE))
        gltf["meshes"] = []
        points, batch = tt._mesh_vertices(gltf, binary)
        assert len(points) == 0 and batch is None

    def test_node_matrix_is_applied_to_vertices(self) -> None:
        glb = _mesh_glb(_SQUARE, node={"translation": [10.0, 0.0, 0.0]})
        gltf, binary = tt._parse_glb(glb)
        points, _ = tt._mesh_vertices(gltf, binary)
        assert points[:, 0].min() == 10.0


class TestB3dmEdges:
    """Edge cases inside ``_parse_b3dm_tile``: RTC_CENTER, no vertices, degenerate feature."""

    def test_rtc_center_shifts_the_footprint(self) -> None:
        tile = _b3dm(
            _mesh_glb(_SQUARE), feature_table={"RTC_CENTER": [100.0, 200.0, 0.0]}
        )
        objects = tt._parse_b3dm_tile(tile, np.eye(4), up_axis="Z")
        ring = objects[0].geometries[0].boundaries[0][0]
        assert min(point[0] for point in ring) == 100.0

    def test_tile_with_no_vertices_yields_no_objects(self) -> None:
        empty = _mesh_glb([])
        assert tt._parse_b3dm_tile(_b3dm(empty), np.eye(4)) == []

    def test_degenerate_feature_is_skipped(self) -> None:
        collinear = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
        assert tt._parse_b3dm_tile(_b3dm(_mesh_glb(collinear)), np.eye(4)) == []

    def test_binary_backed_batch_property_is_skipped(self) -> None:
        tile = _b3dm(
            _mesh_glb(_SQUARE, batch_ids=[0, 0, 0, 0]),
            feature_table={"BATCH_LENGTH": 1},
            batch_table={
                "height": {"byteOffset": 0, "componentType": "FLOAT", "type": "SCALAR"}
            },
        )
        objects = tt._parse_b3dm_tile(tile, np.eye(4), up_axis="Z")
        assert objects[0].attributes == {}

    def test_wrong_magic_raises(self) -> None:
        with pytest.raises(UnsupportedGeometryError, match="not a b3dm"):
            tt._parse_b3dm_tile(b"xxxx" + b"\x00" * 24, np.eye(4))


class TestMalformedPayload:
    """A corrupt payload raises the documented ``UnsupportedGeometryError``, not a raw error (SF1)."""

    def test_truncated_buffer_raises_unsupported_not_valueerror(self) -> None:
        """A POSITION accessor claiming more vertices than the BIN chunk holds is caught and wrapped."""
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "buffers": [{"byteLength": 4}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"}
            ],
        }
        tile = _b3dm(_pack_glb(gltf, struct.pack("<f", 1.0)))
        with pytest.raises(UnsupportedGeometryError, match="malformed"):
            tt._parse_b3dm_tile(tile, np.eye(4), up_axis="Z")

    def test_bad_accessor_index_raises_unsupported(self) -> None:
        """A primitive pointing at a non-existent accessor is wrapped, not an IndexError."""
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 9}}]}],
            "buffers": [{"byteLength": 4}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 4}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 0, "type": "VEC3"}
            ],
        }
        tile = _b3dm(_pack_glb(gltf, struct.pack("<f", 1.0)))
        with pytest.raises(UnsupportedGeometryError):
            tt._parse_b3dm_tile(tile, np.eye(4), up_axis="Z")


class TestBatchLength:
    """A declared ``BATCH_LENGTH`` below the real maximum id must not drop vertices (SF3)."""

    def test_under_declared_batch_length_keeps_all_features(self) -> None:
        positions = _SQUARE + _square_at(100.0)
        glb = _mesh_glb(positions, batch_ids=[0, 0, 0, 0, 1, 1, 1, 1])
        tile = _b3dm(glb, feature_table={"BATCH_LENGTH": 1})
        objects = tt._parse_b3dm_tile(tile, np.eye(4), up_axis="Z")
        assert len(objects) == 2, (
            "batch id 1 must not be dropped by a too-small BATCH_LENGTH"
        )


class TestTilesetWalk:
    """Nested tilesets, tile transforms, and the malformed-source guards."""

    def test_nested_tileset_is_recursed(self, tmp_path) -> None:
        (tmp_path / "inner.b3dm").write_bytes(_b3dm(_mesh_glb(_SQUARE)))
        (tmp_path / "inner.json").write_text(
            '{"asset": {"version": "1.0", "gltfUpAxis": "Z"}, "geometricError": 0, '
            '"root": {"content": {"uri": "inner.b3dm"}}}',
            encoding="utf-8",
        )
        outer = tmp_path / "tileset.json"
        outer.write_text(
            '{"asset": {"version": "1.0"}, "geometricError": 0, '
            '"root": {"content": {"uri": "inner.json"}}}',
            encoding="utf-8",
        )
        model = tt.parse_3dtiles(outer)
        assert len(model.objects) == 1

    def test_tile_transform_composes(self, tmp_path) -> None:
        (tmp_path / "t.b3dm").write_bytes(_b3dm(_mesh_glb(_SQUARE)))
        shift = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 50, 0, 0, 1]
        tileset = tmp_path / "tileset.json"
        tileset.write_text(
            json.dumps(
                {
                    "asset": {"version": "1.0", "gltfUpAxis": "Z"},
                    "geometricError": 0,
                    "root": {"transform": shift, "content": {"uri": "t.b3dm"}},
                }
            ),
            encoding="utf-8",
        )
        model = tt.parse_3dtiles(tileset)
        ring = next(iter(model.objects.values())).geometries[0].boundaries[0][0]
        assert min(point[0] for point in ring) == 50.0

    def test_bad_json_string_is_not_a_tileset(self) -> None:
        document, base = tt._load_tileset("{not valid json")
        assert document is None and base is None

    def test_unreadable_path_is_not_a_tileset(self, tmp_path) -> None:
        document, base = tt._load_tileset(tmp_path / "missing.json")
        assert document is None and base is None

    def test_plain_json_is_not_a_tileset(self) -> None:
        assert tt.looks_like_3dtiles('{"hello": "world"}') is False

    def test_root_not_a_dict_yields_empty_model(self) -> None:
        model = tt.parse_3dtiles(
            '{"asset": {"version": "1.0"}, "geometricError": 0, "root": "nope"}'
        )
        assert model.objects == {}

    def test_child_tiles_are_walked(self, tmp_path) -> None:
        (tmp_path / "c.b3dm").write_bytes(_b3dm(_mesh_glb(_SQUARE)))
        tileset = tmp_path / "tileset.json"
        tileset.write_text(
            json.dumps(
                {
                    "asset": {"version": "1.0", "gltfUpAxis": "Z"},
                    "geometricError": 0,
                    "root": {"children": [{"content": {"uri": "c.b3dm"}}]},
                }
            ),
            encoding="utf-8",
        )
        assert len(tt.parse_3dtiles(tileset).objects) == 1

    def test_content_json_that_is_not_a_tileset_is_ignored(self, tmp_path) -> None:
        (tmp_path / "plain.json").write_text('{"hello": "world"}', encoding="utf-8")
        tileset = tmp_path / "tileset.json"
        tileset.write_text(
            '{"asset": {"version": "1.0"}, "geometricError": 0, '
            '"root": {"content": {"uri": "plain.json"}}}',
            encoding="utf-8",
        )
        assert tt.parse_3dtiles(tileset).objects == {}

    def test_tiny_content_file_is_ignored(self, tmp_path) -> None:
        (tmp_path / "stub.cmpt").write_bytes(b"ab")
        tileset = tmp_path / "tileset.json"
        tileset.write_text(
            '{"asset": {"version": "1.0"}, "geometricError": 0, '
            '"root": {"content": {"uri": "stub.cmpt"}}}',
            encoding="utf-8",
        )
        assert tt.parse_3dtiles(tileset).objects == {}

    def test_read_bytes_missing_path_is_none(self, tmp_path) -> None:
        assert tt._read_bytes(tmp_path / "nope.b3dm") is None


class TestSceneNodes:
    """Scene-node walking branches in ``_mesh_world_transforms``."""

    def test_parent_child_node_transform_composes(self) -> None:
        glb = _mesh_glb(_SQUARE)
        gltf, binary = tt._parse_glb(glb)
        gltf["nodes"] = [
            {"translation": [10.0, 0.0, 0.0], "children": [1]},
            {"mesh": 0},
        ]
        gltf["scenes"] = [{"nodes": [0]}]
        points, _ = tt._mesh_vertices(gltf, binary)
        assert points[:, 0].min() == 10.0

    def test_out_of_range_scene_root_is_skipped(self) -> None:
        """An out-of-range scene root records no transform; the mesh still reads at identity."""
        glb = _mesh_glb(_SQUARE)
        gltf, binary = tt._parse_glb(glb)
        gltf["scenes"] = [{"nodes": [99]}]
        points, _ = tt._mesh_vertices(gltf, binary)
        assert len(points) == 4 and points[:, 0].min() == 0.0


# A square footprint authored y-up: it lies in the X-Z plane (y=0 ground) and is extruded to a
# y=10 roof. Its plan is the X-Z square; its naive XY projection would be a 2x10 side wall.
_YUP_FOOTPRINT = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.0, 0.0, -2.0),
    (0.0, 0.0, -2.0),
    (0.0, 10.0, 0.0),
    (2.0, 10.0, 0.0),
    (2.0, 10.0, -2.0),
    (0.0, 10.0, -2.0),
]


class TestUpAxis:
    """``gltfUpAxis`` turns a y-up mesh into a plan footprint, not a side elevation (MF2)."""

    def _extent(self, objects: list) -> tuple[float, float, float, float]:
        ring = objects[0].geometries[0].boundaries[0][0]
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return (min(xs), min(ys), max(xs), max(ys))

    def test_default_y_up_yields_the_plan_not_a_side_wall(self) -> None:
        """A y-up prism (2x2 footprint, 10 tall) must come back as a 2x2 plan, not a 2x10 wall."""
        tile = _b3dm(_mesh_glb(_YUP_FOOTPRINT), feature_table={"BATCH_LENGTH": 1})
        objects = tt._parse_b3dm_tile(tile, np.eye(4), up_axis="Y")
        assert self._extent(objects) == (0.0, 0.0, 2.0, 2.0)

    def test_up_axis_matrix_maps_axes_as_cesium_does(self) -> None:
        y_up = tt._up_axis_to_zup("Y")[:3, :3] @ np.array([0.0, 1.0, 0.0])
        x_up = tt._up_axis_to_zup("X")[:3, :3] @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(y_up, [0.0, 0.0, 1.0])
        assert np.allclose(x_up, [0.0, 0.0, 1.0])
        assert np.allclose(tt._up_axis_to_zup("Z"), np.eye(4))

    def test_z_up_tileset_opts_out_of_rotation(self, tmp_path) -> None:
        """A tileset declaring gltfUpAxis=Z keeps z-up geometry un-rotated."""
        (tmp_path / "flat.b3dm").write_bytes(
            _b3dm(_mesh_glb(_SQUARE), feature_table={"BATCH_LENGTH": 1})
        )
        tileset = tmp_path / "tileset.json"
        tileset.write_text(
            '{"asset": {"version": "1.0", "gltfUpAxis": "Z"}, "geometricError": 0, '
            '"root": {"content": {"uri": "flat.b3dm"}}}',
            encoding="utf-8",
        )
        model = tt.parse_3dtiles(tileset)
        extent = self._extent(list(model.objects.values()))
        assert extent == (0.0, 0.0, 2.0, 2.0)

    def test_no_scenes_falls_back_to_all_nodes(self) -> None:
        glb = _mesh_glb(_SQUARE)
        gltf, binary = tt._parse_glb(glb)
        del gltf["scenes"]
        del gltf["scene"]
        points, _ = tt._mesh_vertices(gltf, binary)
        assert len(points) == 4


def _square_at(dx: float) -> list:
    """A z-up unit square footprint shifted by ``dx`` along X."""
    return [(dx, 0.0, 0.0), (dx + 2.0, 0.0, 0.0), (dx + 2.0, 2.0, 0.0), (dx, 2.0, 0.0)]


def _two_tile_tileset(tmp_path, batch_table: dict | None = None):
    """Write a z-up, two-child tileset whose tiles sit at distinct X offsets; return its path."""
    (tmp_path / "a.b3dm").write_bytes(
        _b3dm(
            _mesh_glb(_square_at(0.0)),
            feature_table={"BATCH_LENGTH": 1},
            batch_table=batch_table,
        )
    )
    (tmp_path / "b.b3dm").write_bytes(
        _b3dm(
            _mesh_glb(_square_at(100.0)),
            feature_table={"BATCH_LENGTH": 1},
            batch_table=batch_table,
        )
    )
    tileset = tmp_path / "tileset.json"
    tileset.write_text(
        json.dumps(
            {
                "asset": {"version": "1.0", "gltfUpAxis": "Z"},
                "geometricError": 0,
                "root": {
                    "children": [
                        {"content": {"uri": "a.b3dm"}},
                        {"content": {"uri": "b.b3dm"}},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return tileset


class TestMultiTileIds:
    """Cross-tile feature ids must not silently drop buildings (MF1)."""

    def test_two_tiles_without_ids_keep_both_features(self, tmp_path) -> None:
        """Two id-less tiles both yield feature-0 locally; namespacing keeps both, not one."""
        model = tt.parse_3dtiles(_two_tile_tileset(tmp_path))
        assert len(model.objects) == 2
        xmins = sorted(
            min(point[0] for point in obj.geometries[0].boundaries[0][0])
            for obj in model.objects.values()
        )
        assert xmins == [0.0, 100.0]

    def test_duplicate_real_id_across_tiles_warns_and_keeps_both(
        self, tmp_path
    ) -> None:
        """Two tiles sharing the batch-table id 'X' are both kept (second re-keyed) with a warning."""
        with pytest.warns(UserWarning, match="occurs in more than one tile"):
            model = tt.parse_3dtiles(
                _two_tile_tileset(tmp_path, batch_table={"id": ["X"]})
            )
        assert len(model.objects) == 2
        assert "X" in model.objects and "X#2" in model.objects

    def test_insert_object_is_a_no_op_rename_for_unique_ids(self) -> None:
        from pyramids_citymodels.ir import CityObject

        objects: dict = {}
        tt._insert_object(objects, CityObject(id="a", type="Building"))
        tt._insert_object(objects, CityObject(id="b", type="Building"))
        assert sorted(objects) == ["a", "b"]
