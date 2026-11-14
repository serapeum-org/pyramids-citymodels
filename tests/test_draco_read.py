"""Tests for reading Draco-compressed 3D Tiles glTF (``KHR_draco_mesh_compression``, issue #80).

A Draco-compressed tile carries its geometry as a Draco bitstream in one ``bufferView``; the
``KHR_draco_mesh_compression`` extension maps each glTF attribute (``POSITION``, ``_BATCHID`` /
``_FEATURE_ID_0``) to a Draco unique id. With the optional ``draco`` extra installed the reader decodes
it with ``DracoPy`` and produces the same features and footprints as the uncompressed equivalent; without
the extra a Draco tile raises a clear "install the draco extra" error, and uncompressed tiles are
unaffected. These tests build their own Draco fixtures in-process (no committed binary), so they are
marked ``draco`` and need the extra importable.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import json
import struct
import sys

import numpy as np
import pytest

from pyramids_citymodels.errors import UnsupportedGeometryError
from pyramids_citymodels.readers import threedtiles as reader
from pyramids_citymodels.readers.threedtiles import parse_3dtiles

pytestmark = pytest.mark.draco

# Two squares authored y-up, identical to the uncommitted uncompressed fixture: after the default
# gltfUpAxis="Y" rotation (x, y, z) -> (x, -z, y) the plan footprints are feature 0 -> (0,0)..(2,2) and
# feature 1 -> (5,5)..(7,7).
_SQUARES = {
    0: [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, -2.0), (0.0, 0.0, -2.0)],
    1: [(5.0, 0.0, -5.0), (7.0, 0.0, -5.0), (7.0, 0.0, -7.0), (5.0, 0.0, -7.0)],
}


def _pad(data: bytes, boundary: int, fill: bytes) -> bytes:
    remainder = len(data) % boundary
    return data if remainder == 0 else data + fill * (boundary - remainder)


def _draco_glb(squares: dict, feature_attr: str, with_feature: bool = True) -> bytes:
    """A binary glTF whose single primitive is Draco-compressed, with POSITION (+ a feature id).

    Encodes the squares' corners (and, when ``with_feature``, a per-vertex feature id) with ``DracoPy``,
    discovers the Draco unique ids back from the encoded stream, and wires them into the
    ``KHR_draco_mesh_compression`` attribute map so the fixture is self-consistent.
    """
    import DracoPy

    positions: list[tuple[float, float, float]] = []
    feature_ids: list[int] = []
    for feature, corners in squares.items():
        for point in corners:
            positions.append(point)
            feature_ids.append(feature)
    points = np.array(positions, dtype=np.float64)
    count = len(points)
    # Two triangles per square, indices into the flat vertex list.
    faces = []
    for base in range(0, count, 4):
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
    generic = (
        {0: np.array(feature_ids, dtype=np.uint32).reshape(-1, 1)}
        if with_feature
        else None
    )
    blob = DracoPy.encode(
        points, np.array(faces, dtype=np.uint32), generic_attributes=generic
    )

    # Discover the Draco unique ids the encoder assigned, by attribute type (0 = POSITION, 4 = GENERIC).
    decoded = DracoPy.decode(blob)
    position_uid = next(
        a["unique_id"] for a in decoded.attributes if a["attribute_type"] == 0
    )
    draco_attributes = {"POSITION": position_uid}
    primitive_attributes = {"POSITION": 0}
    accessors = [{"componentType": 5126, "count": count, "type": "VEC3"}]
    if with_feature:
        generic_uid = next(
            a["unique_id"] for a in decoded.attributes if a["attribute_type"] == 4
        )
        draco_attributes[feature_attr] = generic_uid
        primitive_attributes[feature_attr] = 1
        accessors.append({"componentType": 5125, "count": count, "type": "SCALAR"})

    binary = _pad(blob, 4, b"\x00")
    gltf = {
        "asset": {"version": "2.0"},
        "extensionsRequired": ["KHR_draco_mesh_compression"],
        "extensionsUsed": ["KHR_draco_mesh_compression"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": primitive_attributes,
                        "mode": 4,
                        "extensions": {
                            "KHR_draco_mesh_compression": {
                                "bufferView": 0,
                                "attributes": draco_attributes,
                            }
                        },
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(blob)}],
        "accessors": accessors,
    }
    json_chunk = _pad(json.dumps(gltf).encode("utf-8"), 4, b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    header = struct.pack("<III", 0x46546C67, 2, total)
    json_part = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    bin_part = struct.pack("<II", len(binary), 0x004E4942) + binary
    return header + json_part + bin_part


def _wrap_b3dm(glb: bytes, batch_length: int) -> bytes:
    """Wrap a glb in a b3dm with a BATCH_LENGTH feature table and an id/type batch table."""
    feature_table = _pad(
        json.dumps({"BATCH_LENGTH": batch_length}).encode("utf-8"), 8, b" "
    )
    batch_table = _pad(
        json.dumps(
            {
                "id": [f"b{i}" for i in range(batch_length)],
                "type": ["Building"] * batch_length,
            }
        ).encode("utf-8"),
        8,
        b" ",
    )
    body = feature_table + batch_table + glb
    header = struct.pack(
        "<4sIIIIII",
        b"b3dm",
        1,
        28 + len(body),
        len(feature_table),
        0,
        len(batch_table),
        0,
    )
    return header + body


def _tileset(tmp_path, content_name: str) -> "object":
    tileset = tmp_path / "tileset.json"
    tileset.write_text(
        json.dumps(
            {
                "asset": {"version": "1.0"},
                "geometricError": 100.0,
                "root": {
                    "boundingVolume": {"box": [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]},
                    "geometricError": 0.0,
                    "refine": "ADD",
                    "content": {"uri": content_name},
                },
            }
        ),
        encoding="utf-8",
    )
    return tileset


def _bounds(city_object) -> tuple[float, float, float, float]:
    """The XY bounding box of a read feature's footprint ring."""
    ring = city_object.geometries[0].boundaries[0][0]
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


class TestDracoB3dm:
    """A Draco-compressed b3dm reads to the same features/footprints as its uncompressed twin."""

    def test_two_features_round_trip_with_footprints(self, tmp_path) -> None:
        """Both ``_BATCHID`` features decode to their known XY footprints (0,0)-(2,2) and (5,5)-(7,7).

        Test scenario:
            A two-square Draco b3dm reads as two objects whose convex-hull footprints match the extents
            the uncompressed fixture asserts, proving POSITION *and* the feature id survive the decode.
        """
        glb = _draco_glb(_SQUARES, "_BATCHID")
        (tmp_path / "draco.b3dm").write_bytes(_wrap_b3dm(glb, 2))
        model = parse_3dtiles(_tileset(tmp_path, "draco.b3dm"))
        assert set(model.objects) == {"b0", "b1"}, "both batched features must be read"
        bounds = {oid: _bounds(obj) for oid, obj in model.objects.items()}
        assert bounds["b0"] == pytest.approx((0, 0, 2, 2), abs=0.05)
        assert bounds["b1"] == pytest.approx((5, 5, 7, 7), abs=0.05)

    def test_single_feature_without_batch_ids(self, tmp_path) -> None:
        """A Draco tile with no feature id reads as a single whole-tile feature, no crash.

        Test scenario:
            One square, no ``_BATCHID`` attribute — the tile is one feature spanning its footprint.
        """
        glb = _draco_glb({0: _SQUARES[0]}, "_BATCHID", with_feature=False)
        (tmp_path / "draco.b3dm").write_bytes(_wrap_b3dm(glb, 1))
        model = parse_3dtiles(_tileset(tmp_path, "draco.b3dm"))
        assert len(model.objects) == 1, "a feature-id-less Draco tile is one feature"
        bounds = _bounds(next(iter(model.objects.values())))
        assert bounds == pytest.approx((0, 0, 2, 2), abs=0.05)


class TestDracoGltfContent:
    """A Draco-compressed 3D Tiles 1.1 glTF (.glb) tile reads via its ``_FEATURE_ID_0`` attribute."""

    def test_glb_content_two_features(self, tmp_path) -> None:
        """A bare Draco ``.glb`` tile's two ``_FEATURE_ID_0`` features decode to their footprints.

        Test scenario:
            The same two squares as a glTF-content tile (no b3dm wrapper) read as two features with the
            expected footprints — covering the ``_FEATURE_ID_0`` attribute path, not just ``_BATCHID``.
        """
        glb = _draco_glb(_SQUARES, "_FEATURE_ID_0")
        (tmp_path / "content.glb").write_bytes(glb)
        model = parse_3dtiles(_tileset(tmp_path, "content.glb"))
        assert len(model.objects) == 2, "both glTF-content features must be read"
        extents = sorted(_bounds(obj)[0] for obj in model.objects.values())
        assert extents == pytest.approx([0, 5], abs=0.05)


class TestDracoBackendAbsent:
    """Without the extra, a Draco tile fails fast with an actionable error; meshopt stays rejected."""

    def test_missing_dracopy_raises_install_hint(self, tmp_path, monkeypatch) -> None:
        """A Draco tile with ``DracoPy`` unimportable raises a clear "install the draco extra" error.

        Test scenario:
            ``DracoPy`` is masked in ``sys.modules`` so the lazy import fails; reading a Draco tile then
            raises ``UnsupportedGeometryError`` naming the extra, rather than an ImportError or a crash.
        """
        glb = _draco_glb(_SQUARES, "_BATCHID")
        (tmp_path / "draco.b3dm").write_bytes(_wrap_b3dm(glb, 2))
        monkeypatch.setitem(
            sys.modules, "DracoPy", None
        )  # force ImportError on `import DracoPy`
        with pytest.raises(
            UnsupportedGeometryError, match=r"draco.*extra|extra.*draco"
        ):
            parse_3dtiles(_tileset(tmp_path, "draco.b3dm"))

    def test_corrupt_draco_payload_is_wrapped(self, tmp_path) -> None:
        """A truncated Draco bitstream raises ``UnsupportedGeometryError``, not a raw decoder error.

        Test scenario:
            A primitive points at a bufferView of garbage bytes; the decode failure is wrapped so a
            caller walking a tileset can catch one error type to skip a bad tile.
        """
        glb = _draco_glb(_SQUARES, "_BATCHID")
        corrupt = bytearray(glb)
        # Flip bytes in the BIN chunk (the Draco payload) near the end to corrupt it.
        corrupt[-8:] = b"\x00" * 8
        (tmp_path / "draco.b3dm").write_bytes(_wrap_b3dm(bytes(corrupt), 2))
        with pytest.raises(UnsupportedGeometryError):
            parse_3dtiles(_tileset(tmp_path, "draco.b3dm"))


def test_reject_function_passes_draco_when_backend_present() -> None:
    """``_reject_unsupported_compression`` lets a Draco-required glTF through when DracoPy is importable."""
    reader._reject_unsupported_compression(
        {"extensionsRequired": ["KHR_draco_mesh_compression"]}
    )
