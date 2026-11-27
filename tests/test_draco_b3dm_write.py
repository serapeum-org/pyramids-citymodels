"""Tests for writing Draco-compressed 3D Tiles b3dm content (issue #84).

`to_3dtiles(content="b3dm", compression="draco")` Draco-compresses the mesh *inside* the b3dm's embedded
glb (`KHR_draco_mesh_compression`), reusing the `[draco]` extra from #83. The b3dm path carries per-object
identity in its batch table and a `_BATCHID` vertex attribute (not the glTF path's `_FEATURE_ID_0` +
metadata extensions); `_BATCHID` rides as a lossless uint32 Draco generic attribute, so batch ids survive
exactly and a tile round-trips through this package's own reader to the same features and footprints as the
uncompressed b3dm. The whole module needs `DracoPy`, so it is marked `draco`.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.errors import UnsupportedGeometryError
from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import parse_3dtiles

pytestmark = pytest.mark.draco


def _square(
    x: float, y: float, z: float = 0.0
) -> list[list[tuple[float, float, float]]]:
    """A 4-unit square surface at ``(x, y)`` on the ``z`` plane."""
    return [[(x, y, z), (x + 4, y, z), (x + 4, y + 4, z), (x, y + 4, z)]]


def _model(count: int = 2, surfaces: int = 1) -> CityModel:
    """``count`` buildings on a diagonal, each a stack of ``surfaces`` squares with distinctive attrs."""
    return CityModel(
        objects={
            f"o{i}": CityObject(
                id=f"o{i}",
                type="Building" if i % 2 == 0 else "Bridge",
                attributes={"height": 3.5 + i, "floors": i},
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            _square(i * 50.0, i * 50.0, 3.0 * s)
                            for s in range(surfaces)
                        ],
                    )
                ],
            )
            for i in range(count)
        }
    )


def _footprint(obj: CityObject) -> tuple[float, float, float, float]:
    """The (minx, miny, maxx, maxy) extent of a read-back object's surfaces."""
    xs = [
        p[0]
        for g in obj.geometries
        for surf in g.boundaries
        for ring in surf
        for p in ring
    ]
    ys = [
        p[1]
        for g in obj.geometries
        for surf in g.boundaries
        for ring in surf
        for p in ring
    ]
    return (min(xs), min(ys), max(xs), max(ys))


def _write(model: CityModel, **kwargs) -> Path:
    """Write ``model`` into a fresh temp dir and return the directory."""
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, **kwargs)
    return out


def _assert_same_features(draco: CityModel, plain: CityModel, tol: float = 0.2) -> None:
    """Assert two reads recovered the same object ids with per-id footprints within ``tol``."""
    assert set(draco.objects) == set(plain.objects), "same object ids as uncompressed"
    for object_id in draco.objects:
        drift = max(
            abs(a - b)
            for a, b in zip(
                _footprint(draco.objects[object_id]),
                _footprint(plain.objects[object_id]),
            )
        )
        assert drift < tol, (
            f"{object_id} footprint drifted {drift} (>= tolerance {tol})"
        )


def _glb_in_b3dm(b3dm: bytes) -> dict:
    """Parse the glTF JSON out of the glb embedded in a b3dm tile."""
    header = struct.Struct("<4sIIIIII")
    magic, _ver, _total, ft_json, ft_bin, bt_json, bt_bin = header.unpack_from(b3dm, 0)
    assert magic == b"b3dm", "b3dm magic"
    glb = b3dm[header.size + ft_json + ft_bin + bt_json + bt_bin :]
    assert glb[:4] == b"glTF", "embedded glb magic"
    json_len = struct.unpack_from("<I", glb, 12)[0]
    return json.loads(glb[20 : 20 + json_len])


class TestDracoB3dmRoundTrip:
    """A Draco-compressed b3dm reads back to the same features as its uncompressed twin."""

    def test_single_tile_matches_uncompressed(self) -> None:
        """A single Draco b3dm recovers the same object ids, types, attributes and footprints.

        Test scenario:
            The Draco and uncompressed b3dm writes of one model read back to identical feature sets,
            proving POSITION and the _BATCHID batch id survive compression.
        """
        model = _model(2)
        draco = parse_3dtiles(
            _write(model, content="b3dm", compression="draco") / "tileset.json"
        )
        plain = parse_3dtiles(_write(model, content="b3dm") / "tileset.json")
        _assert_same_features(draco, plain)
        assert {o.type for o in draco.objects.values()} == {"Building", "Bridge"}, (
            "types kept"
        )

    def test_attributes_round_trip(self) -> None:
        """The batch table's per-object attributes survive a Draco b3dm round-trip.

        Test scenario:
            Reading back a 4-feature Draco b3dm yields the input ``floors`` values 0..3, confirming the
            batch table (which carries identity) is unaffected by compressing the embedded glb.
        """
        back = parse_3dtiles(
            _write(_model(4), content="b3dm", compression="draco") / "tileset.json"
        )
        floors = sorted(obj.attributes["floors"] for obj in back.objects.values())
        assert floors == [0, 1, 2, 3], "batch-table attributes survive"

    @pytest.mark.parametrize(
        "mode",
        [
            {"max_objects_per_tile": 1},
            {"implicit_tiling": True, "max_objects_per_tile": 1},
            {"max_objects_per_tile": 1, "simplify": 0.02},
        ],
        ids=["explicit-multitile", "implicit", "simplify-lod"],
    )
    def test_draco_b3dm_composes_with_tiling(self, mode: dict) -> None:
        """Draco b3dm composes with every tiling mode and per-tile LOD.

        Args:
            mode: Extra ``to_3dtiles`` keywords combined with ``content="b3dm", compression="draco"``.

        Test scenario:
            Each mode's Draco b3dm recovers the same objects/footprints as the uncompressed b3dm.
        """
        model = _model(4)
        draco = parse_3dtiles(
            _write(model, content="b3dm", compression="draco", **mode) / "tileset.json"
        )
        plain = parse_3dtiles(_write(model, content="b3dm", **mode) / "tileset.json")
        _assert_same_features(draco, plain)


class TestDracoB3dmStructure:
    """The embedded glb declares the Draco extension and keeps _BATCHID as an integer attribute."""

    def _glb(self) -> dict:
        b3dm = (
            _write(_model(2), content="b3dm", compression="draco") / "content.b3dm"
        ).read_bytes()
        return _glb_in_b3dm(b3dm)

    def test_extension_required_and_maps_batchid(self) -> None:
        """KHR_draco is required and the extension maps POSITION and _BATCHID to Draco ids.

        Test scenario:
            The embedded glb's primitive carries the compression extension (required), mapping both the
            position and the _BATCHID attribute the reader resolves.
        """
        doc = self._glb()
        assert doc["extensionsRequired"] == ["KHR_draco_mesh_compression"], "required"
        ext = doc["meshes"][0]["primitives"][0]["extensions"][
            "KHR_draco_mesh_compression"
        ]
        assert set(ext["attributes"]) == {"POSITION", "_BATCHID"}, (
            "both attributes mapped"
        )

    def test_batchid_accessor_is_unsigned_int_and_bufferviewless(self) -> None:
        """_BATCHID stays an UNSIGNED_INT accessor (lossless uint32 Draco attr), without a bufferView.

        Test scenario:
            Unlike the glTF path's FLOAT _FEATURE_ID_0, b3dm keeps _BATCHID as UNSIGNED_INT (5125); and
            the geometry accessors drop bufferView under Draco.
        """
        accessors = self._glb()["accessors"]
        assert accessors[1]["componentType"] == 5125, (
            "_BATCHID accessor is UNSIGNED_INT"
        )
        assert all("bufferView" not in a for a in accessors[:3]), (
            "geometry accessors are bufferView-less"
        )


class TestDracoB3dmCompressionAndGuards:
    """Draco b3dm shrinks geometry, leaves the uncompressed path intact, and gates the backend."""

    def test_draco_shrinks_a_dense_b3dm(self) -> None:
        """The Draco b3dm is smaller than the uncompressed one for a multi-surface model.

        Test scenario:
            A 60-building, three-surface model compresses to well under the uncompressed b3dm size.
        """
        model = _model(60, surfaces=3)
        draco = (
            _write(model, content="b3dm", compression="draco") / "content.b3dm"
        ).read_bytes()
        plain = (_write(model, content="b3dm") / "content.b3dm").read_bytes()
        assert len(draco) < len(plain), (
            f"draco {len(draco)}B should be < plain {len(plain)}B"
        )

    def test_uncompressed_b3dm_byte_identical(self) -> None:
        """``compression=None`` b3dm is byte-identical to omitting the argument.

        Test scenario:
            The new branch must not perturb the uncompressed b3dm output.
        """
        model = _model(3)
        explicit = (
            _write(model, content="b3dm", compression=None) / "content.b3dm"
        ).read_bytes()
        default = (_write(model, content="b3dm") / "content.b3dm").read_bytes()
        assert explicit == default, "compression=None must not alter uncompressed b3dm"

    def test_missing_dracopy_raises_install_hint(self, monkeypatch) -> None:
        """Draco b3dm with DracoPy unimportable raises the install-hint error before writing.

        Test scenario:
            With DracoPy masked, requesting Draco b3dm raises UnsupportedGeometryError naming the extra
            and writes no files (atomicity pre-flight).
        """
        monkeypatch.setitem(sys.modules, "DracoPy", None)
        out = Path(tempfile.mkdtemp())
        model = _model(4)
        with pytest.raises(
            UnsupportedGeometryError, match=r"pyramids-citymodels\[draco\]"
        ):
            model.to_3dtiles(
                out, content="b3dm", compression="draco", max_objects_per_tile=1
            )
        assert list(out.iterdir()) == [], "no orphaned files when the backend is absent"
