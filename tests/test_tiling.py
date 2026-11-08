"""Tests for multi-tile 3D Tiles tiling (issue #51).

With `max_objects_per_tile` set, `to_3dtiles` partitions objects by footprint into an XY quadtree —
one `content_<n>.b3dm` per leaf and a nested `tileset.json` of bounding-volume tiles — and the whole
tileset round-trips through this package's own 3D-Tiles reader. Without it, a single tile is written.
"""

import json
import struct
import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import parse_3dtiles
from pyramids_citymodels.writers.threedtiles import _B3DM_HEADER


def _grid_model(n: int) -> CityModel:
    """``n`` single-triangle buildings spread across a grid so they fall into different quadrants."""
    objects = {}
    for i in range(n):
        x, y = (i % 8) * 100.0, (i // 8) * 100.0
        surface = [[(x, y, 0.0), (x + 10.0, y, 0.0), (x + 10.0, y + 10.0, 0.0)]]
        objects[f"o{i}"] = CityObject(
            id=f"o{i}",
            type="Building",
            geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])],
        )
    return CityModel(objects=objects)


def _write(model: CityModel, **kw: object) -> Path:
    directory = Path(tempfile.mkdtemp())
    model.to_3dtiles(directory, **kw)  # type: ignore[arg-type]
    return directory


def test_no_max_writes_a_single_tile() -> None:
    directory = _write(_grid_model(20))
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    assert tileset["root"]["content"]["uri"] == "content.b3dm"
    assert "children" not in tileset["root"]
    assert (directory / "content.b3dm").exists()


def test_small_model_stays_single_even_with_max() -> None:
    directory = _write(_grid_model(3), max_objects_per_tile=4)
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    assert "children" not in tileset["root"]  # 3 <= 4, not subdivided


def test_large_model_is_subdivided_into_a_tile_tree() -> None:
    directory = _write(_grid_model(32), max_objects_per_tile=4)
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    assert "children" in tileset["root"]  # subdivided
    assert "content" not in tileset["root"]  # the root is internal, not a leaf
    leaves = sorted(directory.glob("content_*.b3dm"))
    assert len(leaves) > 1  # several leaf tiles


def test_every_tile_has_a_bounding_volume() -> None:
    directory = _write(_grid_model(32), max_objects_per_tile=4)
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))

    def check(node: dict) -> None:
        assert len(node["boundingVolume"]["box"]) == 12
        for child in node.get("children", []):
            check(child)

    check(tileset["root"])


def test_all_objects_round_trip_through_the_tile_tree() -> None:
    model = _grid_model(32)
    directory = _write(model, max_objects_per_tile=4)
    back = parse_3dtiles(directory / "tileset.json")
    assert sorted(back.objects) == sorted(model.objects)  # all 32 recovered


def test_each_leaf_b3dm_is_8_byte_aligned() -> None:
    directory = _write(_grid_model(32), max_objects_per_tile=4)
    for leaf in directory.glob("content_*.b3dm"):
        data = leaf.read_bytes()
        _magic, _ver, length, ftj, ftb, btj, btb = _B3DM_HEADER.unpack_from(data, 0)
        glb_offset = _B3DM_HEADER.size + ftj + ftb + btj + btb
        assert glb_offset % 8 == 0
        assert len(data) % 8 == 0
        assert length == len(data)
        _gm, _gv, glb_len = struct.unpack_from("<III", data[glb_offset:], 0)
        assert glb_len % 8 == 0
