"""Tests for 3D Tiles 1.1 implicit tiling (issue #55).

`to_3dtiles(..., implicit_tiling=True, scheme=...)` writes a 3D Tiles 1.1 implicit tileset: a root
`tileset.json` carrying only the root box + an `implicitTiling` rule, templated
`content/{level}/{x}/{y}[/{z}].b3dm` files, and a compact binary `.subtree` of availability bitstreams
— for both the quadtree (XY) and octree (XYZ) subdivision schemes. The whole tileset round-trips
through this package's own 3D-Tiles reader, which expands the subtree(s) instead of walking an explicit
tree. The low-level `_implicit_tiling` codec (Morton codes + the `.subtree` binary) is unit-tested too.
"""

import json
import struct
import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels import _implicit_tiling as it
from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import parse_3dtiles

# --------------------------------------------------------------------------- codec


@pytest.mark.parametrize("dimension", [2, 3])
def test_morton_round_trips(dimension: int) -> None:
    for a in range(6):
        for b in range(6):
            coords = (a, b) if dimension == 2 else (a, b, a ^ b)
            assert it.demorton(it.morton(coords), dimension) == coords


def test_level_offsets_and_node_count() -> None:
    assert [it.level_offset(level, 4) for level in range(4)] == [0, 1, 5, 21]
    assert it.node_count(3, 4) == 21  # 1 + 4 + 16
    assert it.node_count(2, 8) == 9  # 1 + 8


def test_bit_packing_is_lsb_first() -> None:
    bits = [True, False, False, False, False, False, False, False, True]
    packed = it.pack_bits(bits)
    assert packed[0] == 0b00000001  # bit 0 is the LSB of byte 0
    assert packed[1] == 0b00000001  # bit 8 is the LSB of byte 1
    assert it.unpack_bits(packed, len(bits)) == bits


def test_subtree_round_trips_with_a_bitstream() -> None:
    levels, children = 3, 4
    nodes = it.node_count(levels, children)
    tile = [False] * nodes
    content = [False] * nodes
    for cell in [(0, 0), (1, 1), (3, 2)]:  # occupied leaf cells at level 2
        for level in range(levels):
            ancestor = tuple(c >> (levels - 1 - level) for c in cell)
            tile[it.level_offset(level, children) + it.morton(ancestor)] = True
        content[it.level_offset(2, children) + it.morton(cell)] = True
    child = [False] * (children**levels)
    blob = it.encode_subtree(tile, content, child)
    assert blob[:4] == b"subt"
    back_tile, back_content, back_child = it.decode_subtree(blob, levels, children)
    assert back_tile == tile
    assert back_content == content
    assert back_child == child


def test_uniform_availability_is_stored_as_a_constant() -> None:
    # An all-false / all-true availability needs no bitstream — the binary chunk stays empty.
    blob = it.encode_subtree([False] * 5, [False] * 5, [False] * 16)
    json_length = struct.unpack_from("<Q", blob, 8)[0]
    document = json.loads(blob[24 : 24 + json_length])
    assert document["tileAvailability"] == {"constant": 0}
    assert "buffers" not in document  # no binary buffer for all-constant availability


# ----------------------------------------------------------------------- writer / round-trip


def _grid(n: int) -> CityModel:
    objects = {}
    for i in range(n):
        x, y, z = (i % 6) * 50.0, (i // 6) * 50.0, (i % 3) * 20.0
        surface = [[(x, y, z), (x + 10.0, y, z), (x + 10.0, y + 10.0, z), (x, y, z)]]
        objects[f"o{i}"] = CityObject(
            id=f"o{i}",
            type="Building",
            attributes={"h": float(i)},
            geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])],
        )
    return CityModel(objects=objects)


def _write(model: CityModel, **kw: object) -> Path:
    directory = Path(tempfile.mkdtemp())
    model.to_3dtiles(directory, implicit_tiling=True, **kw)  # type: ignore[arg-type]
    return directory


def test_quadtree_tileset_declares_implicit_tiling() -> None:
    directory = _write(_grid(30), max_objects_per_tile=4)
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    root = tileset["root"]
    assert tileset["asset"]["version"] == "1.1"
    assert root["implicitTiling"]["subdivisionScheme"] == "QUADTREE"
    assert (
        root["content"]["uri"] == "content/{level}/{x}/{y}.b3dm"
    )  # a template, not a file
    assert (
        root["implicitTiling"]["subtrees"]["uri"] == "subtrees/{level}/{x}/{y}.subtree"
    )
    assert (directory / "subtrees" / "0" / "0" / "0.subtree").exists()
    assert list(directory.glob("content/**/*.b3dm"))  # at least one content cell


def test_octree_templates_carry_the_z_axis() -> None:
    directory = _write(_grid(30), max_objects_per_tile=4, scheme="octree")
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    root = tileset["root"]
    assert root["implicitTiling"]["subdivisionScheme"] == "OCTREE"
    assert root["content"]["uri"] == "content/{level}/{x}/{y}/{z}.b3dm"
    assert (directory / "subtrees" / "0" / "0" / "0" / "0.subtree").exists()


@pytest.mark.parametrize("scheme", ["quadtree", "octree"])
def test_implicit_tileset_round_trips(scheme: str) -> None:
    model = _grid(30)
    directory = _write(model, max_objects_per_tile=4, scheme=scheme)
    back = parse_3dtiles(directory / "tileset.json")
    assert sorted(back.objects) == sorted(model.objects)  # every object recovered
    assert back.objects["o7"].attributes["h"] == 7.0  # batch-table attribute survives


def test_small_model_uses_a_single_level_and_still_round_trips() -> None:
    # Fewer objects than max → one level (the root cell), still a valid implicit tileset.
    model = _grid(3)
    directory = _write(model, max_objects_per_tile=64)
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    assert tileset["root"]["implicitTiling"]["subtreeLevels"] == 1
    back = parse_3dtiles(directory / "tileset.json")
    assert sorted(back.objects) == sorted(model.objects)


def test_subtree_content_availability_matches_written_content_files() -> None:
    directory = _write(_grid(30), max_objects_per_tile=4)
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    levels = tileset["root"]["implicitTiling"]["subtreeLevels"]
    blob = (directory / "subtrees" / "0" / "0" / "0.subtree").read_bytes()
    _tile, content, _child = it.decode_subtree(blob, levels, 4)
    assert sum(content) == len(
        list(directory.glob("content/**/*.b3dm"))
    )  # one bit per content file


def test_reader_follows_child_subtrees() -> None:
    # The reader must read an implicit tileset split across MULTIPLE subtrees (as other tools emit),
    # not only the single-subtree form this writer produces. Hand-build a 2-subtree quadtree
    # (subtreeLevels=1): an empty root subtree pointing at one child subtree that holds the content.
    from pyramids_citymodels.writers.threedtiles import build_b3dm

    directory = Path(tempfile.mkdtemp())
    surface = [
        [(100.0, 0.0, 0.0), (110.0, 0.0, 0.0), (110.0, 10.0, 0.0), (100.0, 0.0, 0.0)]
    ]
    obj = CityObject(
        id="c",
        type="Building",
        attributes={"h": 9.0},
        geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])],
    )
    content = (
        directory / "content" / "1" / "1" / "0.b3dm"
    )  # child cell at tileset level 1, (1, 0)
    content.parent.mkdir(parents=True, exist_ok=True)
    b3dm = build_b3dm(CityModel(objects={"c": obj}))
    assert b3dm is not None
    content.write_bytes(b3dm)

    # root subtree (level 0): root tile exists, no content, one child subtree at Morton 1 -> cell (1, 0).
    root_subtree = directory / "subtrees" / "0" / "0" / "0.subtree"
    root_subtree.parent.mkdir(parents=True, exist_ok=True)
    root_subtree.write_bytes(
        it.encode_subtree([True], [False], [False, True, False, False])
    )
    # child subtree rooted at level 1, (1, 0): one tile, with content, no further children.
    child_subtree = directory / "subtrees" / "1" / "1" / "0.subtree"
    child_subtree.parent.mkdir(parents=True, exist_ok=True)
    child_subtree.write_bytes(
        it.encode_subtree([True], [True], [False, False, False, False])
    )

    box = [0.0, 0.0, 0.0, 200.0, 0, 0, 0, 200.0, 0, 0, 0, 200.0]
    tileset = {
        "asset": {"version": "1.1", "gltfUpAxis": "Z"},
        "geometricError": 100.0,
        "root": {
            "boundingVolume": {"box": box},
            "refine": "REPLACE",
            "geometricError": 100.0,
            "implicitTiling": {
                "subdivisionScheme": "QUADTREE",
                "subtreeLevels": 1,
                "availableLevels": 2,
                "subtrees": {"uri": "subtrees/{level}/{x}/{y}.subtree"},
            },
            "content": {"uri": "content/{level}/{x}/{y}.b3dm"},
        },
    }
    (directory / "tileset.json").write_text(json.dumps(tileset), encoding="utf-8")
    back = parse_3dtiles(directory / "tileset.json")
    assert "c" in back.objects  # content reached via the child subtree
    assert back.objects["c"].attributes["h"] == 9.0
