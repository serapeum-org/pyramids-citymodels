"""Shared 3D Tiles 1.1 **implicit tiling** machinery: Morton codes, level maths, and the ``.subtree``
binary codec.

Implicit tiling (3D Tiles 1.1) replaces an enumerated ``tileset.json`` tree with a *rule*: a tile
declares a ``subdivisionScheme`` (``QUADTREE``/``OCTREE``) and ``subtreeLevels``, and small binary
**subtree** files carry three **availability** bitstreams — which tiles exist, which have content, and
which child subtrees exist. A tile is then addressed by its ``(level, x, y[, z])`` coordinate, so the
tree is described rather than listed.

This module is the piece the writer (`writers/threedtiles.py`) and the reader
(`readers/threedtiles.py`) share: the writer encodes availability into a subtree file, the reader
decodes one. The grid/subdivision policy itself lives in the writer; the per-tile expansion lives in
the reader. Availability within a subtree is indexed **level by level, Morton order within a level**:
the index of the node at ``(level, morton)`` is ``level_offset(level) + morton``.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import json
import struct
from typing import Any

#: The ``.subtree`` binary header: magic ``subt``, ``version`` (1), JSON byte length, binary byte length.
_SUBTREE_HEADER = struct.Struct("<4sIQQ")
_SUBTREE_MAGIC = b"subt"
_SUBTREE_VERSION = 1


def children_per_node(scheme: str) -> int:
    """4 for a ``QUADTREE``, 8 for an ``OCTREE`` (case-insensitive)."""
    return 8 if scheme.upper() == "OCTREE" else 4


def dimensions(scheme: str) -> int:
    """The number of subdivided axes: 2 for a ``QUADTREE``, 3 for an ``OCTREE``."""
    return 3 if scheme.upper() == "OCTREE" else 2


def level_offset(level: int, children: int) -> int:
    """The availability index of the first node of ``level`` (sum of all shallower levels' node counts)."""
    return int((children**level - 1) // (children - 1))


def node_count(levels: int, children: int) -> int:
    """The total number of nodes across ``levels`` levels of a complete ``children``-ary tree."""
    return int((children**levels - 1) // (children - 1))


def _part_1by1(value: int) -> int:
    """Spread the low 32 bits of ``value`` so bit *i* lands at position *2i* (2-D Morton interleave)."""
    value &= 0xFFFFFFFF
    value = (value | (value << 16)) & 0x0000FFFF0000FFFF
    value = (value | (value << 8)) & 0x00FF00FF00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F0F0F0F0F
    value = (value | (value << 2)) & 0x3333333333333333
    value = (value | (value << 1)) & 0x5555555555555555
    return value


def _compact_1by1(value: int) -> int:
    """Inverse of :func:`_part_1by1`: gather bits at even positions back into the low bits."""
    value &= 0x5555555555555555
    value = (value | (value >> 1)) & 0x3333333333333333
    value = (value | (value >> 2)) & 0x0F0F0F0F0F0F0F0F
    value = (value | (value >> 4)) & 0x00FF00FF00FF00FF
    value = (value | (value >> 8)) & 0x0000FFFF0000FFFF
    value = (value | (value >> 16)) & 0x00000000FFFFFFFF
    return value


def _part_1by2(value: int) -> int:
    """Spread the low 21 bits of ``value`` so bit *i* lands at position *3i* (3-D Morton interleave)."""
    value &= 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    value = (value | (value << 2)) & 0x1249249249249249
    return value


def _compact_1by2(value: int) -> int:
    """Inverse of :func:`_part_1by2`: gather bits at every-third position back into the low bits."""
    value &= 0x1249249249249249
    value = (value | (value >> 2)) & 0x10C30C30C30C30C3
    value = (value | (value >> 4)) & 0x100F00F00F00F00F
    value = (value | (value >> 8)) & 0x1F0000FF0000FF
    value = (value | (value >> 16)) & 0x1F00000000FFFF
    value = (value | (value >> 32)) & 0x1FFFFF
    return value


def morton(coordinates: tuple[int, ...]) -> int:
    """The Morton (Z-order) index that interleaves a cell's ``(x, y)`` or ``(x, y, z)`` coordinates."""
    if len(coordinates) == 2:
        return _part_1by1(coordinates[0]) | (_part_1by1(coordinates[1]) << 1)
    return (
        _part_1by2(coordinates[0])
        | (_part_1by2(coordinates[1]) << 1)
        | (_part_1by2(coordinates[2]) << 2)
    )


def demorton(index: int, dimension: int) -> tuple[int, ...]:
    """Recover a cell's ``(x, y)`` or ``(x, y, z)`` coordinates from its Morton ``index``."""
    if dimension == 2:
        return (_compact_1by1(index), _compact_1by1(index >> 1))
    return (
        _compact_1by2(index),
        _compact_1by2(index >> 1),
        _compact_1by2(index >> 2),
    )


def pack_bits(bits: list[bool]) -> bytes:
    """Pack booleans into bytes, LSB-first within each byte (3D Tiles availability bit order)."""
    out = bytearray((len(bits) + 7) // 8)
    for position, bit in enumerate(bits):
        if bit:
            out[position >> 3] |= 1 << (position & 7)
    return bytes(out)


def unpack_bits(data: bytes, count: int) -> list[bool]:
    """Unpack ``count`` booleans from ``data``, LSB-first within each byte (inverse of :func:`pack_bits`)."""
    return [
        bool(data[position >> 3] & (1 << (position & 7))) for position in range(count)
    ]


def _pad(blob: bytes, boundary: int, fill: bytes) -> bytes:
    """Right-pad ``blob`` with ``fill`` to a multiple of ``boundary`` bytes."""
    remainder = len(blob) % boundary
    return blob if remainder == 0 else blob + fill * (boundary - remainder)


def _availability(
    bits: list[bool], binary: bytearray, buffer_views: list[dict[str, int]]
) -> dict[str, Any]:
    """Encode one availability: ``{"constant": 0|1}`` when uniform, else a packed bitstream bufferView."""
    if not any(bits):
        return {"constant": 0}
    if all(bits):
        return {"constant": 1}
    packed = pack_bits(bits)
    byte_offset = len(binary)
    buffer_views.append(
        {"buffer": 0, "byteOffset": byte_offset, "byteLength": len(packed)}
    )
    binary.extend(packed)
    binary.extend(
        b"\x00" * (len(_pad(packed, 8, b"\x00")) - len(packed))
    )  # 8-align the next view
    return {"bitstream": len(buffer_views) - 1, "availableCount": sum(bits)}


def encode_subtree(
    tile_availability: list[bool],
    content_availability: list[bool],
    child_subtree_availability: list[bool],
) -> bytes:
    """Serialize the three availability arrays into one binary ``.subtree`` file.

    A uniform array is stored as a ``constant`` (no bytes); a mixed one as a packed, 8-byte-aligned
    bitstream bufferView over the single binary buffer. The layout is the 24-byte header, the padded
    JSON chunk, then the binary chunk.
    """
    binary = bytearray()
    buffer_views: list[dict[str, int]] = []
    document: dict[str, Any] = {
        "tileAvailability": _availability(tile_availability, binary, buffer_views),
        "contentAvailability": [
            _availability(content_availability, binary, buffer_views)
        ],
        "childSubtreeAvailability": _availability(
            child_subtree_availability, binary, buffer_views
        ),
    }
    if binary:
        document["buffers"] = [{"byteLength": len(binary)}]
        document["bufferViews"] = buffer_views
    json_chunk = _pad(json.dumps(document).encode("utf-8"), 8, b" ")
    header = _SUBTREE_HEADER.pack(
        _SUBTREE_MAGIC, _SUBTREE_VERSION, len(json_chunk), len(binary)
    )
    return header + json_chunk + bytes(binary)


def decode_subtree(
    data: bytes, subtree_levels: int, children: int
) -> tuple[list[bool], list[bool], list[bool]]:
    """Decode a ``.subtree`` file into ``(tile, content, child_subtree)`` availability arrays.

    The tile and content arrays have ``node_count(subtree_levels)`` entries; the child-subtree array
    has ``children ** subtree_levels`` (the cells one level below the subtree).
    """
    magic, _version, json_length, _binary_length = _SUBTREE_HEADER.unpack_from(data, 0)
    if magic != _SUBTREE_MAGIC:
        raise ValueError("not a 3D Tiles .subtree file (bad magic)")
    json_start = _SUBTREE_HEADER.size
    document = json.loads(data[json_start : json_start + json_length])
    binary = data[json_start + json_length :]
    tile_nodes = node_count(subtree_levels, children)
    child_nodes = children**subtree_levels

    def read(availability: dict[str, Any], count: int) -> list[bool]:
        if "constant" in availability:
            return [bool(availability["constant"])] * count
        view = document["bufferViews"][availability["bitstream"]]
        chunk = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        return unpack_bits(chunk, count)

    tile = read(document["tileAvailability"], tile_nodes)
    content = read(document["contentAvailability"][0], tile_nodes)
    child = read(document["childSubtreeAvailability"], child_nodes)
    return tile, content, child


def fill_template(template: str, level: int, coordinates: tuple[int, ...]) -> str:
    """Substitute ``{level}``/``{x}``/``{y}``/``{z}`` in an implicit-tiling URI template."""
    axes = {0: "x", 1: "y", 2: "z"}
    result = template.replace("{level}", str(level))
    for axis, value in enumerate(coordinates):
        result = result.replace("{" + axes[axis] + "}", str(value))
    return result
