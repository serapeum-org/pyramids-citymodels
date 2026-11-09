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

#: Upper bound on the cells in one subtree. A subtree's availability arrays are materialized in full,
#: so this bounds both the depth the writer chooses and what the reader accepts from an untrusted file
#: — a tiny ``.subtree`` that *declares* a huge ``subtreeLevels`` must not force a dense-tree allocation.
MAX_SUBTREE_NODES = 1 << 20


def max_subtree_levels(children: int) -> int:
    """The deepest subtree whose cell count (``children ** levels``) stays within :data:`MAX_SUBTREE_NODES`.

    Args:
        children: Children per node (4 for a quadtree, 8 for an octree).

    Returns:
        The largest ``levels`` (≥ 1) with ``children ** levels <= MAX_SUBTREE_NODES``.

    Examples:
        - A quadtree allows 10 levels, an octree 6, within the ~1M-cell bound:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import max_subtree_levels
            >>> max_subtree_levels(4)
            10
            >>> max_subtree_levels(8)
            6

            ```
    """
    levels = 1
    while children ** (levels + 1) <= MAX_SUBTREE_NODES:
        levels += 1
    return levels


def children_per_node(scheme: str) -> int:
    """4 for a ``QUADTREE``, 8 for an ``OCTREE`` (case-insensitive).

    Args:
        scheme: The subdivision scheme name (``"quadtree"`` / ``"octree"``, any case).

    Returns:
        The number of children each node splits into (4 or 8).

    Examples:
        - A quadtree node has four children, an octree eight:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import children_per_node
            >>> children_per_node("QUADTREE")
            4
            >>> children_per_node("octree")
            8

            ```
    """
    return 8 if scheme.upper() == "OCTREE" else 4


def dimensions(scheme: str) -> int:
    """The number of subdivided axes: 2 for a ``QUADTREE``, 3 for an ``OCTREE``.

    Args:
        scheme: The subdivision scheme name (``"quadtree"`` / ``"octree"``, any case).

    Returns:
        2 for a quadtree (XY), 3 for an octree (XYZ).

    Examples:
        - A quadtree subdivides two axes, an octree three:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import dimensions
            >>> dimensions("quadtree")
            2
            >>> dimensions("OCTREE")
            3

            ```
    """
    return 3 if scheme.upper() == "OCTREE" else 2


def level_offset(level: int, children: int) -> int:
    """The availability index of the first node of ``level`` (sum of all shallower levels' node counts).

    Args:
        level: The 0-based tree level.
        children: Children per node (4 for a quadtree, 8 for an octree).

    Returns:
        The index at which ``level``'s nodes begin in a level-order availability array.

    Examples:
        - Quadtree level offsets are 0, 1, 5, 21 (each level adds ``4**level`` nodes):
            ```python
            >>> from pyramids_citymodels._implicit_tiling import level_offset
            >>> [level_offset(level, 4) for level in range(4)]
            [0, 1, 5, 21]

            ```
    """
    return int((children**level - 1) // (children - 1))


def node_count(levels: int, children: int) -> int:
    """The total number of nodes across ``levels`` levels of a complete ``children``-ary tree.

    Args:
        levels: The number of levels (a one-level tree is just the root).
        children: Children per node (4 for a quadtree, 8 for an octree).

    Returns:
        The node total, i.e. the length of a subtree's tile/content availability array.

    Examples:
        - A 3-level quadtree has 1 + 4 + 16 = 21 nodes; a 2-level octree 1 + 8 = 9:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import node_count
            >>> node_count(3, 4)
            21
            >>> node_count(2, 8)
            9

            ```
    """
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
    """The Morton (Z-order) index that interleaves a cell's ``(x, y)`` or ``(x, y, z)`` coordinates.

    Args:
        coordinates: A cell's ``(x, y)`` (quadtree) or ``(x, y, z)`` (octree) grid position.

    Returns:
        The Morton code locating the cell within its level's availability array.

    Examples:
        - ``x`` occupies the even bits, ``y`` the odd, so ``(1, 0)`` and ``(0, 1)`` differ by a bit:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import morton
            >>> morton((1, 0))
            1
            >>> morton((0, 1))
            2
            >>> morton((1, 1))
            3

            ```
    """
    if len(coordinates) == 2:
        return _part_1by1(coordinates[0]) | (_part_1by1(coordinates[1]) << 1)
    return (
        _part_1by2(coordinates[0])
        | (_part_1by2(coordinates[1]) << 1)
        | (_part_1by2(coordinates[2]) << 2)
    )


def demorton(index: int, dimension: int) -> tuple[int, ...]:
    """Recover a cell's ``(x, y)`` or ``(x, y, z)`` coordinates from its Morton ``index``.

    Args:
        index: A Morton code (see :func:`morton`).
        dimension: 2 for a quadtree cell, 3 for an octree cell.

    Returns:
        The cell's ``(x, y)`` or ``(x, y, z)`` coordinates — the inverse of :func:`morton`.

    Examples:
        - ``demorton`` undoes ``morton`` and splits a code back into its axes:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import morton, demorton
            >>> demorton(3, 2)
            (1, 1)
            >>> demorton(morton((2, 3)), 2)
            (2, 3)
            >>> demorton(morton((1, 2, 3)), 3)
            (1, 2, 3)

            ```
    """
    compact = _compact_1by1 if dimension == 2 else _compact_1by2
    return tuple(compact(index >> axis) for axis in range(dimension))


def pack_bits(bits: list[bool]) -> bytes:
    """Pack booleans into bytes, LSB-first within each byte (3D Tiles availability bit order).

    Args:
        bits: The availability flags, bit 0 first.

    Returns:
        The packed bytes; bit ``i`` is the ``i & 7``-th bit (LSB-first) of byte ``i >> 3``.

    Examples:
        - Bits 0 and 2 set give ``0b00000101`` = ``0x05``; bit 8 is the LSB of byte 1:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import pack_bits
            >>> pack_bits([True, False, True]).hex()
            '05'
            >>> pack_bits([False] * 8 + [True]).hex()
            '0001'

            ```
    """
    out = bytearray((len(bits) + 7) // 8)
    for position, bit in enumerate(bits):
        if bit:
            out[position >> 3] |= 1 << (position & 7)
    return bytes(out)


def unpack_bits(data: bytes, count: int) -> list[bool]:
    """Unpack ``count`` booleans from ``data``, LSB-first within each byte (inverse of :func:`pack_bits`).

    Args:
        data: Packed availability bytes.
        count: How many bits to read.

    Returns:
        ``count`` booleans, bit 0 first.

    Examples:
        - Reading back the bits packed into ``0x05``:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import unpack_bits
            >>> unpack_bits(b"\\x05", 3)
            [True, False, True]

            ```
    """
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

    Args:
        tile_availability: Which of the subtree's nodes exist, level-order Morton-indexed.
        content_availability: Which existing nodes have tile content.
        child_subtree_availability: Which child subtrees exist (the cells one level below).

    Returns:
        The ``.subtree`` file bytes, beginning with the ``subt`` magic.

    Examples:
        - Encoding a one-node subtree yields a file that starts with the ``subt`` magic and decodes
          back to the same arrays:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import encode_subtree, decode_subtree
            >>> blob = encode_subtree([True], [True], [False, False, False, False])
            >>> blob[:4]
            b'subt'
            >>> decode_subtree(blob, 1, 4)
            ([True], [True], [False, False, False, False])

            ```
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

    Args:
        data: The ``.subtree`` file bytes.
        subtree_levels: How many levels the subtree spans (sets the array lengths).
        children: Children per node (4 for a quadtree, 8 for an octree).

    Returns:
        ``(tile, content, child_subtree)`` availability arrays of booleans.

    Raises:
        ValueError: If ``data`` is not a well-formed ``.subtree`` (bad magic, truncated header,
            unparseable JSON, missing availability), or if ``subtree_levels`` is out of the supported
            range — ``children ** subtree_levels`` must not exceed :data:`MAX_SUBTREE_NODES`, so a tiny
            file that declares a huge subtree is rejected rather than materialized.

    Examples:
        - A bad-magic file is rejected:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import decode_subtree
            >>> decode_subtree(b"XXXX" + b"\\x00" * 20, 1, 4)
            Traceback (most recent call last):
                ...
            ValueError: not a 3D Tiles .subtree file (bad magic)

            ```
    """
    # Compare the *exponent* against the cap, never ``children ** subtree_levels`` — computing that
    # power for an untrusted ``subtree_levels`` (e.g. 1e9) is itself the DoS (a multi-hundred-MB bigint).
    if subtree_levels < 1 or subtree_levels > max_subtree_levels(children):
        raise ValueError(
            f"subtreeLevels {subtree_levels} is out of the supported range for a "
            f"{children}-ary subtree (cells would exceed {MAX_SUBTREE_NODES})"
        )
    if data[:4] != _SUBTREE_MAGIC:
        raise ValueError("not a 3D Tiles .subtree file (bad magic)")
    tile_nodes = node_count(subtree_levels, children)
    child_nodes = children**subtree_levels
    try:
        _magic, _version, json_length, _binary_length = _SUBTREE_HEADER.unpack_from(
            data, 0
        )
        json_start = _SUBTREE_HEADER.size
        document = json.loads(data[json_start : json_start + json_length])
        binary = data[json_start + json_length :]

        def read(availability: dict[str, Any], count: int) -> list[bool]:
            if "constant" in availability:
                return [bool(availability["constant"])] * count
            view = document["bufferViews"][availability["bitstream"]]
            chunk = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
            return unpack_bits(chunk, count)

        tile = read(document["tileAvailability"], tile_nodes)
        content = read(document["contentAvailability"][0], tile_nodes)
        child = read(document["childSubtreeAvailability"], child_nodes)
    except (struct.error, ValueError, KeyError, IndexError, TypeError) as error:
        raise ValueError(f"malformed 3D Tiles .subtree file: {error}") from error
    return tile, content, child


def fill_template(template: str, level: int, coordinates: tuple[int, ...]) -> str:
    """Substitute ``{level}``/``{x}``/``{y}``/``{z}`` in an implicit-tiling URI template.

    Args:
        template: A 3D Tiles implicit content/subtrees URI template.
        level: The tile level to substitute for ``{level}``.
        coordinates: The ``(x, y)`` or ``(x, y, z)`` cell coordinates to substitute.

    Returns:
        The resolved path (unused placeholders, e.g. ``{z}`` for a quadtree, are left untouched).

    Examples:
        - Resolving a quadtree content path and an octree subtree path:
            ```python
            >>> from pyramids_citymodels._implicit_tiling import fill_template
            >>> fill_template("content/{level}/{x}/{y}.b3dm", 2, (1, 3))
            'content/2/1/3.b3dm'
            >>> fill_template("subtrees/{level}/{x}/{y}/{z}.subtree", 0, (0, 0, 0))
            'subtrees/0/0/0/0.subtree'

            ```
    """
    axes = {0: "x", 1: "y", 2: "z"}
    result = template.replace("{level}", str(level))
    for axis, value in enumerate(coordinates):
        result = result.replace("{" + axes[axis] + "}", str(value))
    return result
