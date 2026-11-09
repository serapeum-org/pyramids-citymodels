"""Serialize the IR to a Cesium 3D Tiles tileset (feature #15 — tiling / packaging).

3D Tiles is the runtime streaming format for massive city models: a ``tileset.json`` tree of bounding
volumes whose leaves point at **binary** ``b3dm`` (Batched 3D Model) content. This writer produces,
depending on :func:`write_3dtiles`'s arguments, a **single-tile** tileset (one ``tileset.json`` + one
``.b3dm``), an **explicit multi-tile** XY quadtree (one ``.b3dm`` per leaf, the tree enumerated in the
JSON), or a **3D Tiles 1.1 implicit** tileset (a quadtree/octree described by a rule: templated content
URIs + a compact binary ``.subtree`` of availability, so a huge tree is not enumerated — see
:mod:`pyramids_citymodels._implicit_tiling`). All three round-trip through this package's own 3D-Tiles
reader.

The ``b3dm`` embeds a binary glTF (``glb``) built the same way as the :mod:`glTF writer
<pyramids_citymodels.writers.gltf>` — every surface's exterior ring fan-triangulated into one
positions mesh — but with two additions that make it a *batched* model:

- a ``_BATCHID`` vertex attribute assigning every vertex to its source :class:`CityObject` (one batch
  per object), so a client can pick and style buildings individually;
- a **batch table** carrying each object's ``id``, ``type`` and attributes as parallel columns.

So the tileset round-trips through this package's own 3D-Tiles reader back to one footprint
:class:`CityObject` per batch — each recovering its id, type and attribute **values**. Note the
batch table is *columnar*: the identity columns (``id``/``type``) surface as attribute keys on read,
and an attribute present on only some objects is ``null`` for the rest (the reader drops those
``null``\\ s), so the read-back ``attributes`` dict is not byte-identical to the input. The tileset
declares ``asset.gltfUpAxis = "Z"`` (the IR is z-up), so a reader applies no up-axis rotation and
footprints are the true plan view.

Deliberate, documented limits (plain geometry, like the OBJ/glTF writers):

- **One tile, exterior rings only**, fan-triangulated — interior rings (holes) are dropped, and a
  strongly non-convex polygon's fan can self-overlap.
- **Positions + batch ids only** — no normals, materials or textures; appearance round-trips through
  the CityJSON writer, not here.
- **No reprojection** — coordinates are written in the model's own CRS; the bounding volume ``box`` is
  the model's axis-aligned extent in that frame.

Only the standard library (``json`` + ``struct``) is used.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import json
import math
import struct
from pathlib import Path
from typing import Any

from pyramids_citymodels import _implicit_tiling as implicit
from pyramids_citymodels.ir import CityModel, CityObject, iter_surfaces
from pyramids_citymodels.writers._geometry import open_ring, xyz

__all__ = ["write_3dtiles"]

#: glTF / glb container constants (little-endian): the ``"glTF"`` magic and the JSON / BIN chunk tags.
_GLB_MAGIC = 0x46546C67
_GLB_CHUNK_JSON = 0x4E4F534A
_GLB_CHUNK_BIN = 0x004E4942
#: glTF accessor componentTypes: float32 positions, uint32 batch ids / indices.
_FLOAT, _UINT = 5126, 5125
#: glTF bufferView targets (ARRAY_BUFFER / ELEMENT_ARRAY_BUFFER) and the TRIANGLES primitive mode.
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER, _TRIANGLES = 34962, 34963, 4
#: The b3dm header: the 4-byte magic then seven uint32 (version, byteLength, and the four table
#: lengths) — 28 bytes, matching the reader's ``_B3DM_HEADER``.
_B3DM_HEADER = struct.Struct("<4sIIIIII")
#: The default content file name written beside ``tileset.json``.
_CONTENT_NAME = "content.b3dm"


def _exterior_rings(
    city_object: CityObject,
) -> list[list[tuple[float, float, float]]]:
    """A city object's surfaces' exterior rings (opened, as xyz triples, each with ≥3 vertices)."""
    rings: list[list[tuple[float, float, float]]] = []
    for geometry in city_object.geometries:
        for surface, _value in iter_surfaces(
            geometry.boundaries, geometry.semantic_values
        ):
            if not surface:
                continue
            ring = [xyz(coordinate) for coordinate in open_ring(surface[0])]
            if len(ring) >= 3:
                rings.append(ring)
    return rings


def _object_rings(
    model: CityModel,
) -> list[tuple[CityObject, list[list[tuple[float, float, float]]]]]:
    """Each city object paired with its surfaces' exterior rings (opened, ≥3 vertices, as xyz).

    An object with no areal surface is dropped, so it contributes no batch — the returned order is
    the batch order (batch id = position in this list).
    """
    contributing: list[tuple[CityObject, list[list[tuple[float, float, float]]]]] = []
    for city_object in model.objects.values():
        rings = _exterior_rings(city_object)
        if rings:
            contributing.append((city_object, rings))
    return contributing


def _bounds(
    rings: list[list[tuple[float, float, float]]],
) -> tuple[list[float], list[float]]:
    """The ``(minimum, maximum)`` per-axis corners over every vertex of every ring."""
    minima = [
        min(ring[v][axis] for ring in rings for v in range(len(ring)))
        for axis in range(3)
    ]
    maxima = [
        max(ring[v][axis] for ring in rings for v in range(len(ring)))
        for axis in range(3)
    ]
    return minima, maxima


def _bounding_box(minima: list[float], maxima: list[float]) -> list[float]:
    """A 3D-Tiles ``box`` bounding volume (centre + three half-axis vectors) for an AABB."""
    center = [(minima[axis] + maxima[axis]) / 2.0 for axis in range(3)]
    half = [(maxima[axis] - minima[axis]) / 2.0 for axis in range(3)]
    return [
        center[0],
        center[1],
        center[2],
        half[0],
        0.0,
        0.0,
        0.0,
        half[1],
        0.0,
        0.0,
        0.0,
        half[2],
    ]


def _batch_table(
    contributing: list[tuple[CityObject, list[Any]]],
) -> dict[str, list[Any]]:
    """A batch table: ``id`` and ``type`` columns plus one column per attribute key (``None``-filled).

    Each column is a list indexed by batch id (the object's position), so the reader reads back each
    batch's ``id``/``type`` and attributes. A key absent from one object is ``None`` in that row.
    """
    keys: list[str] = []
    for city_object, _rings in contributing:
        for key in city_object.attributes:
            if key not in keys:
                keys.append(key)
    table: dict[str, list[Any]] = {
        "id": [city_object.id for city_object, _rings in contributing],
        "type": [city_object.type for city_object, _rings in contributing],
    }
    for key in keys:
        if key in ("id", "type"):
            continue  # an attribute named id/type must not clobber the identity columns
        table[key] = [
            city_object.attributes.get(key) for city_object, _rings in contributing
        ]
    return table


def _mesh_buffers(
    contributing: list[tuple[CityObject, list[list[tuple[float, float, float]]]]],
    translate: list[float],
) -> tuple[bytes, bytes, bytes, int, int]:
    """Build the position / batch-id / index binary blobs for the batched mesh.

    Every ring is fan-triangulated; each vertex is shifted by ``translate`` (for float32 precision)
    and tagged with its object's batch id. Vertices are not de-duplicated, so a batch id is
    unambiguous per vertex.

    Returns:
        ``(position_blob, batchid_blob, index_blob, vertex_count, triangle_count)``.
    """
    positions: list[tuple[float, float, float]] = []
    batch_ids: list[int] = []
    triangles: list[tuple[int, int, int]] = []
    for batch, (_city_object, rings) in enumerate(contributing):
        for ring in rings:
            base = len(positions)
            for coordinate in ring:
                positions.append(
                    (
                        coordinate[0] - translate[0],
                        coordinate[1] - translate[1],
                        coordinate[2] - translate[2],
                    )
                )
                batch_ids.append(batch)
            for apex in range(1, len(ring) - 1):  # fan from the ring's first vertex
                triangles.append((base, base + apex, base + apex + 1))
    position_blob = b"".join(struct.pack("<3f", *position) for position in positions)
    batchid_blob = b"".join(struct.pack("<I", batch) for batch in batch_ids)
    index_blob = b"".join(
        struct.pack("<I", index) for triangle in triangles for index in triangle
    )
    return position_blob, batchid_blob, index_blob, len(positions), len(triangles)


def _glb_document(
    position_blob: bytes,
    batchid_blob: bytes,
    index_blob: bytes,
    vertex_count: int,
    triangle_count: int,
    translate: list[float],
) -> dict[str, Any]:
    """The glTF JSON for the batched mesh (one buffer: positions, then batch ids, then indices)."""
    stored = [
        struct.unpack_from("<3f", position_blob, vertex * 12)
        for vertex in range(vertex_count)
    ]
    minima = [min(point[axis] for point in stored) for axis in range(3)]
    maxima = [max(point[axis] for point in stored) for axis in range(3)]
    batchid_offset = len(position_blob)
    index_offset = batchid_offset + len(batchid_blob)
    return {
        "asset": {"version": "2.0", "generator": "pyramids-citymodels"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": translate}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "_BATCHID": 1},
                        "indices": 2,
                        "mode": _TRIANGLES,
                    }
                ]
            }
        ],
        "buffers": [{"byteLength": index_offset + len(index_blob)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(position_blob),
                "target": _ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": batchid_offset,
                "byteLength": len(batchid_blob),
                "target": _ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": len(index_blob),
                "target": _ELEMENT_ARRAY_BUFFER,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _FLOAT,
                "count": vertex_count,
                "type": "VEC3",
                "min": minima,
                "max": maxima,
            },
            {
                "bufferView": 1,
                "componentType": _UINT,
                "count": vertex_count,
                "type": "SCALAR",
            },
            {
                "bufferView": 2,
                "componentType": _UINT,
                "count": triangle_count * 3,
                "type": "SCALAR",
            },
        ],
    }


def _pad(blob: bytes, boundary: int, fill: bytes) -> bytes:
    """Pad ``blob`` with ``fill`` up to the next ``boundary``-byte multiple."""
    return _pad_to(blob, 0, boundary, fill)


def _pad_to(blob: bytes, start: int, boundary: int, fill: bytes) -> bytes:
    """Pad ``blob`` with ``fill`` so that ``start + len(blob)`` lands on a ``boundary`` multiple.

    ``start`` is the byte offset at which ``blob`` begins in the surrounding container, so a section
    that follows a header of non-multiple size (e.g. the b3dm's 28-byte header) can still be aligned.
    """
    remainder = (start + len(blob)) % boundary
    return blob if remainder == 0 else blob + fill * (boundary - remainder)


def _encode_glb(document: dict[str, Any], buffer: bytes) -> bytes:
    """Pack a glTF document and its single binary buffer into a glTF 2.0 ``glb`` container.

    Both chunks are 4-aligned (the glTF requirement): the BIN chunk is zero-padded only to 4, so it
    exceeds ``buffers[0].byteLength`` by at most 3 bytes (glTF caps that overhang at 3). The whole
    glb is then 8-aligned by extending the **JSON** chunk with trailing spaces — JSON-chunk padding is
    unconstrained by any declared length — so the b3dm can start *and* end the embedded glb on 8-byte
    boundaries without over-padding the BIN chunk past spec.
    """
    json_chunk = _pad(
        json.dumps(document, separators=(",", ":")).encode("utf-8"), 4, b" "
    )
    binary_chunk = _pad(
        buffer, 4, b"\x00"
    )  # <= 3 bytes over buffers[0].byteLength (glTF-legal)
    # 12 header + 8 json-chunk-header + json + 8 bin-chunk-header + bin. Everything but the chunk
    # bodies sums to 28 (4 mod 8) and both bodies are 4-aligned, so any misalignment is exactly 4;
    # add one 4-space group to the JSON chunk (keeps it 4-aligned) to reach 8-alignment.
    if (12 + 8 + len(json_chunk) + 8 + len(binary_chunk)) % 8 != 0:
        json_chunk += b" " * 4
    total = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    out = struct.pack("<III", _GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(json_chunk), _GLB_CHUNK_JSON) + json_chunk
    out += struct.pack("<II", len(binary_chunk), _GLB_CHUNK_BIN) + binary_chunk
    return out


def _encode_b3dm(glb: bytes, batch_length: int, batch_table: dict[str, Any]) -> bytes:
    """Wrap a ``glb`` in a ``b3dm`` tile with a ``BATCH_LENGTH`` feature table and a batch table.

    The 28-byte b3dm header is itself ``4 (mod 8)``, so the feature-table and batch-table JSON are
    padded **accounting for that offset** — ``28 + len(feature_table)`` and
    ``28 + len(feature_table) + len(batch_table)`` are each driven to an 8-byte boundary — and
    ``_encode_glb`` already made the glb's length a multiple of 8. The embedded glb therefore both
    starts and ends on an 8-byte boundary and the whole tile length is a multiple of 8, as the b3dm
    spec requires (and the reader, which keys off the declared lengths, is unaffected).
    """
    header_size = _B3DM_HEADER.size
    feature_table = _pad_to(
        json.dumps({"BATCH_LENGTH": batch_length}).encode("utf-8"), header_size, 8, b" "
    )
    batch_table_json = (
        _pad_to(
            json.dumps(batch_table).encode("utf-8"),
            header_size + len(feature_table),
            8,
            b" ",
        )
        if batch_table
        else b""
    )
    total = header_size + len(feature_table) + len(batch_table_json) + len(glb)
    header = _B3DM_HEADER.pack(
        b"b3dm", 1, total, len(feature_table), 0, len(batch_table_json), 0
    )
    return header + feature_table + batch_table_json + glb


def _b3dm_from_rings(
    contributing: list[tuple[CityObject, list[list[tuple[float, float, float]]]]],
) -> bytes | None:
    """Encode an already-gathered ``contributing`` list into ``b3dm`` bytes (``None`` if empty)."""
    if not contributing:
        return None
    all_rings = [ring for _city_object, rings in contributing for ring in rings]
    minima, _maxima = _bounds(all_rings)
    position_blob, batchid_blob, index_blob, vertices, tris = _mesh_buffers(
        contributing, minima
    )
    document = _glb_document(
        position_blob, batchid_blob, index_blob, vertices, tris, minima
    )
    glb = _encode_glb(document, position_blob + batchid_blob + index_blob)
    return _encode_b3dm(glb, len(contributing), _batch_table(contributing))


def _tileset_from_rings(
    contributing: list[tuple[CityObject, list[list[tuple[float, float, float]]]]],
    content_uri: str | None,
) -> dict[str, Any]:
    """Build the ``tileset.json`` document from an already-gathered ``contributing`` list."""
    if contributing:
        all_rings = [ring for _city_object, rings in contributing for ring in rings]
        minima, maxima = _bounds(all_rings)
        box = _bounding_box(minima, maxima)
        error = math.sqrt(sum((maxima[axis] - minima[axis]) ** 2 for axis in range(3)))
    else:
        box = _bounding_box([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        error = 0.0
    root: dict[str, Any] = {
        "boundingVolume": {"box": box},
        # A single leaf tile is the finest LOD, so its own geometricError is 0 (no further
        # refinement); the tileset-level geometricError above is the model's diagonal.
        "geometricError": 0.0,
        "refine": "ADD",
    }
    if content_uri is not None:
        root["content"] = {"uri": content_uri}
    return {"asset": dict(_ASSET), "geometricError": error, "root": root}


#: The tileset ``asset`` block (``gltfUpAxis="Z"`` because the IR is z-up).
_ASSET = {"version": "1.0", "gltfUpAxis": "Z", "generator": "pyramids-citymodels"}

_Contributing = list[tuple[CityObject, list[list[tuple[float, float, float]]]]]


def _centroid_xy(rings: list[list[tuple[float, float, float]]]) -> tuple[float, float]:
    """The XY centroid (mean of every ring vertex) used to place an object in a quadrant."""
    xs = [point[0] for ring in rings for point in ring]
    ys = [point[1] for ring in rings for point in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _split_xy(
    contributing: _Contributing, minima: list[float], maxima: list[float]
) -> list[_Contributing]:
    """Partition objects into up to four XY quadrants around the group's centre (non-empty buckets)."""
    mid_x = (minima[0] + maxima[0]) / 2
    mid_y = (minima[1] + maxima[1]) / 2
    buckets: dict[tuple[bool, bool], _Contributing] = {}
    for city_object, rings in contributing:
        cx, cy = _centroid_xy(rings)
        buckets.setdefault((cx >= mid_x, cy >= mid_y), []).append((city_object, rings))
    return list(buckets.values())


def _build_tile(
    contributing: _Contributing, target: Path, counter: list[int], max_per_tile: int
) -> dict[str, Any]:
    """Recursively build a tileset node for ``contributing``, writing one ``b3dm`` per leaf.

    A group larger than ``max_per_tile`` is split into XY quadrants (children); a group at or below
    it — or one that cannot be split further (all objects in one quadrant) — is a leaf with its own
    ``b3dm``. ``counter`` numbers the leaf content files across the whole tree.
    """
    all_rings = [ring for _city_object, rings in contributing for ring in rings]
    minima, maxima = _bounds(all_rings)
    diagonal = math.sqrt(sum((maxima[axis] - minima[axis]) ** 2 for axis in range(3)))
    node: dict[str, Any] = {
        "boundingVolume": {"box": _bounding_box(minima, maxima)},
        "refine": "ADD",
    }
    quadrants = (
        _split_xy(contributing, minima, maxima)
        if len(contributing) > max_per_tile
        else [contributing]
    )
    if len(quadrants) > 1:  # genuinely subdivided
        node["geometricError"] = diagonal
        node["children"] = [
            _build_tile(quadrant, target, counter, max_per_tile)
            for quadrant in quadrants
        ]
    else:  # leaf (the finest LOD → geometricError 0)
        node["geometricError"] = 0.0
        b3dm = _b3dm_from_rings(contributing)
        if b3dm is not None:
            name = f"content_{counter[0]}.b3dm"
            counter[0] += 1
            (target / name).write_bytes(b3dm)
            node["content"] = {"uri": name}
    return node


def build_b3dm(model: CityModel) -> bytes | None:
    """Build the batched ``b3dm`` tile bytes for ``model``, or ``None`` when it has no areal geometry.

    A standalone entry point that walks the model itself; :func:`write_3dtiles` is the efficient path
    when both the tile and the tileset are needed (it gathers the geometry once and feeds both).

    Args:
        model: The city model to encode.

    Returns:
        The ``b3dm`` bytes, or ``None`` when no object contributes an areal surface (then the tileset
        has no content).
    """
    return _b3dm_from_rings(_object_rings(model))


def build_tileset(model: CityModel, content_uri: str | None) -> dict[str, Any]:
    """Build the ``tileset.json`` document for ``model``.

    A standalone entry point that walks the model itself; :func:`write_3dtiles` is the efficient path
    when both the tileset and the tile are needed (it gathers the geometry once and feeds both).

    Args:
        model: The city model being written.
        content_uri: The ``b3dm`` file name the root tile points at, or ``None`` for an empty model
            (a valid tileset whose root has no content).

    Returns:
        The tileset document.
    """
    return _tileset_from_rings(_object_rings(model), content_uri)


#: Cap on a single subtree's depth (this writer emits one subtree per tileset). A quadtree at 12
#: levels already addresses 2**11 cells per axis; deeper would need multi-subtree chaining (a follow-up).
_MAX_IMPLICIT_LEVELS = 12


def _centroid(
    rings: list[list[tuple[float, float, float]]], dimension: int
) -> tuple[float, ...]:
    """The mean of every ring vertex over the first ``dimension`` axes (XY for a quadtree, XYZ octree)."""
    sums = [0.0] * dimension
    count = 0
    for ring in rings:
        for point in ring:
            for axis in range(dimension):
                sums[axis] += point[axis]
            count += 1
    return tuple(total / count for total in sums)


def _choose_levels(count: int, max_per_tile: int, children: int) -> int:
    """The number of implicit levels: enough that the deepest grid has ~one cell per ``max_per_tile``.

    The deepest level ``L-1`` has ``children ** (L-1)`` cells; pick the smallest ``L`` whose cell count
    covers ``ceil(count / max_per_tile)`` (capped at :data:`_MAX_IMPLICIT_LEVELS`). Objects cluster, so
    a cell may still exceed the target — the cap keeps one subtree's bitstream bounded.
    """
    if count <= max_per_tile:
        return 1
    cells_needed = math.ceil(count / max_per_tile)
    levels = 1
    while children ** (levels - 1) < cells_needed and levels < _MAX_IMPLICIT_LEVELS:
        levels += 1
    return levels


def _cell_of(
    centroid: tuple[float, ...],
    minima: list[float],
    maxima: list[float],
    cells_per_axis: int,
    dimension: int,
) -> tuple[int, ...]:
    """The ``(x, y[, z])`` grid cell a centroid falls in, over a ``cells_per_axis`` uniform grid."""
    cell = []
    for axis in range(dimension):
        extent = maxima[axis] - minima[axis]
        if extent <= 0:
            cell.append(0)
            continue
        index = int((centroid[axis] - minima[axis]) / extent * cells_per_axis)
        cell.append(min(max(index, 0), cells_per_axis - 1))
    return tuple(cell)


def _write_implicit(
    contributing: _Contributing, target: Path, scheme: str, max_per_tile: int
) -> dict[str, Any]:
    """Write a 3D Tiles 1.1 **implicit** tileset: templated ``b3dm`` content + one ``.subtree`` file.

    Objects are binned by centroid into a uniform quadtree/octree grid at the chosen depth; each
    occupied deepest-level cell gets its own ``content/{level}/{x}/{y}[/{z}].b3dm``. One subtree file
    carries the availability (which tiles exist, which have content); the root ``tileset.json`` holds
    only the root box + the ``implicitTiling`` rule, so the whole tree is described, not enumerated.
    """
    children = implicit.children_per_node(scheme)
    dimension = implicit.dimensions(scheme)
    scheme_name = "OCTREE" if dimension == 3 else "QUADTREE"
    all_rings = [ring for _city_object, rings in contributing for ring in rings]
    minima, maxima = _bounds(all_rings)
    diagonal = math.sqrt(sum((maxima[axis] - minima[axis]) ** 2 for axis in range(3)))
    levels = _choose_levels(len(contributing), max_per_tile, children)
    deepest = levels - 1
    cells_per_axis = 2**deepest

    bins: dict[tuple[int, ...], _Contributing] = {}
    for city_object, rings in contributing:
        cell = _cell_of(
            _centroid(rings, dimension), minima, maxima, cells_per_axis, dimension
        )
        bins.setdefault(cell, []).append((city_object, rings))

    axes_suffix = "/{z}" if dimension == 3 else ""
    content_template = "content/{level}/{x}/{y}" + axes_suffix + ".b3dm"
    subtree_template = "subtrees/{level}/{x}/{y}" + axes_suffix + ".subtree"
    node_total = implicit.node_count(levels, children)
    tile_availability = [False] * node_total
    content_availability = [False] * node_total
    for cell, cell_contributing in bins.items():
        tile = _b3dm_from_rings(cell_contributing)
        if tile is None:
            continue
        content_availability[
            implicit.level_offset(deepest, children) + implicit.morton(cell)
        ] = True
        for level in range(levels):  # the cell and every ancestor is an existing tile
            ancestor = tuple(coord >> (deepest - level) for coord in cell)
            tile_availability[
                implicit.level_offset(level, children) + implicit.morton(ancestor)
            ] = True
        content_path = target / implicit.fill_template(content_template, deepest, cell)
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_bytes(tile)

    root_cell = (0, 0, 0) if dimension == 3 else (0, 0)
    subtree_path = target / implicit.fill_template(subtree_template, 0, root_cell)
    subtree_path.parent.mkdir(parents=True, exist_ok=True)
    subtree_path.write_bytes(
        implicit.encode_subtree(
            tile_availability,
            content_availability,
            [False] * (children**levels),  # one subtree: no child subtrees
        )
    )
    return {
        "asset": {
            "version": "1.1",
            "gltfUpAxis": "Z",
            "generator": "pyramids-citymodels",
        },
        "geometricError": diagonal,
        "root": {
            "boundingVolume": {"box": _bounding_box(minima, maxima)},
            "refine": "REPLACE",
            "geometricError": diagonal,
            "implicitTiling": {
                "subdivisionScheme": scheme_name,
                "subtreeLevels": levels,
                "availableLevels": levels,
                "subtrees": {"uri": subtree_template},
            },
            "content": {"uri": content_template},
        },
    }


def write_3dtiles(
    model: CityModel,
    directory: Any,
    max_objects_per_tile: int | None = None,
    implicit_tiling: bool = False,
    scheme: str = "quadtree",
) -> dict[str, Any]:
    """Write ``model`` as a 3D Tiles tileset into ``directory``.

    Three modes, in order of scale:

    - **Single tile** (default): ``tileset.json`` + one ``content.b3dm``.
    - **Explicit multi-tile** (``max_objects_per_tile=N``): objects are partitioned by footprint into
      an **XY quadtree**, writing one ``content_<n>.b3dm`` per leaf and a nested ``tileset.json`` of
      bounding-volume tiles — the whole tree enumerated in the JSON.
    - **Implicit tiling** (``implicit_tiling=True``): 3D Tiles 1.1 implicit tiling — objects are binned
      into a uniform ``scheme`` (``"quadtree"`` XY or ``"octree"`` XYZ) grid, each occupied cell gets a
      templated ``content/{level}/{x}/{y}[/{z}].b3dm``, and a compact binary ``.subtree`` file carries
      the tile/content availability. The root ``tileset.json`` holds only the root box + the
      ``implicitTiling`` rule, so a million-tile model is *described*, not listed. ``scheme`` is
      ignored unless ``implicit_tiling`` is set; ``max_objects_per_tile`` (default 64 here) sets the
      target objects per cell.

    Geometry is passed through per tile (no per-tile LOD/simplification), tile content stays ``b3dm``,
    and the output round-trips through this package's own 3D-Tiles reader in every mode.

    Args:
        model: The city model to serialize.
        directory: The output directory (created if missing); 3D Tiles is a multi-file format, so a
            directory — not a single file — is required.
        max_objects_per_tile: Target objects per tile for the multi-tile / implicit modes. ``None``
            (default) with ``implicit_tiling=False`` writes a single tile; it defaults to 64 for
            implicit tiling.
        implicit_tiling: Write a 3D Tiles 1.1 implicit tileset (templated content + ``.subtree``) rather
            than an explicit tree.
        scheme: ``"quadtree"`` (XY, default) or ``"octree"`` (XYZ) subdivision — implicit tiling only.

    Returns:
        The written ``tileset.json`` document.

    Note:
        The round-trip contract is via the **tileset** (which declares ``gltfUpAxis="Z"``). Reading a
        ``b3dm`` on its own, outside the tileset, applies the 3D-Tiles default y-up→z-up rotation and
        rotates the footprints — read back through ``tileset.json``, not a bare tile.

    Examples:
        - A one-building model writes a single-tile tileset and a b3dm into the directory:
            ```python
            >>> import tempfile
            >>> from pathlib import Path
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> out = Path(tempfile.mkdtemp())
            >>> tileset = write_3dtiles(model, out)
            >>> tileset["root"]["content"]["uri"]
            'content.b3dm'
            >>> (out / "tileset.json").exists() and (out / "content.b3dm").exists()
            True

            ```
        - Implicit tiling writes a 3D Tiles 1.1 tileset — templated content + a `.subtree` file:
            ```python
            >>> import tempfile
            >>> from pathlib import Path
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> out = Path(tempfile.mkdtemp())
            >>> tileset = write_3dtiles(model, out, implicit_tiling=True)
            >>> tileset["root"]["implicitTiling"]["subdivisionScheme"]
            'QUADTREE'
            >>> tileset["root"]["content"]["uri"]
            'content/{level}/{x}/{y}.b3dm'
            >>> (out / "subtrees" / "0" / "0" / "0.subtree").exists()
            True

            ```
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    # Gather the model's geometry once and feed both builders, rather than walking it twice.
    contributing = _object_rings(model)
    if implicit_tiling and contributing:
        tileset = _write_implicit(
            contributing, target, scheme, max_objects_per_tile or 64
        )
    elif max_objects_per_tile is not None and len(contributing) > max_objects_per_tile:
        root = _build_tile(contributing, target, [0], max_objects_per_tile)
        tileset = {
            "asset": dict(_ASSET),
            "geometricError": root["geometricError"],
            "root": root,
        }
    else:
        b3dm = _b3dm_from_rings(contributing)
        content_uri = _CONTENT_NAME if b3dm is not None else None
        tileset = _tileset_from_rings(contributing, content_uri)
        if b3dm is not None:
            (target / _CONTENT_NAME).write_bytes(b3dm)
    (target / "tileset.json").write_text(
        json.dumps(tileset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tileset
