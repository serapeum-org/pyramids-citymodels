"""Serialize the IR to a Cesium 3D Tiles tileset (feature #15 — tiling / packaging).

3D Tiles is the runtime streaming format for massive city models: a ``tileset.json`` tree of bounding
volumes whose leaves point at **binary** ``b3dm`` (Batched 3D Model) content. This writer produces a
single-tile tileset — one ``tileset.json`` plus one ``.b3dm`` — which is the degenerate (but fully
valid) tree every 3D-Tiles client reads; spatial subdivision into many tiles is a documented
follow-up.

The ``b3dm`` embeds a binary glTF (``glb``) built the same way as the :mod:`glTF writer
<pyramids_citymodels.writers.gltf>` — every surface's exterior ring fan-triangulated into one
positions mesh — but with two additions that make it a *batched* model:

- a ``_BATCHID`` vertex attribute assigning every vertex to its source :class:`CityObject` (one batch
  per object), so a client can pick and style buildings individually;
- a **batch table** carrying each object's ``id``, ``type`` and attributes as parallel columns.

So the tileset round-trips through this package's own 3D-Tiles reader back to one footprint
:class:`CityObject` per batch, attributes and all. The tileset declares ``asset.gltfUpAxis = "Z"`` (the
IR is z-up), so a reader applies no up-axis rotation and footprints are the true plan view.

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


def _object_rings(
    model: CityModel,
) -> list[tuple[CityObject, list[list[tuple[float, float, float]]]]]:
    """Each city object paired with its surfaces' exterior rings (opened, ≥3 vertices, as xyz).

    An object with no areal surface is dropped, so it contributes no batch — the returned order is
    the batch order (batch id = position in this list).
    """
    contributing: list[tuple[CityObject, list[list[tuple[float, float, float]]]]] = []
    for city_object in model.objects.values():
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

    The JSON chunk is space-padded to 4 (glTF requires 4-byte chunk alignment); the BIN chunk is
    zero-padded so the **whole glb** length is a multiple of 8, which lets the b3dm start *and* end
    the embedded glb on 8-byte boundaries (the 3D Tiles b3dm requirement).
    """
    json_chunk = _pad(
        json.dumps(document, separators=(",", ":")).encode("utf-8"), 4, b" "
    )
    # Pad the BIN chunk so (12 header + 8 json-chunk-header + json + 8 bin-chunk-header + bin) is a
    # multiple of 8 — i.e. the glb's total declared length is 8-aligned.
    binary_chunk = _pad_to(buffer, 12 + 8 + len(json_chunk) + 8, 8, b"\x00")
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


def build_b3dm(model: CityModel) -> bytes | None:
    """Build the batched ``b3dm`` tile bytes for ``model``, or ``None`` when it has no areal geometry.

    Args:
        model: The city model to encode.

    Returns:
        The ``b3dm`` bytes, or ``None`` when no object contributes an areal surface (then the tileset
        has no content).
    """
    contributing = _object_rings(model)
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


def build_tileset(model: CityModel, content_uri: str | None) -> dict[str, Any]:
    """Build the ``tileset.json`` document for ``model``.

    Args:
        model: The city model being written.
        content_uri: The ``b3dm`` file name the root tile points at, or ``None`` for an empty model
            (a valid tileset whose root has no content).

    Returns:
        The tileset document.
    """
    contributing = _object_rings(model)
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
        "geometricError": 0.0,
        "refine": "ADD",
    }
    if content_uri is not None:
        root["content"] = {"uri": content_uri}
    return {
        "asset": {
            "version": "1.0",
            "gltfUpAxis": "Z",
            "generator": "pyramids-citymodels",
        },
        "geometricError": error,
        "root": root,
    }


def write_3dtiles(model: CityModel, directory: Any) -> dict[str, Any]:
    """Write ``model`` as a single-tile 3D Tiles tileset into ``directory``.

    Writes ``tileset.json`` and (when the model has areal geometry) ``content.b3dm`` into
    ``directory``, creating it if needed. See the module docstring for the format and its limits.

    Args:
        model: The city model to serialize.
        directory: The output directory (created if missing); 3D Tiles is a multi-file format, so a
            directory — not a single file — is required.

    Returns:
        The written ``tileset.json`` document.

    Examples:
        - A one-building model writes a tileset and a b3dm into the directory:
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
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    b3dm = build_b3dm(model)
    content_uri = _CONTENT_NAME if b3dm is not None else None
    tileset = build_tileset(model, content_uri)
    (target / "tileset.json").write_text(
        json.dumps(tileset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if b3dm is not None:
        (target / _CONTENT_NAME).write_bytes(b3dm)
    return tileset
