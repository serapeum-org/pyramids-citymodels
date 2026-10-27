"""Parse Cesium 3D Tiles into the shared IR (issue #16).

3D Tiles is a different animal from CityJSON/CityGML: a ``tileset.json`` is a tree of bounding
volumes whose leaves point at **binary** content — ``b3dm`` (Batched 3D Model, a glTF mesh with a
per-feature *batch table* of attributes), ``pnts`` (point cloud) and bare ``glTF``/``glb``. This is
why the package is ``citymodels`` and not ``citygml``.

What maps cleanly onto a ``FeatureCollection`` is a building's **footprint plus its batch-table
attributes**; the full textured mesh is for a renderer (Digital-Earth's 3-D layer), not a flat
GeoDataFrame (issue #16, scoping caveats). So this reader walks the tileset tree, decodes each
``b3dm`` tile's glTF mesh, groups the vertices by ``_BATCHID`` into one feature per batch id, and
emits each feature's footprint (the 2-D convex hull of its vertices) as an IR ``CityObject`` with a
``MultiSurface`` footprint geometry and the batch-table row as its attributes. The flattener
(:mod:`pyramids_citymodels.flatten`) turns that into footprint rows like any other city model.

Deliberate scope and limitations of this first reader (all documented, none silent):

- **Footprints, not meshes.** The emitted geometry is the convex-hull footprint of each feature's
  vertices projected to XY — an approximation, not the true concave outline. Full mesh geometry is
  not surfaced here; a renderer consumes the ``b3dm`` directly.
- **Vertices are in the tileset's stored coordinate system.** Tile and node transforms are composed
  and applied, ``RTC_CENTER`` is added, and the glTF up-axis convention is honoured — glTF content
  is **y-up** by convention, so the tileset's ``asset.gltfUpAxis`` (default ``"Y"``) selects the
  y-up→z-up (or x-up→z-up) rotation Cesium applies before placing a tile, and this reader applies it
  too, so a footprint is the true plan view, not a side elevation. What it does **not** do is project
  the resulting (typically Earth-centred Earth-fixed, ECEF, metre) coordinates to a geographic or
  projected CRS; the model's ``crs`` is therefore left ``None``.
- **Uncompressed glTF only.** Draco (``KHR_draco_mesh_compression``) and meshopt payloads are
  rejected with a clear :class:`~pyramids_citymodels.errors.UnsupportedGeometryError`; decoding them
  needs an optional backend, a documented follow-up (issue #12's extra convention).
- **``pnts`` is routed out, not read here.** Point-cloud tiles belong to the ``pointclouds`` package
  (serapeum-org/pointclouds#10); a ``pnts`` tile is skipped with a warning rather than mis-read as a
  mesh.

Only the standard library plus ``numpy`` is used (``json`` + ``struct`` for the containers, ``numpy``
for the vertex arithmetic and the hull), so importing this module pulls in no heavy backend — the
same import-light contract the CityJSON and CityGML readers keep.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import json
import os
import struct
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pyramids_citymodels.errors import UnsupportedGeometryError
from pyramids_citymodels.ir import CityModel, CityObject, Geometry

__all__ = ["looks_like_3dtiles", "parse_3dtiles"]

#: Little-endian magic for a binary glTF container (``"glTF"``).
_GLB_MAGIC = 0x46546C67
#: glTF chunk-type tags (``"JSON"`` and ``"BIN\0"``), little-endian uint32.
_GLB_CHUNK_JSON = 0x4E4F534A
_GLB_CHUNK_BIN = 0x004E4942
#: The b3dm header is seven uint32 after the 4-byte magic — 28 bytes in total.
_B3DM_HEADER = struct.Struct("<4sIIIIII")
#: glTF ``componentType`` codes → numpy dtypes, for the accessors this reader reads.
_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
#: glTF ``type`` → component count.
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_bytes(source: Any) -> bytes | None:
    """Return the raw bytes of a binary tile source, or ``None`` if it is not byte-like.

    Accepts raw ``bytes``/``bytearray``, an open binary file, or a filesystem path. Never raises for
    a path that cannot be read — a sniff must answer, not explode.
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        data = source.read()
        return bytes(data) if isinstance(data, (bytes, bytearray)) else None
    if isinstance(source, (str, os.PathLike)):
        try:
            return Path(source).read_bytes()
        except OSError:
            return None
    return None


def _load_tileset(source: Any) -> tuple[dict[str, Any] | None, Path | None]:
    """Return a decoded ``tileset.json`` document and its base directory, or ``(None, None)``.

    The base directory is where relative ``content.uri`` references resolve; it is ``None`` for an
    in-memory source (a dict or a JSON string), whose content references cannot be followed.
    """
    if isinstance(source, dict):
        return source, None
    if isinstance(source, str) and source.lstrip().startswith("{"):
        try:
            return json.loads(source), None
        except ValueError:
            return None, None
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            # ValueError covers UnicodeDecodeError — a binary tile (b3dm/glb) handed to the
            # tileset sniff is simply not a JSON tileset, not an error.
            return None, None
        try:
            return json.loads(text), path.parent
        except ValueError:
            return None, None
    return None, None


def _is_tileset(document: Any) -> bool:
    """Whether a decoded JSON document is a 3D Tiles tileset (``asset`` + ``root``/``geometricError``)."""
    return (
        isinstance(document, dict)
        and isinstance(document.get("asset"), dict)
        and ("root" in document or "geometricError" in document)
    )


def looks_like_3dtiles(source: Any) -> bool:
    """Return ``True`` when ``source`` is a 3D Tiles tileset or a ``b3dm`` tile. Never raises.

    The driver's ``can_open`` uses this to claim a ``.json`` that is a *tileset* (not CityJSON or
    GeoJSON) and a ``.b3dm``/``.glb`` binary. A tileset is recognised by its ``asset`` object plus a
    ``root`` or ``geometricError``; a binary tile by its ``b3dm`` / ``glTF`` magic.

    Examples:
        - A tileset.json is recognised; a CityJSON document sharing the ``.json`` suffix is not:
            ```python
            >>> looks_like_3dtiles('{"asset": {"version": "1.0"}, "geometricError": 0, "root": {}}')
            True
            >>> looks_like_3dtiles('{"type": "CityJSON", "CityObjects": {}, "vertices": []}')
            False

            ```
        - A non-tileset, non-binary source is declined without raising:
            ```python
            >>> looks_like_3dtiles("not a tileset")
            False

            ```
    """
    document, _ = _load_tileset(source)
    if _is_tileset(document):
        return True
    data = _read_bytes(source)
    if data is None or len(data) < 4:
        return False
    return data[:4] == b"b3dm" or (
        len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == _GLB_MAGIC
    )


def _parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Split a binary glTF (``glb``) into its JSON document and its binary (``BIN``) chunk.

    Args:
        data: The ``glb`` bytes (starting at the ``glTF`` magic).

    Returns:
        The parsed glTF JSON and the binary buffer (empty when the file has no ``BIN`` chunk).

    Raises:
        UnsupportedGeometryError: The data is not a glTF 2.0 ``glb`` container.
    """
    if len(data) < 12 or struct.unpack_from("<I", data, 0)[0] != _GLB_MAGIC:
        raise UnsupportedGeometryError(
            "3D Tiles content is not a glTF binary (glb) payload"
        )
    offset = 12
    gltf_json: dict[str, Any] = {}
    binary = b""
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == _GLB_CHUNK_JSON:
            gltf_json = json.loads(chunk.decode("utf-8"))
        elif chunk_type == _GLB_CHUNK_BIN:
            binary = chunk
    return gltf_json, binary


def _reject_compressed(gltf_json: dict[str, Any]) -> None:
    """Raise if the glTF uses a mesh-compression extension this reader cannot decode."""
    required = set(gltf_json.get("extensionsRequired", []))
    unsupported = required & {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}
    if unsupported:
        raise UnsupportedGeometryError(
            f"3D Tiles glTF requires mesh-compression extension(s) {sorted(unsupported)}, which "
            f"this reader cannot decode; a compressed-mesh backend is a documented follow-up (#12)"
        )


def _accessor_array(
    gltf: dict[str, Any], binary: bytes, accessor_index: int
) -> np.ndarray:
    """Read one glTF accessor into an ``(count, components)`` numpy array.

    Honours the accessor's and buffer view's byte offsets and an optional interleaving
    ``byteStride``; assumes the single embedded ``glb`` buffer (``buffers[0]`` with no ``uri``),
    which is what a ``b3dm`` payload always carries.
    """
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPE[accessor["componentType"]]).newbyteorder("<")
    components = _TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or components * dtype.itemsize
    rows = []
    for row in range(count):
        base = start + row * stride
        rows.append(np.frombuffer(binary, dtype=dtype, count=components, offset=base))
    return np.array(rows, dtype=np.float64) if count else np.empty((0, components))


def _node_matrix(node: dict[str, Any]) -> np.ndarray:
    """Return a node's local 4x4 transform, from an explicit ``matrix`` or its TRS components."""
    if "matrix" in node:
        # glTF stores a matrix column-major; reshape as such then transpose to row-major.
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4)
    translation = node.get("translation")
    rotation = node.get("rotation")
    scale = node.get("scale")
    if scale is not None:
        matrix = matrix @ np.diag([*scale, 1.0])
    if rotation is not None:
        matrix = _quaternion_matrix(rotation) @ matrix
    if translation is not None:
        shift = np.eye(4)
        shift[:3, 3] = translation
        matrix = shift @ matrix
    return matrix


def _quaternion_matrix(quaternion: list[float]) -> np.ndarray:
    """Return the 4x4 rotation matrix for a glTF ``[x, y, z, w]`` unit quaternion."""
    x, y, z, w = quaternion
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix


def _up_axis_to_zup(axis: Any) -> np.ndarray:
    """Return the 4x4 rotation that brings a glTF tile's up-axis to the z-up tile frame.

    glTF content is y-up by convention; 3D Tiles places it with the rotation named by the tileset's
    ``asset.gltfUpAxis`` (``"Y"`` by default). Mirrors Cesium's ``Axis.*_UP_TO_Z_UP``:

    - ``"Z"`` → identity (content is already z-up);
    - ``"Y"`` → +90° about X, mapping ``(x, y, z)`` to ``(x, -z, y)``;
    - ``"X"`` → -90° about Y, mapping ``(x, y, z)`` to ``(-z, y, x)``.

    An unrecognised value falls back to the ``"Y"`` default rather than raising.
    """
    matrix = np.eye(4)
    label = str(axis).upper() if axis is not None else "Y"
    if label == "Z":
        return matrix
    if label == "X":
        matrix[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
        return matrix
    matrix[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    return matrix


def _mesh_vertices(
    gltf: dict[str, Any], binary: bytes
) -> tuple[np.ndarray, np.ndarray | None]:
    """Collect every mesh vertex (with node transforms applied) and its ``_BATCHID``, if any.

    Returns:
        ``(points, batch_ids)`` where ``points`` is an ``(n, 3)`` array in the glTF's root frame and
        ``batch_ids`` is an ``(n,)`` integer array, or ``None`` when no primitive declares
        ``_BATCHID`` (the whole tile is then one feature).
    """
    mesh_transforms = _mesh_world_transforms(gltf)
    all_points: list[np.ndarray] = []
    all_batch: list[np.ndarray] = []
    saw_batch = False
    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        transform = mesh_transforms.get(mesh_index, np.eye(4))
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "POSITION" not in attributes:
                continue
            points = _accessor_array(gltf, binary, attributes["POSITION"])
            homogeneous = np.column_stack([points, np.ones(len(points))])
            world = (transform @ homogeneous.T).T[:, :3]
            all_points.append(world)
            if "_BATCHID" in attributes:
                saw_batch = True
                batch = _accessor_array(gltf, binary, attributes["_BATCHID"]).reshape(
                    -1
                )
            else:
                batch = np.zeros(len(points))
            all_batch.append(batch)
    if not all_points:
        return np.empty((0, 3)), None
    points = np.vstack(all_points)
    batch_ids = np.concatenate(all_batch).astype(int) if saw_batch else None
    return points, batch_ids


def _mesh_world_transforms(gltf: dict[str, Any]) -> dict[int, np.ndarray]:
    """Map each mesh index to the composed world transform of the first node that instances it.

    Walks the default scene's node tree, composing parent→child transforms, so a glTF that places
    its mesh under a y-up→z-up root node (as a ``b3dm`` conventionally does) has that rotation
    applied to the vertices.
    """
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    scene_index = gltf.get("scene", 0)
    roots = (
        scenes[scene_index].get("nodes", [])
        if scenes and 0 <= scene_index < len(scenes)
        else list(range(len(nodes)))
    )
    transforms: dict[int, np.ndarray] = {}

    def walk(node_index: int, parent: np.ndarray) -> None:
        node = nodes[node_index]
        world = parent @ _node_matrix(node)
        if "mesh" in node and node["mesh"] not in transforms:
            transforms[node["mesh"]] = world
        for child in node.get("children", []):
            walk(child, world)

    for root in roots:
        if 0 <= root < len(nodes):
            walk(root, np.eye(4))
    return transforms


def _convex_hull_ring(points_xy: np.ndarray) -> list[tuple[float, float]] | None:
    """Return the closed convex-hull ring of 2-D points (Andrew's monotone chain), or ``None``.

    ``None`` when the points are degenerate (fewer than three distinct, or all collinear), so the
    caller skips a feature that has no areal footprint rather than emitting a zero-area polygon.
    """
    unique = np.unique(np.round(points_xy, 9), axis=0)
    if len(unique) < 3:
        return None
    order = np.lexsort((unique[:, 1], unique[:, 0]))
    ordered = unique[order]

    def half(chain_points: np.ndarray) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        for point in chain_points:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower = half(ordered)
    upper = half(ordered[::-1])
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None
    ring = [(float(point[0]), float(point[1])) for point in hull]
    ring.append(ring[0])
    return ring


def _cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """2-D cross product of the vectors ``origin→first`` and ``origin→second``."""
    return float(
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def _batch_table_rows(
    batch_table_json: dict[str, Any], length: int
) -> list[dict[str, Any]]:
    """Turn a (JSON-backed) batch table into one attribute dict per batch id.

    Only JSON-array properties are read (the common case for hand-authored and most exported
    tilesets); a binary-backed property (a ``{byteOffset, componentType, type}`` descriptor) is
    skipped rather than mis-decoded, so an unknown binary column never corrupts attributes.
    """
    rows: list[dict[str, Any]] = [{} for _ in range(length)]
    for key, values in batch_table_json.items():
        if key in ("extensions", "extras") or not isinstance(values, list):
            continue
        for index in range(min(length, len(values))):
            rows[index][key] = values[index]
    return rows


def _feature_table(data: bytes, json_length: int, offset: int) -> dict[str, Any]:
    """Decode a feature/batch table's JSON header (the binary body is handled separately)."""
    if json_length == 0:
        return {}
    raw = data[offset : offset + json_length].decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def _parse_b3dm_tile(
    data: bytes, tile_transform: np.ndarray, up_axis: Any = "Y"
) -> list[CityObject]:
    """Decode one ``b3dm`` tile into footprint :class:`CityObject` s.

    Args:
        data: The ``b3dm`` bytes.
        tile_transform: The composed tileset transform to apply to the tile's vertices.
        up_axis: The tileset's ``asset.gltfUpAxis`` (``"Y"`` by default); selects the glTF
            up-axis→z-up rotation applied before the tile transform.

    Returns:
        One :class:`CityObject` per batch id whose footprint is areal, each with its batch-table row
        as attributes; an empty list when the tile has no areal feature.

    Raises:
        UnsupportedGeometryError: The glTF payload is compressed or malformed.
    """
    magic, _version, _length, ft_json, ft_bin, bt_json, bt_bin = (
        _B3DM_HEADER.unpack_from(data, 0)
    )
    if magic != b"b3dm":
        raise UnsupportedGeometryError("tile content is not a b3dm payload")
    offset = _B3DM_HEADER.size
    feature_table = _feature_table(data, ft_json, offset)
    offset += ft_json + ft_bin
    batch_table = _feature_table(data, bt_json, offset)
    offset += bt_json + bt_bin
    gltf, binary = _parse_glb(data[offset:])
    _reject_compressed(gltf)

    points, batch_ids = _mesh_vertices(gltf, binary)
    if len(points) == 0:
        return []
    rtc = feature_table.get("RTC_CENTER")
    if rtc:
        points = points + np.array(rtc, dtype=np.float64)
    # Bring the y-up glTF content into the z-up tile frame (gltfUpAxis), then place the tile.
    placement = tile_transform @ _up_axis_to_zup(up_axis)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    points = (placement @ homogeneous.T).T[:, :3]

    if batch_ids is None:
        groups = {0: np.arange(len(points))}
        batch_length = 1
    else:
        batch_length = int(feature_table.get("BATCH_LENGTH", batch_ids.max() + 1))
        groups = {
            batch: np.where(batch_ids == batch)[0] for batch in range(batch_length)
        }
    rows = _batch_table_rows(batch_table, batch_length)

    objects: list[CityObject] = []
    for batch, indices in groups.items():
        if len(indices) == 0:
            continue
        ring = _convex_hull_ring(points[indices][:, :2])
        if ring is None:
            continue
        attributes = dict(rows[batch]) if batch < len(rows) else {}
        identifier = str(
            attributes.get("id") or attributes.get("name") or f"feature-{batch}"
        )
        objects.append(
            CityObject(
                id=identifier,
                type=str(attributes.get("type", "Building")),
                attributes=attributes,
                geometries=[
                    Geometry(type="MultiSurface", lod=None, boundaries=[[ring]])
                ],
            )
        )
    return objects


def _tile_transform(tile: dict[str, Any], parent: np.ndarray) -> np.ndarray:
    """Compose a tile's own ``transform`` (column-major 4x4) onto its parent's."""
    raw = tile.get("transform")
    if not raw:
        return parent
    return parent @ np.array(raw, dtype=np.float64).reshape(4, 4).T


def _walk_tileset(
    tile: dict[str, Any],
    base: Path | None,
    transform: np.ndarray,
    up_axis: Any,
    objects: dict[str, CityObject],
) -> None:
    """Depth-first walk of a tileset tree, decoding each tile's content into ``objects``."""
    here = _tile_transform(tile, transform)
    content = tile.get("content") or {}
    uri = content.get("uri") or content.get("url")
    if uri and base is not None:
        _decode_content(base / uri, here, up_axis, objects)
    for child in tile.get("children", []):
        _walk_tileset(child, base, here, up_axis, objects)


def _decode_content(
    path: Path, transform: np.ndarray, up_axis: Any, objects: dict[str, CityObject]
) -> None:
    """Decode one referenced content file (a nested tileset, a ``b3dm``, or a routed-out ``pnts``)."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        nested, nested_base = _load_tileset(path)
        if (
            _is_tileset(nested)
            and nested is not None
            and isinstance(nested.get("root"), dict)
        ):
            nested_up = (nested.get("asset") or {}).get("gltfUpAxis", up_axis)
            _walk_tileset(nested["root"], nested_base, transform, nested_up, objects)
        return
    data = _read_bytes(path)
    if data is None or len(data) < 4:
        return
    if data[:4] == b"pnts":
        warnings.warn(
            f"3D Tiles point-cloud tile {path.name!r} is skipped: pnts payloads are read by the "
            f"pointclouds package, not the city-model vector driver",
            UserWarning,
            stacklevel=2,
        )
        return
    for city_object in _parse_b3dm_tile(data, transform, up_axis):
        objects.setdefault(city_object.id, city_object)


def parse_3dtiles(source: Any) -> CityModel:
    """Parse a 3D Tiles tileset or a single ``b3dm`` tile into a :class:`CityModel`.

    A ``tileset.json`` path (or decoded dict/JSON string) is walked from its root, following every
    ``content.uri`` relative to the tileset file and composing the tile transforms; a ``b3dm`` path
    or byte string is decoded on its own. Each batched feature becomes a :class:`CityObject` whose
    geometry is its convex-hull footprint and whose attributes are its batch-table row.

    Args:
        source: A ``tileset.json`` path / dict / JSON string, or a ``b3dm`` path / bytes.

    Returns:
        The parsed model. Its ``crs`` is ``None`` — see the module docstring on coordinate systems.
        An in-memory tileset (dict/string) has no base directory, so its content cannot be followed
        and the model is empty.

    Raises:
        UnsupportedGeometryError: A decoded tile is compressed or not a glTF/b3dm payload.
        TypeError: ``source`` is neither a tileset nor a readable binary tile.

    Examples:
        - An in-memory tileset has no base directory, so its content cannot be followed and the
          model comes back well-formed but empty:
            ```python
            >>> from pyramids_citymodels.readers.threedtiles import parse_3dtiles
            >>> model = parse_3dtiles(
            ...     '{"asset": {"version": "1.0"}, "geometricError": 0, "root": {}}'
            ... )
            >>> model.source_format
            '3DTiles'
            >>> model.crs is None
            True
            >>> model.objects
            {}

            ```
        - A decoded tileset dict is accepted the same way as a JSON string:
            ```python
            >>> from pyramids_citymodels.readers.threedtiles import parse_3dtiles
            >>> model = parse_3dtiles({"asset": {"version": "1.0"}, "root": {}})
            >>> model.source_format
            '3DTiles'

            ```
    """
    objects: dict[str, CityObject] = {}
    document, base = _load_tileset(source)
    if _is_tileset(document) and document is not None:
        up_axis = (document.get("asset") or {}).get("gltfUpAxis", "Y")
        root = document.get("root")
        if isinstance(root, dict):
            _walk_tileset(root, base, np.eye(4), up_axis, objects)
        return CityModel(objects=objects, source_format="3DTiles")

    data = _read_bytes(source)
    if data is not None and len(data) >= 4 and data[:4] == b"b3dm":
        for city_object in _parse_b3dm_tile(data, np.eye(4), "Y"):
            objects.setdefault(city_object.id, city_object)
        return CityModel(objects=objects, source_format="3DTiles")

    raise TypeError(
        f"cannot read 3D Tiles from {type(source).__name__}: expected a tileset.json or a b3dm tile"
    )
