"""Serialize the IR to glTF 2.0 (feature #14 — export to meshes).

glTF is the runtime-oriented mesh format web and game engines consume. This writer triangulates
every surface's exterior ring (a fan from its first vertex) into one indexed triangle mesh and emits
a single-file ``.gltf``: a JSON document whose one buffer is embedded as a base64 ``data:`` URI, so
the result is self-contained with no sidecar ``.bin``.

Because glTF stores positions as 32-bit floats — which lose precision at the large magnitudes of a
projected CRS (metre eastings in the hundreds of thousands) — every position is shifted by the
model's per-axis minimum and that minimum is written back as the node's ``translation``, so the
rendered mesh sits in the right place while the stored floats stay small and precise.

Deliberate, documented limits (plain geometry, like the OBJ writer):

- **Exterior rings only**, triangulated as a fan — interior rings (holes) are dropped, and a strongly
  non-convex polygon's fan can self-overlap (fan triangulation is not ear-clipping).
- **Positions only by default; opt-in materials/textures** (``materials=True``) — one primitive per
  ``(material, texture)`` with a PBR ``baseColorFactor`` from the CityJSON ``diffuseColor`` and, when a
  surface is textured, a ``TEXCOORD_0`` from the ``vertices-texture`` pool plus a ``baseColorTexture``
  **referenced by image URI** (not embedded, and not fetched). No normals; the full appearance pool
  still round-trips losslessly through the CityJSON writer, not here.
- An empty model (no areal surface) yields a valid glTF with an empty scene.

Only the standard library (``json`` + ``struct`` + ``base64``) is used.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import base64
import json
import struct
from typing import Any

from pyramids_citymodels.ir import CityModel, Geometry, iter_surfaces
from pyramids_citymodels.writers._geometry import open_ring as _open_ring
from pyramids_citymodels.writers._geometry import xyz as _xyz

__all__ = ["write_gltf"]

#: glTF constants: float32 / uint32 component types, ARRAY_BUFFER / ELEMENT_ARRAY_BUFFER targets.
_FLOAT, _UINT = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_TRIANGLES = 4
#: A fully opaque default base colour for a material whose CityJSON entry carries no ``diffuseColor``.
_DEFAULT_BASE_COLOR = [0.8, 0.8, 0.8, 1.0]

#: One exterior ring ready to mesh: its xyz vertices and (when textured) a UV index per vertex.
_Ring = tuple[list[tuple[float, float, float]], "list[int] | None"]
#: The appearance key grouping surfaces into one primitive: ``(material index, texture index)``.
_GroupKey = tuple["int | None", "int | None"]


def _rings(model: CityModel) -> list[list[tuple[float, float, float]]]:
    """Every surface's exterior ring (opened, as xyz triples) with at least three vertices."""
    rings: list[list[tuple[float, float, float]]] = []
    for city_object in model.objects.values():
        for geometry in city_object.geometries:
            for surface, _value in iter_surfaces(
                geometry.boundaries, geometry.semantic_values
            ):
                if not surface:
                    continue
                ring = [_xyz(coordinate) for coordinate in _open_ring(surface[0])]
                if len(ring) >= 3:
                    rings.append(ring)
    return rings


def _empty_gltf() -> dict[str, Any]:
    """A valid, mesh-less glTF 2.0 document (for a model with no areal surface)."""
    return {
        "asset": {"version": "2.0", "generator": "pyramids-citymodels"},
        "scene": 0,
        "scenes": [{"nodes": []}],
    }


def _build_gltf(model: CityModel) -> dict[str, Any]:
    """Build the glTF document dict from the IR (no file I/O)."""
    rings = _rings(model)
    if not rings:
        return _empty_gltf()

    translate = [
        min(ring[vertex][axis] for ring in rings for vertex in range(len(ring)))
        for axis in range(3)
    ]
    pool: dict[tuple[float, float, float], int] = {}
    positions: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def intern(coordinate: tuple[float, float, float]) -> int:
        shifted = (
            coordinate[0] - translate[0],
            coordinate[1] - translate[1],
            coordinate[2] - translate[2],
        )
        index = pool.get(shifted)
        if index is None:
            index = len(positions)
            pool[shifted] = index
            positions.append(shifted)
        return index

    for ring in rings:
        indices = [intern(coordinate) for coordinate in ring]
        for apex in range(1, len(indices) - 1):  # fan from the first vertex
            triangles.append((indices[0], indices[apex], indices[apex + 1]))

    position_blob = b"".join(struct.pack("<3f", *position) for position in positions)
    index_blob = b"".join(
        struct.pack("<I", index) for triangle in triangles for index in triangle
    )
    buffer = (
        position_blob + index_blob
    )  # position_blob is 12-byte aligned, so indices are 4-aligned
    uri = "data:application/octet-stream;base64," + base64.b64encode(buffer).decode(
        "ascii"
    )

    # Declare min/max from the float32 bytes actually stored, not the float64 positions — otherwise
    # a bound that is not float32-exact disagrees with the data and the glTF validator flags
    # ACCESSOR_MIN/MAX_MISMATCH (a loader could frustum-cull a sliver of geometry).
    stored = [
        struct.unpack_from("<3f", position_blob, vertex * 12)
        for vertex in range(len(positions))
    ]
    minima = [min(point[axis] for point in stored) for axis in range(3)]
    maxima = [max(point[axis] for point in stored) for axis in range(3)]
    return {
        "asset": {"version": "2.0", "generator": "pyramids-citymodels"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": translate}],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0}, "indices": 1, "mode": _TRIANGLES}
                ]
            }
        ],
        "buffers": [{"byteLength": len(buffer), "uri": uri}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(position_blob),
                "target": _ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": len(position_blob),
                "byteLength": len(index_blob),
                "target": _ELEMENT_ARRAY_BUFFER,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _FLOAT,
                "count": len(positions),
                "type": "VEC3",
                "min": minima,
                "max": maxima,
            },
            {
                "bufferView": 1,
                "componentType": _UINT,
                "count": len(triangles) * 3,
                "type": "SCALAR",
            },
        ],
    }


def _appearance_values(geometry: Geometry, kind: str, default_theme: Any) -> Any:
    """The ``material`` / ``texture`` ``values`` for a geometry's chosen theme (``None`` if it has none).

    ``kind`` is ``"material"`` or ``"texture"``. The document's ``default-theme-*`` is used when the
    geometry carries it; otherwise the geometry's first theme is taken, so a single-theme model needs no
    configuration. The returned structure mirrors the geometry's boundaries (one leaf per surface), so
    :func:`~pyramids_citymodels.ir.iter_surfaces` pairs it with each surface in lock-step.
    """
    block = (geometry.appearance or {}).get(kind)
    if not block:
        return None
    theme = default_theme if default_theme in block else next(iter(block))
    return (block.get(theme) or {}).get("values")


def _texture_for_ring(
    texture_leaf: Any, ring_length: int, uv_pool_size: int
) -> tuple[int | None, list[int] | None]:
    """A surface's ``(texture index, UV indices)`` from its CityJSON texture leaf, or ``(None, None)``.

    A CityJSON texture value for a surface is ``[[texture_index, uv0, uv1, ...], ...]`` — one list per
    ring, whose head is the texture index and whose tail is a UV index per ring vertex. Only the
    exterior ring is meshed, the UV count must match the ring, and every UV index must be within the
    ``vertices-texture`` pool; otherwise the surface is treated as untextured, so a malformed, partial
    or out-of-range texture never corrupts the mesh (nor raises an ``IndexError``).
    """
    if not isinstance(texture_leaf, list) or not texture_leaf:
        return None, None
    entry = texture_leaf[0]
    if not isinstance(entry, list) or not entry or not isinstance(entry[0], int):
        return None, None
    uv_indices = entry[1:]
    if len(uv_indices) != ring_length or not all(
        isinstance(i, int) for i in uv_indices
    ):
        return None, None
    if not all(0 <= index < uv_pool_size for index in uv_indices):
        return None, None
    return entry[0], uv_indices


def _collect_groups(model: CityModel) -> dict[_GroupKey, list[_Ring]]:
    """Group every exterior ring by its ``(material index, texture index)`` appearance key.

    Each group becomes one glTF primitive sharing a material, so a textured model emits as few
    primitives as it has distinct material/texture combinations. A surface with no material/texture
    lands under ``(None, None)`` — the plain-geometry primitive.
    """
    groups: dict[_GroupKey, list[_Ring]] = {}
    default_mat = model.appearances.get("default-theme-material")
    default_tex = model.appearances.get("default-theme-texture")
    uv_pool_size = len(model.appearances.get("vertices-texture") or [])
    for city_object in model.objects.values():
        for geometry in city_object.geometries:
            material_values = _appearance_values(geometry, "material", default_mat)
            texture_values = _appearance_values(geometry, "texture", default_tex)
            paired = zip(
                iter_surfaces(geometry.boundaries, material_values),
                iter_surfaces(geometry.boundaries, texture_values),
            )
            for (surface, material_leaf), (_surface, texture_leaf) in paired:
                if not surface:
                    continue
                ring = [_xyz(coordinate) for coordinate in _open_ring(surface[0])]
                if len(ring) < 3:
                    continue
                material_index = (
                    material_leaf if isinstance(material_leaf, int) else None
                )
                texture_index, uv_indices = _texture_for_ring(
                    texture_leaf, len(ring), uv_pool_size
                )
                groups.setdefault((material_index, texture_index), []).append(
                    (ring, uv_indices)
                )
    return groups


def _primitive_blobs(
    rings: list[_Ring],
    translate: list[float],
    uv_pool: list[Any],
    textured: bool,
) -> tuple[bytes, bytes, bytes, int, int]:
    """Weld one group's rings into ``(position, texcoord, index)`` blobs + vertex / triangle counts.

    A vertex is the ``(shifted position, UV)`` pair, so two rings that meet at a corner share a vertex
    only when their texture coordinates also agree; each ring is fan-triangulated. ``texcoord`` is empty
    when the group is untextured.
    """
    pool: dict[tuple[tuple[float, float, float], Any], int] = {}
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for ring, uv_indices in rings:
        indices = []
        for vertex, coordinate in enumerate(ring):
            shifted = (
                coordinate[0] - translate[0],
                coordinate[1] - translate[1],
                coordinate[2] - translate[2],
            )
            uv = (
                tuple(uv_pool[uv_indices[vertex]])
                if textured and uv_indices is not None
                else None
            )
            key = (shifted, uv)
            index = pool.get(key)
            if index is None:
                index = len(positions)
                pool[key] = index
                positions.append(shifted)
                if textured:
                    texcoords.append(uv if uv is not None else (0.0, 0.0))
            indices.append(index)
        for apex in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[apex], indices[apex + 1]))
    position_blob = b"".join(struct.pack("<3f", *position) for position in positions)
    texcoord_blob = b"".join(struct.pack("<2f", *uv) for uv in texcoords)
    index_blob = b"".join(
        struct.pack("<I", index) for triangle in triangles for index in triangle
    )
    return position_blob, texcoord_blob, index_blob, len(positions), len(triangles)


def _pbr_material(
    material_index: int | None,
    texture_index: int | None,
    materials_pool: list[Any],
    gltf_texture: int | None,
) -> dict[str, Any]:
    """A glTF PBR material: ``baseColorFactor`` from the CityJSON ``diffuseColor`` (+ optional texture)."""
    pbr: dict[str, Any] = {}
    if material_index is not None and material_index < len(materials_pool):
        diffuse = materials_pool[material_index].get("diffuseColor")
        pbr["baseColorFactor"] = (
            [*diffuse, 1.0] if diffuse else list(_DEFAULT_BASE_COLOR)
        )
    else:
        pbr["baseColorFactor"] = list(_DEFAULT_BASE_COLOR)
    if texture_index is not None and gltf_texture is not None:
        pbr["baseColorTexture"] = {"index": gltf_texture, "texCoord": 0}
    return {"pbrMetallicRoughness": pbr}


def _build_textured_gltf(model: CityModel) -> dict[str, Any]:
    """Build a glTF document carrying IR appearances (feature #14) — PBR materials + textures.

    Surfaces are grouped by ``(material, texture)`` into one primitive each; a material's
    ``baseColorFactor`` comes from the CityJSON ``diffuseColor`` and a texture contributes
    ``TEXCOORD_0`` (from the ``vertices-texture`` UV pool) + a ``baseColorTexture`` whose image is
    referenced by its CityJSON ``image`` URI. A group with neither is a plain-geometry primitive.
    """
    groups = _collect_groups(model)
    all_positions = [
        point for rings in groups.values() for ring, _uv in rings for point in ring
    ]
    if not all_positions:
        return _empty_gltf()
    translate = [min(point[axis] for point in all_positions) for axis in range(3)]
    materials_pool = model.appearances.get("materials") or []
    textures_pool = model.appearances.get("textures") or []
    uv_pool = model.appearances.get("vertices-texture") or []

    buffer = b""
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    primitives: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    textures: list[dict[str, Any]] = []
    texture_by_source: dict[
        int, int
    ] = {}  # CityJSON texture index -> glTF texture index

    def add_view(blob: bytes, target: int) -> int:
        nonlocal buffer
        buffer += b"\x00" * (-len(buffer) % 4)  # 4-align (float32 / uint32 components)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": len(buffer),
                "byteLength": len(blob),
                "target": target,
            }
        )
        buffer += blob
        return len(buffer_views) - 1

    def gltf_texture_for(source: int) -> int | None:
        image = (
            textures_pool[source].get("image") if source < len(textures_pool) else None
        )
        if not image:
            return None
        if source not in texture_by_source:
            images.append({"uri": image})
            textures.append({"source": len(images) - 1, "sampler": 0})
            texture_by_source[source] = len(textures) - 1
        return texture_by_source[source]

    for (material_index, texture_index), rings in groups.items():
        gltf_texture = (
            gltf_texture_for(texture_index) if texture_index is not None else None
        )
        textured = gltf_texture is not None
        position_blob, texcoord_blob, index_blob, vertices, tris = _primitive_blobs(
            rings, translate, uv_pool, textured
        )
        stored = [
            struct.unpack_from("<3f", position_blob, v * 12) for v in range(vertices)
        ]
        position_view = add_view(position_blob, _ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": position_view,
                "componentType": _FLOAT,
                "count": vertices,
                "type": "VEC3",
                "min": [min(p[a] for p in stored) for a in range(3)],
                "max": [max(p[a] for p in stored) for a in range(3)],
            }
        )
        attributes = {"POSITION": len(accessors) - 1}
        if textured:
            texcoord_view = add_view(texcoord_blob, _ARRAY_BUFFER)
            accessors.append(
                {
                    "bufferView": texcoord_view,
                    "componentType": _FLOAT,
                    "count": vertices,
                    "type": "VEC2",
                }
            )
            attributes["TEXCOORD_0"] = len(accessors) - 1
        index_view = add_view(index_blob, _ELEMENT_ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": index_view,
                "componentType": _UINT,
                "count": tris * 3,
                "type": "SCALAR",
            }
        )
        primitive: dict[str, Any] = {
            "attributes": attributes,
            "indices": len(accessors) - 1,
            "mode": _TRIANGLES,
        }
        if material_index is not None or textured:
            materials.append(
                _pbr_material(
                    material_index, texture_index, materials_pool, gltf_texture
                )
            )
            primitive["material"] = len(materials) - 1
        primitives.append(primitive)

    uri = "data:application/octet-stream;base64," + base64.b64encode(buffer).decode(
        "ascii"
    )
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "pyramids-citymodels"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": translate}],
        "meshes": [{"primitives": primitives}],
        "buffers": [{"byteLength": len(buffer), "uri": uri}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    if materials:
        document["materials"] = materials
    if textures:
        document["textures"] = textures
        document["images"] = images
        document["samplers"] = [{}]
    return document


def write_gltf(
    model: CityModel, path: Any = None, materials: bool = False
) -> dict[str, Any] | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to a self-contained glTF 2.0 document.

    With ``materials=False`` (the default) every surface's exterior ring is triangulated (a fan) into
    one **positions-only** mesh, shifted by the model's per-axis minimum (written as the node
    ``translation`` for float32 precision) and embedded as a base64 buffer.

    With ``materials=True`` the model's **appearances are carried into the glTF**: surfaces are grouped
    by their ``(material, texture)`` into one primitive each, a material's ``baseColorFactor`` comes from
    the CityJSON ``diffuseColor``, and a texture adds a ``TEXCOORD_0`` attribute (from the
    ``vertices-texture`` UV pool) plus a ``baseColorTexture`` whose image is referenced by its CityJSON
    ``image`` URI (not embedded). A model with no appearances yields the same single positions-only
    primitive as the default. See the module docstring for the documented limits.

    Args:
        model: The city model to serialize.
        path: A path to write the ``.gltf`` JSON to, or ``None`` to return the document dict.
        materials: Carry IR appearances (PBR materials + textures) into the glTF rather than writing a
            positions-only mesh.

    Returns:
        The glTF document as a ``dict`` when ``path`` is ``None``; otherwise ``None``.

    Examples:
        - A one-triangle building yields a positions-only triangle mesh:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> gltf = write_gltf(model)
            >>> gltf["accessors"][0]["count"]  # three positions
            3
            >>> gltf["accessors"][1]["count"]  # one triangle -> three indices
            3
            >>> gltf["meshes"][0]["primitives"][0]["mode"]  # TRIANGLES
            4

            ```
        - ``materials=True`` turns a surface's CityJSON material into a glTF PBR ``baseColorFactor``:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> geometry = Geometry(type="MultiSurface", lod="2", boundaries=[surface],
            ...     appearance={"material": {"": {"values": [0]}}})
            >>> model = CityModel(
            ...     objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            ...     appearances={"materials": [{"name": "red", "diffuseColor": [1.0, 0.0, 0.0]}]})
            >>> write_gltf(model, materials=True)["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"]
            [1.0, 0.0, 0.0, 1.0]

            ```
        - An empty model is a valid, mesh-less glTF:
            ```python
            >>> write_gltf(CityModel())["scenes"]
            [{'nodes': []}]

            ```
    """
    document = _build_textured_gltf(model) if materials else _build_gltf(model)
    if path is None:
        return document
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    return None
