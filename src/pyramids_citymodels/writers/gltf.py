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
- **One mesh, positions only** — no normals, materials or textures; appearance round-trips through
  the CityJSON writer, not here.
- An empty model (no areal surface) yields a valid glTF with an empty scene.

Only the standard library (``json`` + ``struct`` + ``base64``) is used.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import base64
import json
import struct
from typing import Any

from pyramids_citymodels.ir import CityModel, iter_surfaces
from pyramids_citymodels.writers._geometry import open_ring as _open_ring
from pyramids_citymodels.writers._geometry import xyz as _xyz

__all__ = ["write_gltf"]

#: glTF constants: float32 / uint32 component types, ARRAY_BUFFER / ELEMENT_ARRAY_BUFFER targets.
_FLOAT, _UINT = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_TRIANGLES = 4


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


def write_gltf(model: CityModel, path: Any = None) -> dict[str, Any] | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to a self-contained glTF 2.0 document.

    Every surface's exterior ring is triangulated (a fan) into one positions-only mesh, shifted by
    the model's per-axis minimum (written as the node ``translation`` for float32 precision) and
    embedded as a base64 buffer. See the module docstring for the documented limits.

    Args:
        model: The city model to serialize.
        path: A path to write the ``.gltf`` JSON to, or ``None`` to return the document dict.

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
        - Positions are shifted and the shift recorded as the node translation:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(100.0, 200.0, 0.0), (101.0, 200.0, 0.0), (101.0, 201.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> write_gltf(model)["nodes"][0]["translation"]
            [100.0, 200.0, 0.0]

            ```
        - An empty model is a valid, mesh-less glTF:
            ```python
            >>> write_gltf(CityModel())["scenes"]
            [{'nodes': []}]

            ```
    """
    document = _build_gltf(model)
    if path is None:
        return document
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    return None
