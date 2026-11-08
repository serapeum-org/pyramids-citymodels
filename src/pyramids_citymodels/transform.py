"""IR-level geometry processing & cleanup (issue #17 / #52).

Pure transforms over a :class:`~pyramids_citymodels.ir.CityModel`, each returning a **new** model
(inputs untouched), implemented without a new runtime dependency — the unglamorous prep that makes a
downstream consumer's job easier (cf. cjio's ``triangulate`` / ``vertices_clean`` / ``compress``):

- :func:`translate` — shift every coordinate by a per-axis offset (e.g. height change).
- :func:`quantize` — round every coordinate to a given number of decimals (lossy compression prep).
- :func:`clean_rings` — drop consecutive duplicate vertices within each ring.
- :func:`triangulate` — fan-triangulate every surface's exterior ring into triangles.

``translate`` / ``quantize`` / ``clean_rings`` touch only coordinates, so they preserve geometry type,
semantics and appearance. ``triangulate`` changes the surface set, so it keeps each triangle's semantic
value (replicated from its source surface) but **drops** per-geometry appearance and interior rings
(holes), and yields a ``MultiSurface`` (documented limits, like the mesh writers).
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

from pyramids_citymodels.ir import CityModel, Geometry, is_coordinate, iter_surfaces

__all__ = ["translate", "quantize", "clean_rings", "triangulate"]


def _rebuild(
    model: CityModel, geometry_fn: Callable[[Geometry], Geometry]
) -> CityModel:
    """A new model with ``geometry_fn`` applied to every geometry (all else deep-copied).

    Each city object is deep-copied before ``geometry_fn`` runs, so the returned model shares **no**
    mutable structure with the input — its ``semantics`` / ``appearance`` / ``attributes`` are
    independent objects. This mirrors :meth:`CityModel.merge`'s guarantee: a later in-place edit of a
    transformed model must not leak back into the source (the transforms themselves never mutate).
    """
    objects = {}
    for object_id, city_object in model.objects.items():
        copied = deepcopy(city_object)
        objects[object_id] = replace(
            copied,
            geometries=[geometry_fn(geometry) for geometry in copied.geometries],
        )
    return replace(model, objects=objects)


def _map_coordinates(node: Any, point_fn: Callable[[Any], Any]) -> Any:
    """Apply ``point_fn`` to every coordinate in a boundaries structure, preserving its nesting."""
    if is_coordinate(node):
        return point_fn(node)
    if isinstance(node, list):
        return [_map_coordinates(child, point_fn) for child in node]
    return node


def _map_rings(node: Any, ring_fn: Callable[[list[Any]], list[Any]]) -> Any:
    """Apply ``ring_fn`` to every ring (a list whose elements are all coordinates) in ``node``."""
    if isinstance(node, list) and node and all(is_coordinate(child) for child in node):
        return ring_fn(node)
    if isinstance(node, list):
        return [_map_rings(child, ring_fn) for child in node]
    return node


def translate(model: CityModel, offset: "tuple[float, ...]") -> CityModel:
    """A new model with every coordinate shifted by ``offset`` (per axis; missing axes shift by 0)."""

    def shift(point: Any) -> tuple[float, ...]:
        return tuple(
            float(component) + (offset[axis] if axis < len(offset) else 0.0)
            for axis, component in enumerate(point)
        )

    return _rebuild(
        model, lambda g: replace(g, boundaries=_map_coordinates(g.boundaries, shift))
    )


def quantize(model: CityModel, digits: int = 3) -> CityModel:
    """A new model with every coordinate component rounded to ``digits`` decimals."""

    def round_point(point: Any) -> tuple[float, ...]:
        return tuple(round(float(component), digits) for component in point)

    return _rebuild(
        model,
        lambda g: replace(g, boundaries=_map_coordinates(g.boundaries, round_point)),
    )


def clean_rings(model: CityModel) -> CityModel:
    """A new model with consecutive duplicate vertices removed from every ring."""

    def dedupe(ring: list[Any]) -> list[Any]:
        cleaned: list[Any] = []
        for point in ring:
            if not cleaned or tuple(point) != tuple(cleaned[-1]):
                cleaned.append(point)
        return cleaned

    return _rebuild(
        model, lambda g: replace(g, boundaries=_map_rings(g.boundaries, dedupe))
    )


def _open_ring(ring: list[Any]) -> list[Any]:
    """Drop a ring's closing duplicate vertex (``last == first``)."""
    if len(ring) > 1 and tuple(ring[0]) == tuple(ring[-1]):
        return list(ring[:-1])
    return list(ring)


def _triangulate_geometry(geometry: Geometry) -> Geometry:
    """Fan-triangulate a geometry's surfaces into triangle surfaces, keeping per-surface semantics."""
    surfaces: list[Any] = []
    values: list[Any] = []
    for surface, value in iter_surfaces(geometry.boundaries, geometry.semantic_values):
        if not surface:
            continue
        ring = _open_ring(surface[0])
        if len(ring) < 3:
            continue
        for apex in range(1, len(ring) - 1):  # fan from the first vertex
            surfaces.append([[ring[0], ring[apex], ring[apex + 1]]])
            values.append(value)
    return replace(
        geometry,
        type="MultiSurface",
        boundaries=surfaces,
        semantic_values=values if geometry.semantics else None,
        appearance=None,
    )


def triangulate(model: CityModel) -> CityModel:
    """A new model with every surface's exterior ring fan-triangulated into triangles.

    Each triangle keeps its source surface's semantic value; interior rings (holes) and per-geometry
    appearance are dropped, and every geometry becomes a ``MultiSurface`` (documented limits).
    """
    return _rebuild(model, _triangulate_geometry)
