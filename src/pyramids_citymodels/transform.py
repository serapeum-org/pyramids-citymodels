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
from typing import Any, cast

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
    return cast(CityModel, replace(model, objects=objects))


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


def grid_snapper(cell_size: float) -> "Callable[[Any], tuple[float, ...]]":
    """A function snapping a coordinate to the nearest ``cell_size`` grid point (for vertex clustering)."""

    def snap(point: Any) -> tuple[float, ...]:
        return tuple(
            round(float(component) / cell_size) * cell_size for component in point
        )

    return snap


def cluster_ring(
    ring: list[Any], snap: "Callable[[Any], tuple[float, ...]]"
) -> list[Any]:
    """Snap a ring to the grid, dropping consecutive duplicates and a trailing vertex equal to the first.

    Shared by :func:`simplify` and the 3D-Tiles per-level LOD builder so the two cannot drift. The result
    is an **open** ring (no closing duplicate); a ring with fewer than three distinct vertices has
    collapsed and the caller drops its surface.
    """
    cleaned: list[Any] = []
    for point in ring:
        snapped = snap(point)
        if not cleaned or snapped != cleaned[-1]:
            cleaned.append(snapped)
    while len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
        cleaned.pop()
    return cleaned


def is_degenerate_ring(ring: list[Any]) -> bool:
    """True when a ring has fewer than three distinct vertices (it has collapsed to a point or a line).

    The single drop criterion shared by :func:`simplify` and the 3D-Tiles LOD builder, so a ring counting
    repeated or alternating vertices (e.g. a zig-zag ``A, B, A, B`` that snaps to two points) is dropped
    the same way everywhere, not kept as a degenerate sliver in one path.
    """
    return len({tuple(point) for point in ring}) < 3


def _is_ring(node: Any) -> bool:
    """True when ``node`` is a ring — a non-empty list whose children are all coordinates."""
    return isinstance(node, list) and bool(node) and all(is_coordinate(c) for c in node)


def _is_surface_node(node: Any) -> bool:
    """True when ``node`` is a surface — a non-empty list whose children are all rings."""
    return isinstance(node, list) and bool(node) and all(_is_ring(r) for r in node)


def _prune_surfaces(
    boundary: Any, semantic: Any, snap: "Callable[[Any], tuple[float, ...]]"
) -> tuple[Any, Any] | None:
    """Cluster every ring and drop collapsed surfaces, keeping ``semantic_values`` aligned.

    Returns ``(new_boundary, new_semantic)`` with each surface's rings clustered, a surface whose
    exterior ring falls below three distinct vertices dropped (together with its parallel semantic
    value), and any degenerate interior ring (hole) dropped; or ``None`` when the whole node collapses to
    nothing (every descendant surface dropped).
    """
    if _is_surface_node(boundary):
        rings = [cluster_ring(ring, snap) for ring in boundary]
        if is_degenerate_ring(rings[0]):
            return None
        kept = [rings[0]] + [r for r in rings[1:] if not is_degenerate_ring(r)]
        return kept, semantic
    if not isinstance(boundary, list):
        return boundary, semantic
    aligned = isinstance(semantic, list) and len(semantic) == len(boundary)
    paired = (
        zip(boundary, semantic) if aligned else zip(boundary, [None] * len(boundary))
    )
    new_boundary: list[Any] = []
    new_semantic: list[Any] = []
    for child_boundary, child_semantic in paired:
        pruned = _prune_surfaces(child_boundary, child_semantic, snap)
        if pruned is not None:
            kept_boundary, kept_semantic = pruned
            new_boundary.append(kept_boundary)
            new_semantic.append(kept_semantic)
    if not new_boundary:
        return None
    return new_boundary, (new_semantic if aligned else semantic)


def simplify(model: CityModel, cell_size: float) -> CityModel:
    """A coarser new model: vertices clustered to a ``cell_size`` grid (lossy LOD decimation).

    Vertex clustering is the stdlib-only mesh-simplification route (quadric/edge-collapse decimation
    would need a third-party dependency). Every coordinate is snapped to the nearest ``cell_size`` grid
    point, so vertices within one cell merge and consecutive duplicate ring vertices are dropped. A
    surface whose exterior ring then has fewer than three distinct vertices has **collapsed** and is
    **dropped** (its parallel semantic value dropped too, so ``semantic_values`` stays aligned), so the
    result is clean geometry for *every* writer — not a degenerate ring that only the mesh writers skip.
    Each coordinate moves at most ``cell_size / 2`` per axis (a bounded error), and a second ``simplify``
    at the same ``cell_size`` is a no-op (idempotent).

    Per-geometry **appearance is dropped** (like :func:`triangulate`): clustering changes a ring's vertex
    count, which invalidates the per-vertex texture-coordinate correspondence, so it cannot be carried
    faithfully. Semantic surface types are kept (re-aligned to the surviving surfaces).

    Args:
        model: The city model to decimate.
        cell_size: The clustering grid cell size in the model's own units (must be positive); larger is
            coarser.

    Returns:
        A new :class:`~pyramids_citymodels.ir.CityModel` with clustered geometry (inputs untouched).

    Raises:
        ValueError: If ``cell_size`` is not positive.
    """
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size!r}")
    snap = grid_snapper(cell_size)

    def decimate(geometry: Geometry) -> Geometry:
        pruned = _prune_surfaces(geometry.boundaries, geometry.semantic_values, snap)
        boundaries, semantic_values = pruned if pruned is not None else ([], None)
        coarsened: Geometry = replace(
            geometry,
            boundaries=boundaries,
            semantic_values=semantic_values,
            appearance=None,
        )
        return coarsened

    return _rebuild(model, decimate)


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
    return cast(
        Geometry,
        replace(
            geometry,
            type="MultiSurface",
            boundaries=surfaces,
            semantic_values=values if geometry.semantics else None,
            appearance=None,
        ),
    )


def triangulate(model: CityModel) -> CityModel:
    """A new model with every surface's exterior ring fan-triangulated into triangles.

    Each triangle keeps its source surface's semantic value; interior rings (holes) and per-geometry
    appearance are dropped, and every geometry becomes a ``MultiSurface`` (documented limits).
    """
    return _rebuild(model, _triangulate_geometry)
