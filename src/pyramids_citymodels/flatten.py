"""Flatten the IR into a ``FeatureCollection`` row model (issue #8).

The 3D, hierarchical city model becomes a flat feature table using the pattern 3DCityDB and the
QGIS Loader both converge on: **one flat table, a ``surface_type`` discriminator column and a
``parent_id`` foreign key**, with object attributes passed through. That single shape serves both
load strategies (:class:`~pyramids_citymodels.options.LoadStrategy`):

- ``semantic_surfaces`` — one row per boundary surface, carrying ``surface_type`` (RoofSurface,
  WallSurface, …), the owning object's id/type, its ``parent_id`` and its attributes.
- ``footprint`` — one row per city object, geometry = the 2D union of its ground surfaces (or,
  when a model tags no ground, the 2D union of all its surfaces).

Z is kept or dropped per :class:`~pyramids_citymodels.options.GeometryDimension` (a footprint is
always 2D). Identical polygons within one object are de-duplicated, which collapses the common
XLink case where one polygon is referenced by both a semantic surface and a solid shell.

**Surface geometries are planar polygons in 3D (2.5D).** A building's *vertical* surface — a wall —
is a real polygon in 3D space whose projection onto the XY plane is a straight line. Shapely
measures validity and area on that XY footprint only, so a wall row is deliberately **non-areal**:
``is_valid`` is ``False`` and ``area`` is ``0`` for it, while roofs and ground (near-horizontal) are
valid with positive area. This is the documented ``semantic_surfaces`` contract, not a defect —
operate on the surfaces in 3D (the Z is preserved in ``xyz`` mode), and treat 2D-areal operations
(overlay, area, ``to_file`` of a 2D driver) as meaningful only for the horizontal surfaces or for a
``footprint`` load. The footprint load, being a plan projection, excludes these non-areal surfaces.

The result is a :class:`pyramids.feature.FeatureCollection` — a GeoDataFrame subclass — so it
flows straight into the rest of the pyramids stack; this module builds no GIS of its own.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any

import geopandas as gpd
from pyramids.feature import FeatureCollection
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from pyramids_citymodels.ir import (
    CityModel,
    CityObject,
    Geometry,
    iter_surfaces,
    normalize_lod,
)
from pyramids_citymodels.options import GeometryDimension, LoadStrategy, ReadOptions

__all__ = ["to_feature_collection"]


def _ring_coordinates(ring: list[Any], keep_z: bool) -> list[tuple[float, ...]]:
    """Coerce a ring's coordinates to 2D or 3D tuples."""
    if keep_z:
        return [
            (coord[0], coord[1], coord[2])
            if len(coord) >= 3
            else (coord[0], coord[1], 0.0)
            for coord in ring
        ]
    return [(coord[0], coord[1]) for coord in ring]


def _polygon(surface: list[Any], keep_z: bool) -> Polygon | None:
    """Build a shapely :class:`Polygon` from a surface (exterior ring first, then holes)."""
    rings = [ring for ring in surface if ring]
    if not rings:
        return None
    exterior = _ring_coordinates(rings[0], keep_z)
    if len(exterior) < 3:
        return None
    holes = [_ring_coordinates(ring, keep_z) for ring in rings[1:] if len(ring) >= 3]
    return Polygon(exterior, holes)


def _select_geometries(geometries: list[Geometry], lod: str | None) -> list[Geometry]:
    """Pick the geometries to flatten: those matching ``lod``, else the highest LOD present."""
    if not geometries:
        return []
    if lod is not None:
        wanted = normalize_lod(lod)
        selected = [g for g in geometries if normalize_lod(g.lod) == wanted]
        if selected:
            return selected
        return []

    def lod_value(geometry: Geometry) -> float:
        normalized = normalize_lod(geometry.lod)
        try:
            return float(normalized) if normalized is not None else -1.0
        except ValueError:
            return -1.0

    highest = max((lod_value(g) for g in geometries), default=-1.0)
    return [g for g in geometries if lod_value(g) == highest]


def _polygon_key(polygon: Polygon) -> tuple:
    """A hashable key for de-duplicating identical polygons (rounded WKT of the exterior)."""
    return tuple(
        round(value, 6) for value in polygon.exterior.coords[:-1] for value in value
    )


def _attributes_for(
    city_object: CityObject, model: CityModel, options: ReadOptions
) -> dict[str, Any]:
    """The attribute dict for an object's rows, optionally inheriting its parent's attributes."""
    attributes: dict[str, Any] = {}
    if options.child_inherits_parent_attributes:
        parent_id = city_object.parent
        parent = model.objects.get(parent_id) if parent_id else None
        if parent is not None:
            attributes.update(parent.attributes)
    attributes.update(city_object.attributes)
    return attributes


def _semantic_surface_rows(
    city_object: CityObject,
    geometries: list[Geometry],
    model: CityModel,
    options: ReadOptions,
) -> list[dict[str, Any]]:
    """One row per boundary surface, with its ``surface_type`` and owner metadata."""
    keep_z = options.geometry_dimension is GeometryDimension.XYZ
    base_attributes = _attributes_for(city_object, model, options)
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for geometry in geometries:
        surface_index = 0
        for surface, semantic_value in iter_surfaces(
            geometry.boundaries, geometry.semantic_values
        ):
            polygon = _polygon(surface, keep_z)
            if polygon is None:
                continue
            key = _polygon_key(polygon)
            if key in seen:
                continue
            seen.add(key)
            semantic = geometry.semantic_for(semantic_value)
            rows.append(
                {
                    **base_attributes,
                    "city_object_id": city_object.id,
                    "city_object_type": city_object.type,
                    "parent_id": city_object.parent,
                    "lod": geometry.lod,
                    "surface_type": semantic.type if semantic is not None else None,
                    "surface_index": surface_index,
                    "geometry": polygon,
                }
            )
            surface_index += 1
    return rows


def _footprint_row(
    city_object: CityObject,
    geometries: list[Geometry],
    model: CityModel,
    options: ReadOptions,
) -> dict[str, Any] | None:
    """One 2D row per object: the union of its ground surfaces, else of all its surfaces."""
    ground: list[Polygon] = []
    everything: list[Polygon] = []
    for geometry in geometries:
        for surface, semantic_value in iter_surfaces(
            geometry.boundaries, geometry.semantic_values
        ):
            polygon = _polygon(surface, keep_z=False)
            if polygon is None or polygon.area == 0.0:
                # A footprint is a plan projection; a non-areal surface (a vertical wall
                # projects to a line, area 0) is not part of it and would only pollute the
                # union with degenerate, invalid geometry (SF1).
                continue
            everything.append(polygon)
            semantic = geometry.semantic_for(semantic_value)
            if semantic is not None and semantic.type == "GroundSurface":
                ground.append(polygon)
    polygons = ground or everything
    if not polygons:
        return None
    merged = unary_union(polygons)
    if merged.is_empty:
        return None
    if merged.geom_type == "Polygon":
        geometry_value: Any = merged
    elif merged.geom_type == "MultiPolygon":
        geometry_value = merged
    else:
        # A union that collapses to lines/points is not a usable footprint; wrap what polygons
        # we have instead of returning a degenerate geometry.
        geometry_value = MultiPolygon([p for p in polygons if p.geom_type == "Polygon"])
    lods = sorted({g.lod for g in geometries if g.lod is not None})
    return {
        **_attributes_for(city_object, model, options),
        "city_object_id": city_object.id,
        "city_object_type": city_object.type,
        "parent_id": city_object.parent,
        "lod": lods[-1] if lods else None,
        "geometry": geometry_value,
    }


def to_feature_collection(
    model: CityModel,
    *,
    strategy: "str | LoadStrategy" = LoadStrategy.SEMANTIC_SURFACES,
    lod: Any = None,
    geometry_dimension: "str | GeometryDimension" = GeometryDimension.XYZ,
    child_inherits_parent_attributes: bool = False,
) -> FeatureCollection:
    """Flatten a :class:`~pyramids_citymodels.ir.CityModel` into a ``FeatureCollection``.

    Args:
        model: The parsed city model.
        strategy: ``"footprint"`` (one row per object) or ``"semantic_surfaces"`` (one row per
            boundary surface). Accepts a :class:`~pyramids_citymodels.options.LoadStrategy`.
        lod: Level of detail to select; ``None`` takes the highest each object offers.
        geometry_dimension: ``"xyz"`` keeps Z, ``"xy"`` drops it. Ignored for ``footprint``.
        child_inherits_parent_attributes: When ``True``, child rows also carry parent attributes.

    Returns:
        A :class:`pyramids.feature.FeatureCollection` in the model's CRS, with a stable column
        order; empty (but still a valid FeatureCollection) when the model has no usable geometry.

    Examples:
        - Flatten a one-building model into one row per semantic surface and read it back:
            ```python
            >>> from pyramids_citymodels.readers.cityjson import parse_cityjson
            >>> doc = {
            ...     "type": "CityJSON", "version": "2.0",
            ...     "CityObjects": {"b": {"type": "Building", "geometry": [
            ...         {"type": "MultiSurface", "lod": "2", "boundaries": [[[0, 1, 2]]],
            ...          "semantics": {"surfaces": [{"type": "RoofSurface"}], "values": [0]}}]}},
            ...     "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]}
            >>> frame = to_feature_collection(parse_cityjson(doc), strategy="semantic_surfaces")
            >>> len(frame)
            1
            >>> frame["surface_type"].iloc[0]
            'RoofSurface'

            ```
    """
    options = ReadOptions.coerce(
        strategy=strategy,
        lod=lod,
        geometry_dimension=geometry_dimension,
        child_inherits_parent_attributes=child_inherits_parent_attributes,
    )

    rows: list[dict[str, Any]] = []
    for city_object in model.objects.values():
        geometries = _select_geometries(city_object.geometries, options.lod)
        if not geometries:
            continue
        if options.strategy is LoadStrategy.SEMANTIC_SURFACES:
            rows.extend(_semantic_surface_rows(city_object, geometries, model, options))
        else:
            row = _footprint_row(city_object, geometries, model, options)
            if row is not None:
                rows.append(row)

    if rows:
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=model.crs)
    else:
        frame = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=model.crs)
    return FeatureCollection(frame)
