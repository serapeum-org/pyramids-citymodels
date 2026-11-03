"""Read options: how the 3D, hierarchical city model is projected onto flat features.

A GeoDataFrame is 2D-centric, flat and one-geometry-per-row; a city model is 3D, hierarchical
and graph-linked. There is no single right projection, so the projection is exposed as options
rather than hardcoded — the lesson the QGIS CityJSON Loader and 3DCityDB both teach (issues #9
and #10):

- :class:`LoadStrategy` — *what a row is*: one row per city object (its footprint) or one row
  per semantic surface (wall/roof/ground as individual features).
- :class:`GeometryDimension` — *how Z is carried*: dropped for a 2D footprint, or kept for 2.5D
  surfaces. A ``footprint`` load is always 2D regardless, since a footprint is a plan projection.
- ``lod`` — which level of detail to select when a city object carries several.
- ``child_inherits_parent_attributes`` — whether a ``BuildingPart`` row also carries its
  ``Building``'s attributes.

:class:`ReadOptions` bundles these, and :meth:`ReadOptions.coerce` accepts either the enums or
their plain-string values so a driver can pass user kwargs straight through.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "GeometryDimension",
    "LoadStrategy",
    "ObjectFilter",
    "ReadOptions",
]


class LoadStrategy(StrEnum):
    """What a row in the output represents."""

    #: One row per city object, geometry = its 2D footprint (its ground surfaces).
    FOOTPRINT = "footprint"
    #: One row per semantic surface — wall/roof/ground each become an individual feature.
    SEMANTIC_SURFACES = "semantic_surfaces"


class GeometryDimension(StrEnum):
    """How Z is carried in the output geometry."""

    #: Drop Z — 2D geometry.
    XY = "xy"
    #: Keep Z — 2.5D geometry (coordinates retain their height).
    XYZ = "xyz"


def _as_str_set(value: Any) -> "frozenset[str] | None":
    """Coerce an id/type criterion to a ``frozenset[str]``, or ``None`` when empty.

    A lone ``str`` is one member (not a set of its characters); any other iterable is taken
    member-wise. An empty value (``None``, ``""``, ``[]``) becomes ``None`` — "do not filter on
    this" — so an empty criterion never silently rejects every object.

    Examples:
        >>> _as_str_set("Building")
        frozenset({'Building'})
        >>> sorted(_as_str_set(["a", "b"]))
        ['a', 'b']
        >>> _as_str_set([]) is None
        True
    """
    if value is None:
        return None
    members = frozenset(
        {value} if isinstance(value, str) else (str(item) for item in value)
    )
    return members or None


def _as_bbox(value: Any) -> "tuple[float, float, float, float] | None":
    """Coerce a bbox criterion to a normalized ``(xmin, ymin, xmax, ymax)``, or ``None`` when empty.

    Accepts any 4-number sequence and orders the corners, so a caller may pass them in either
    order. An empty value becomes ``None`` — "do not filter by bbox".

    Raises:
        ValueError: If ``value`` is not a sequence of exactly four numbers.

    Examples:
        >>> _as_bbox((10, 0, 0, 10))
        (0.0, 0.0, 10.0, 10.0)
        >>> _as_bbox(None) is None
        True
    """
    if value is None:
        return None
    try:
        coords = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "bbox must be a sequence of four numbers (xmin, ymin, xmax, ymax)"
        ) from error
    if len(coords) != 4:
        raise ValueError(
            f"bbox must be (xmin, ymin, xmax, ymax) — four numbers, got {len(coords)}"
        )
    xmin, ymin, xmax, ymax = coords
    return (min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))


@dataclass(frozen=True)
class ObjectFilter:
    """A read-time subset: which city objects a reader keeps, decided *before* geometry is built.

    This is the parse-time half of "subset & filtering". Rather than read a whole city-scale file
    and discard rows afterwards (filtering a ``FeatureCollection`` is a GIS operation that belongs
    upstream in ``pyramids-core``), a reader consults this to **skip non-matching objects as it
    parses**, so their geometry is never decoded. An object is kept only when it satisfies *every*
    criterion that is set — ``ids`` AND ``types`` AND each entry of ``attributes`` AND ``bbox``; a
    criterion left unset (``None``/empty) is not tested, so an empty filter keeps everything.

    ``ids``/``types``/``attributes`` are **metadata** tests, decided from an object's header before
    any geometry is touched. ``bbox`` is a **conservative prefilter**: an object is skipped only
    when a *declared extent the file already carries* (a CityJSON ``geographicalExtent`` or a
    CityGML ``gml:Envelope``) proves it lies wholly outside the box. An object with no declared
    extent is **kept**, because proving it is outside would require decoding the very geometry this
    is meant to skip; the bbox result is thus a superset of the exact selection, which a caller
    refines downstream with ``FeatureCollection.clip_by_rect``. The bbox is read in the file's own
    CRS (no reprojection here — that is ``pyramids-core``'s job).

    Filtering is per object and does not pull in relatives: keeping a ``BuildingPart`` whose
    ``Building`` was filtered out leaves a dangling ``parents`` id (the flattener resolves a missing
    parent to ``None``), and in CityGML a dropped feature's boundary surfaces roll up to the nearest
    *kept* ancestor feature. This is deliberate — the filter selects, it does not re-link.

    Attributes:
        ids: Keep only objects whose id is in this set, or ``None`` to not filter by id.
        types: Keep only objects whose type is in this set, or ``None`` to not filter by type.
        attributes: Keep only objects whose attributes equal each of these key/value pairs; an
            empty mapping does not filter by attribute.
        bbox: Keep only objects whose declared extent overlaps this ``(xmin, ymin, xmax, ymax)``
            (objects with no declared extent are kept), or ``None`` to not filter by bbox.
    """

    ids: "frozenset[str] | None" = None
    types: "frozenset[str] | None" = None
    attributes: dict[str, Any] = field(default_factory=dict)
    bbox: "tuple[float, float, float, float] | None" = None

    @classmethod
    def coerce(
        cls,
        ids: "str | Iterable[str] | None" = None,
        types: "str | Iterable[str] | None" = None,
        attributes: "Mapping[str, Any] | None" = None,
        bbox: "Iterable[float] | None" = None,
    ) -> "ObjectFilter | None":
        """Build an :class:`ObjectFilter`, or ``None`` when no criterion is given.

        Accepts a lone ``str`` or any iterable of strings for ``ids``/``types``, a mapping for
        ``attributes``, and a 4-number ``(xmin, ymin, xmax, ymax)`` sequence for ``bbox`` (corners
        in either order). Returns ``None`` when every criterion is empty, so a reader can treat "no
        filter" as a cheap ``is None`` check and skip the per-object test entirely.

        Raises:
            ValueError: If ``bbox`` is given but is not four numbers.

        Examples:
            >>> ObjectFilter.coerce() is None
            True
            >>> ObjectFilter.coerce(types="Building").types
            frozenset({'Building'})
            >>> ObjectFilter.coerce(bbox=(0, 0, 10, 10)).bbox
            (0.0, 0.0, 10.0, 10.0)
        """
        id_set = _as_str_set(ids)
        type_set = _as_str_set(types)
        attribute_map = dict(attributes) if attributes else {}
        box = _as_bbox(bbox)
        if id_set is None and type_set is None and not attribute_map and box is None:
            return None
        return cls(ids=id_set, types=type_set, attributes=attribute_map, bbox=box)

    def matches(
        self, object_id: str, object_type: str, attributes: "Mapping[str, Any]"
    ) -> bool:
        """True when a city object passes every criterion this filter sets.

        Args:
            object_id: The object's id, tested against :attr:`ids`.
            object_type: The object's type, tested against :attr:`types`.
            attributes: The object's attributes, tested against :attr:`attributes` by equality.

        Returns:
            ``True`` when the object is kept, ``False`` when it is filtered out.

        Examples:
            >>> keep = ObjectFilter.coerce(types=["Building"], attributes={"storeys": 3})
            >>> keep.matches("b1", "Building", {"storeys": 3})
            True
            >>> keep.matches("b2", "Building", {"storeys": 5})
            False
            >>> keep.matches("w1", "WallSurface", {"storeys": 3})
            False
        """
        if self.ids is not None and object_id not in self.ids:
            return False
        if self.types is not None and object_type not in self.types:
            return False
        for key, value in self.attributes.items():
            if attributes.get(key) != value:
                return False
        return True

    def rejects_extent(
        self, extent: "tuple[float, float, float, float] | None"
    ) -> bool:
        """True when a declared ``extent`` proves the object lies wholly outside :attr:`bbox`.

        The conservative bbox prefilter: it returns ``True`` (skip the object) only when there is a
        bbox *and* a known 2D ``extent`` *and* the two are disjoint. With no bbox, or an unknown
        extent (``None``), or any overlap, it returns ``False`` (keep the object) — so an object is
        never dropped on a bbox the reader cannot prove it misses.

        Args:
            extent: The object's declared extent as ``(xmin, ymin, xmax, ymax)``, or ``None`` when
                the file declares none for it.

        Returns:
            ``True`` to skip the object, ``False`` to keep it.

        Examples:
            >>> box = ObjectFilter.coerce(bbox=(0, 0, 10, 10))
            >>> box.rejects_extent((20, 20, 30, 30))  # disjoint -> skip
            True
            >>> box.rejects_extent((5, 5, 15, 15))  # overlaps -> keep
            False
            >>> box.rejects_extent(None)  # unknown extent -> keep
            False
            >>> ObjectFilter.coerce(ids="a").rejects_extent((20, 20, 30, 30))  # no bbox -> keep
            False
        """
        if self.bbox is None or extent is None:
            return False
        exmin, eymin, exmax, eymax = extent
        bxmin, bymin, bxmax, bymax = self.bbox
        return exmax < bxmin or exmin > bxmax or eymax < bymin or eymin > bymax


@dataclass(frozen=True)
class ReadOptions:
    """The resolved set of read options the flattener consumes.

    Attributes:
        strategy: The :class:`LoadStrategy`.
        lod: The level of detail to select, or ``None`` to take the highest each object offers.
        geometry_dimension: The :class:`GeometryDimension` (ignored for a footprint load, which
            is always 2D).
        child_inherits_parent_attributes: When ``True``, a child object's rows also carry its
            parent's attributes (the parent's values do not overwrite the child's own).
    """

    strategy: LoadStrategy = LoadStrategy.SEMANTIC_SURFACES
    lod: str | None = None
    geometry_dimension: GeometryDimension = GeometryDimension.XYZ
    child_inherits_parent_attributes: bool = False

    @classmethod
    def coerce(
        cls,
        strategy: "str | LoadStrategy" = LoadStrategy.SEMANTIC_SURFACES,
        lod: Any = None,
        geometry_dimension: "str | GeometryDimension" = GeometryDimension.XYZ,
        child_inherits_parent_attributes: bool = False,
    ) -> "ReadOptions":
        """Build a :class:`ReadOptions`, accepting enums or their plain-string values.

        Raises:
            ValueError: If ``strategy`` or ``geometry_dimension`` is not a recognized value.

        Examples:
            >>> ReadOptions.coerce(strategy="footprint").strategy is LoadStrategy.FOOTPRINT
            True
            >>> ReadOptions.coerce(geometry_dimension="xy").geometry_dimension
            <GeometryDimension.XY: 'xy'>
            >>> ReadOptions.coerce(lod=2).lod
            '2'
        """
        try:
            resolved_strategy = LoadStrategy(strategy)
        except ValueError as error:
            valid = ", ".join(repr(member.value) for member in LoadStrategy)
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of {valid}"
            ) from error
        try:
            resolved_dimension = GeometryDimension(geometry_dimension)
        except ValueError as error:
            valid = ", ".join(repr(member.value) for member in GeometryDimension)
            raise ValueError(
                f"unknown geometry_dimension {geometry_dimension!r}; expected one of {valid}"
            ) from error
        return cls(
            strategy=resolved_strategy,
            lod=None if lod is None else str(lod),
            geometry_dimension=resolved_dimension,
            child_inherits_parent_attributes=bool(child_inherits_parent_attributes),
        )
