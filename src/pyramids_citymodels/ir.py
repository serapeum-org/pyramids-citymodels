"""The shared city-model intermediate representation (IR).

Both front doors — CityJSON (:mod:`pyramids_citymodels.readers.cityjson`) and CityGML
(:mod:`pyramids_citymodels.readers.citygml`) — parse a file into this one in-memory model,
and the flattener (:mod:`pyramids_citymodels.flatten`) turns *this* into a
``FeatureCollection``. CityJSON 2.0 is a JSON encoding of a subset of CityGML 3.0, so the two
formats are two serializations of one data model; keeping a single IR is what lets the
hierarchy-to-rows logic be written once instead of per format (issue #5).

The model is deliberately small and format-neutral:

- :class:`CityModel` — the whole file: its city objects, a CRS and free-form metadata.
- :class:`CityObject` — one feature (a ``Building``, a ``BuildingPart``, …) with its
  attributes, its parent/child links and its geometries.
- :class:`Geometry` — one geometry of a city object at one level of detail, with its
  boundaries **already dereferenced to coordinate tuples** (not CityJSON vertex indices) and
  an optional parallel semantic structure.
- :class:`SemanticSurface` — a semantic surface object (``RoofSurface``, ``WallSurface``, …)
  that a geometry's boundaries point at through :attr:`Geometry.semantic_values`.

Boundaries follow the CityJSON nesting (a *surface* is a list of rings, a *ring* is a list of
``(x, y[, z])`` coordinates), so a ``MultiSurface`` is a list of surfaces and a ``Solid`` is a
list of shells of surfaces. :func:`iter_surfaces` walks that nesting to surface level for any
geometry type, which is all the flattener needs.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CityModel",
    "CityObject",
    "Geometry",
    "SemanticSurface",
    "crs_from_identifier",
    "iter_surfaces",
    "normalize_lod",
]

# A coordinate is a 2- or 3-tuple of floats; a ring is a list of coordinates; a surface is a
# list of rings (exterior first, then holes). These aliases are documentation, not enforcement.
Coordinate = "tuple[float, ...]"
Ring = "list[Coordinate]"
Surface = "list[Ring]"


@dataclass
class SemanticSurface:
    """A semantic surface object a geometry's boundaries reference by index.

    Attributes:
        type: The surface class, e.g. ``"RoofSurface"``, ``"WallSurface"``, ``"GroundSurface"``.
        attributes: Any attributes carried on the semantic object itself.
        parent: Index of a parent semantic surface within the same geometry, or ``None``.
        children: Indices of child semantic surfaces within the same geometry.
    """

    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    parent: int | None = None
    children: list[int] = field(default_factory=list)


@dataclass
class Geometry:
    """One geometry of a city object at one level of detail.

    Attributes:
        type: The CityJSON geometry type — ``"MultiSurface"``, ``"CompositeSurface"``,
            ``"Solid"``, ``"MultiSolid"``, … A CityGML LOD2 building is represented here as a
            ``"MultiSurface"`` of its boundary surfaces.
        lod: The level of detail as a string (``"1"``, ``"2"``, ``"2.2"``), or ``None``.
        boundaries: The geometry's boundaries, dereferenced to coordinate tuples (never vertex
            indices). Nested per the CityJSON convention for :attr:`type`.
        semantics: The semantic surface objects this geometry's :attr:`semantic_values` index.
        semantic_values: A structure parallel to :attr:`boundaries` down to surface level, each
            leaf an index into :attr:`semantics` (or ``None``). ``None`` when the geometry
            carries no semantics.
    """

    type: str
    lod: str | None
    boundaries: list[Any]
    semantics: list[SemanticSurface] = field(default_factory=list)
    semantic_values: Any = None

    def semantic_for(self, index: Any) -> SemanticSurface | None:
        """Resolve a semantic value (an index into :attr:`semantics`) to its surface, or ``None``."""
        if index is None or not isinstance(index, int):
            return None
        if 0 <= index < len(self.semantics):
            return self.semantics[index]
        return None


@dataclass
class CityObject:
    """One city object — a feature in the city model.

    Attributes:
        id: The object's identifier, unique within the file.
        type: The object class, e.g. ``"Building"``, ``"BuildingPart"``, ``"Bridge"``.
        attributes: The object's attributes, passed through verbatim.
        geometries: The object's geometries (often one per level of detail).
        parents: Ids of parent objects (e.g. a ``BuildingPart``'s ``Building``).
        children: Ids of child objects.
    """

    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    geometries: list[Geometry] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    @property
    def parent(self) -> str | None:
        """The first parent id, or ``None`` — the value the flattener records as ``parent_id``."""
        return self.parents[0] if self.parents else None


@dataclass
class CityModel:
    """A parsed city model: the whole file as one object graph.

    Attributes:
        objects: The city objects, keyed by id.
        crs: The coordinate reference system as an authority string (``"EPSG:7415"``), or ``None``.
        metadata: Free-form file metadata (title, geographical extent, …).
        version: The format version string as found in the file, or ``None``.
        source_format: ``"CityJSON"`` or ``"CityGML"`` — which front door produced this model.
    """

    objects: dict[str, CityObject] = field(default_factory=dict)
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    source_format: str | None = None


_EPSG_RE = re.compile(r"EPSG.*?(\d{3,6})", re.IGNORECASE)


def crs_from_identifier(identifier: str | None) -> str | None:
    """Extract an ``"EPSG:<code>"`` string from a CRS identifier, or return ``None``.

    Handles the spellings city-model files use interchangeably: an OGC URN
    (``"urn:ogc:def:crs:EPSG::7415"``), an OGC URL
    (``"https://www.opengis.net/def/crs/EPSG/0/7415"``) and the plain ``"EPSG:7415"``.

    Examples:
        >>> crs_from_identifier("urn:ogc:def:crs:EPSG::7415")
        'EPSG:7415'
        >>> crs_from_identifier("https://www.opengis.net/def/crs/EPSG/0/25832")
        'EPSG:25832'
        >>> crs_from_identifier(None) is None
        True
    """
    if not identifier:
        return None
    match = _EPSG_RE.search(identifier)
    return f"EPSG:{match.group(1)}" if match else None


def normalize_lod(lod: Any) -> str | None:
    """Normalize a level-of-detail value so ``"2"``, ``2`` and ``"2.0"`` compare equal.

    Returns the canonical ``float``-stringified form for a numeric LOD, the stripped string for
    a non-numeric one, and ``None`` for ``None``.

    Examples:
        >>> normalize_lod("2")
        '2.0'
        >>> normalize_lod(2)
        '2.0'
        >>> normalize_lod("2.2")
        '2.2'
        >>> normalize_lod(None) is None
        True
    """
    if lod is None:
        return None
    try:
        return str(float(lod))
    except (TypeError, ValueError):
        return str(lod).strip()


def _is_coordinate(value: Any) -> bool:
    """True when ``value`` is a single coordinate — a short sequence of numbers."""
    return (
        isinstance(value, (list, tuple))
        and 2 <= len(value) <= 4
        and all(isinstance(component, (int, float)) for component in value)
    )


def _is_ring(value: Any) -> bool:
    """True when ``value`` is a ring — a non-empty list whose first element is a coordinate."""
    return isinstance(value, list) and len(value) > 0 and _is_coordinate(value[0])


def _is_surface(value: Any) -> bool:
    """True when ``value`` is a surface — a non-empty list whose first element is a ring."""
    return isinstance(value, list) and len(value) > 0 and _is_ring(value[0])


def iter_surfaces(
    boundaries: Any, semantic_values: Any = None
) -> Iterator[tuple[list[Any], Any]]:
    """Walk a geometry's boundaries to surface level, pairing each surface with its semantic value.

    Works for any geometry type by descending the nesting until a *surface* (a list of rings) is
    reached, descending the parallel ``semantic_values`` structure in lock-step so each surface
    is yielded with its own semantic index (or ``None``). This is the one traversal the flattener
    needs for both ``MultiSurface`` (surfaces one level down, values a flat list) and ``Solid``
    (surfaces two levels down, values a list of lists).

    Yields:
        ``(surface, semantic_value)`` where ``surface`` is a list of rings and ``semantic_value``
        is the matching leaf of ``semantic_values`` (an ``int`` index or ``None``).
    """
    if _is_surface(boundaries):
        yield boundaries, semantic_values
        return
    if not isinstance(boundaries, list):
        return
    if isinstance(semantic_values, list) and len(semantic_values) == len(boundaries):
        paired = zip(boundaries, semantic_values)
    else:
        paired = zip(boundaries, [None] * len(boundaries))
    for child, child_values in paired:
        if _is_surface(child):
            yield child, child_values
        elif isinstance(child, list):
            yield from iter_surfaces(child, child_values)
