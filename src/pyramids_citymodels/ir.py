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
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pyramids_citymodels.errors import CityModelError

__all__ = [
    "CityModel",
    "CityObject",
    "Geometry",
    "SemanticSurface",
    "VertexPool",
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
        appearance: The geometry's raw appearance references (CityJSON ``material`` / ``texture``
            maps), passed through untouched, or ``None``. Carried so a later renderer can resolve
            materials/textures against :attr:`CityModel.appearances` without this package decoding
            them (report §5, risk #4).
    """

    type: str
    lod: str | None
    boundaries: list[Any]
    semantics: list[SemanticSurface] = field(default_factory=list)
    semantic_values: Any = None
    appearance: dict[str, Any] | None = None

    def semantic_for(self, index: Any) -> SemanticSurface | None:
        """Resolve a semantic value (an index into :attr:`semantics`) to its surface, or ``None``."""
        if index is None or not isinstance(index, int):
            return None
        if 0 <= index < len(self.semantics):
            return self.semantics[index]
        return None

    def coordinates(self) -> Iterator[tuple[float, ...]]:
        """Yield every coordinate in :attr:`boundaries`, in document order, repeats included.

        Walks the boundary nesting (surface → ring → coordinate) to coordinate level for any
        geometry type, which is what :meth:`CityModel.vertex_pool` needs to size and quantize a
        rebuilt pool. The same coordinate appearing in two rings is yielded twice; de-duplication is
        the pool's job (:meth:`VertexPool.index_of`), not the walk's.

        Yields:
            Each coordinate of :attr:`boundaries` as a ``float`` tuple, in document order.

        Examples:
            - Walk a ``MultiSurface`` (surfaces one level down) to its coordinates:
                ```python
                >>> geometry = Geometry(
                ...     type="MultiSurface", lod="2",
                ...     boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]])
                >>> list(geometry.coordinates())
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]

                ```
            - A ``Solid`` (surfaces two levels down) is walked by the same call, repeats included:
                ```python
                >>> ring = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
                >>> geometry = Geometry(type="Solid", lod="2", boundaries=[[[ring], [ring]]])
                >>> len(list(geometry.coordinates()))
                4

                ```
        """
        yield from _iter_coordinates(self.boundaries)

    def index_into(self, pool: "VertexPool") -> list[Any]:
        """Re-index this geometry's boundaries against ``pool``, the inverse of a reader's deref.

        Returns a boundaries structure of the same nesting as :attr:`boundaries` but with every
        coordinate tuple replaced by its integer index in ``pool`` — the CityJSON on-disk form.
        ``pool`` is filled as a side effect (:meth:`VertexPool.index_of` appends coordinates it has
        not seen), so re-indexing every geometry of a model against one shared pool produces the
        single document-wide ``vertices`` array plus each geometry's indexed ``boundaries`` that a
        CityJSON writer emits (issue #24).

        Args:
            pool: The shared :class:`VertexPool` to index into; it is mutated, gaining any
                coordinate of this geometry it does not already hold.

        Returns:
            The boundaries with each coordinate tuple replaced by its integer index in ``pool``,
            nesting preserved.

        Examples:
            - Re-index a one-surface geometry and read back the pool it filled:
                ```python
                >>> geometry = Geometry(
                ...     type="MultiSurface", lod="2",
                ...     boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]])
                >>> pool = VertexPool()
                >>> geometry.index_into(pool)
                [[[0, 1, 2]]]
                >>> pool.vertices
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]

                ```
            - Indexing is the inverse of :meth:`VertexPool.dereference`, so the round-trip is exact:
                ```python
                >>> geometry = Geometry(
                ...     type="MultiSurface", lod="2",
                ...     boundaries=[[[(2.0, 3.0, 4.0), (5.0, 6.0, 7.0)]]])
                >>> pool = VertexPool()
                >>> pool.dereference(geometry.index_into(pool)) == geometry.boundaries
                True

                ```
        """
        indexed: list[Any] = _map_coordinates(self.boundaries, pool.index_of)
        return indexed


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
        appearances: The document-level appearance pool (CityJSON ``appearance`` — ``materials`` /
            ``textures``), passed through untouched, or an empty dict. The geometries'
            :attr:`Geometry.appearance` maps index into this.
        extensions: Schema-extension information carried through verbatim (CityJSON ``extensions``;
            a CityGML ADE's recorded namespaces), or an empty dict. Pass-through so an unknown
            third-party extension is never silently discarded (report §5, risk #5).
    """

    objects: dict[str, CityObject] = field(default_factory=dict)
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    source_format: str | None = None
    appearances: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    def coordinates(self) -> Iterator[tuple[float, ...]]:
        """Yield every coordinate of every geometry of every object, in document order.

        The flat stream the pool builder consumes to compute a ``transform`` and fill a shared
        :class:`VertexPool`; it visits objects then their geometries in insertion order, so a pool
        rebuilt from it is deterministic.

        Yields:
            Every coordinate of every geometry as a ``float`` tuple, objects then geometries in
            insertion order, repeats included.

        Examples:
            - Stream the coordinates of a one-building model:
                ```python
                >>> geometry = Geometry(
                ...     type="MultiSurface", lod="2",
                ...     boundaries=[[[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]]])
                >>> model = CityModel(
                ...     objects={"b": CityObject(id="b", type="Building", geometries=[geometry])})
                >>> list(model.coordinates())
                [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

                ```
            - An empty model yields nothing:
                ```python
                >>> list(CityModel().coordinates())
                []

                ```
        """
        for city_object in self.objects.values():
            for geometry in city_object.geometries:
                yield from geometry.coordinates()

    def vertex_pool(self, precision: int | None = None) -> "VertexPool":
        """Rebuild the flat, de-duplicated vertex pool a CityJSON writer serializes from (issue #24).

        The CityJSON readers dereference ``vertices`` indices to coordinate tuples and drop the
        ``transform`` (:mod:`pyramids_citymodels.readers.cityjson`), so the IR is format-neutral and
        carries no pool. Rather than retain the source pool — which a CityGML-sourced model never has
        and which would drift from edited boundaries — the pool is **rebuilt from the coordinates**:
        this returns an empty :class:`VertexPool` sized to :meth:`coordinates` (its ``transform`` set
        from their extent when ``precision`` quantizes, ``None`` when lossless), which the caller
        fills by re-indexing each geometry with :meth:`Geometry.index_into`.

        Args:
            precision: Decimal places to quantize coordinates to, via a CityJSON ``transform``. With
                ``None`` (the default) the pool is lossless — coordinates are pooled unchanged and a
                round-trip is exact.

        Returns:
            An empty pool configured for this model's coordinates, ready for
            :meth:`Geometry.index_into` to populate.

        Examples:
            - Rebuild a pool, re-index a geometry against it, and confirm the round-trip is exact:
                ```python
                >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                >>> geometry = Geometry(type="MultiSurface", lod="2", boundaries=[surface])
                >>> model = CityModel(
                ...     objects={"b": CityObject(id="b", type="Building", geometries=[geometry])})
                >>> pool = model.vertex_pool()
                >>> indexed = geometry.index_into(pool)
                >>> indexed
                [[[0, 1, 2]]]
                >>> pool.vertices
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
                >>> pool.dereference(indexed) == geometry.boundaries
                True

                ```
        """
        return VertexPool.for_coordinates(self.coordinates(), precision)


@dataclass
class VertexPool:
    """A flat, de-duplicated coordinate pool plus the optional ``transform`` that quantizes it.

    This is CityJSON's on-disk shape for geometry: every distinct coordinate is stored once in
    :attr:`vertices`, and a geometry's boundaries reference it by integer index — optionally through
    a ``transform`` (per-axis ``scale`` + ``translate``) that recovers quantized integers to reals.
    The readers dereference all of that away (:mod:`pyramids_citymodels.readers.cityjson`), so the IR
    carries coordinate tuples and no pool; a writer must **rebuild** one, and this value object is
    that rebuilt pool (issue #24). It is created empty by :meth:`CityModel.vertex_pool`, filled by
    :meth:`Geometry.index_into`, and :meth:`dereference` is its exact inverse — the property a
    round-trip test asserts.

    Attributes:
        vertices: The distinct coordinates in first-seen order — integer tuples when :attr:`transform`
            is set, float tuples when it is ``None``.
        transform: The CityJSON ``{"scale": [...], "translate": [...]}`` mapping a stored integer
            vertex to a real coordinate, or ``None`` when coordinates are pooled losslessly as-is.

    Examples:
        - Pool coordinates by hand; a repeat returns its first index without growing the pool:
            ```python
            >>> pool = VertexPool()
            >>> pool.index_of((0.0, 0.0, 0.0))
            0
            >>> pool.index_of((1.0, 0.0, 0.0))
            1
            >>> pool.index_of((0.0, 0.0, 0.0))
            0
            >>> pool.vertices
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]

            ```
        - A pool with a ``transform`` stores quantized integers but reads back as reals:
            ```python
            >>> pool = VertexPool(transform={"scale": [0.5, 0.5], "translate": [10.0, 20.0]})
            >>> pool.index_of((11.0, 21.0))
            0
            >>> pool.vertices
            [(2, 2)]
            >>> pool.dereference(0)
            (11.0, 21.0)

            ```
    """

    vertices: list[tuple[float, ...]] = field(default_factory=list)
    transform: dict[str, list[float]] | None = None

    def __post_init__(self) -> None:
        # Index coordinates for O(1) de-duplication in index_of; rebuilt from any preloaded vertices
        # so a pool constructed with vertices already present still dedupes correctly. Track the
        # pool's single coordinate dimension too, validating that any preloaded vertices are uniform.
        self._index: dict[tuple[float, ...], int] = {
            vertex: position for position, vertex in enumerate(self.vertices)
        }
        self._dimension: int | None = None
        for vertex in self.vertices:
            self._dimension = _require_uniform_dimension(self._dimension, len(vertex))

    @classmethod
    def for_coordinates(
        cls, coordinates: Iterable[Sequence[float]], precision: int | None = None
    ) -> "VertexPool":
        """Return an empty pool configured for ``coordinates``, to be filled by :meth:`index_of`.

        With ``precision`` (decimal places) the pool quantizes coordinates to integers via a
        ``transform`` whose ``translate`` is their per-axis minimum and whose ``scale`` is
        ``10 ** -precision`` — the CityJSON storage convention, which keeps the integer vertices
        small. With ``precision`` ``None`` the pool is lossless: :attr:`transform` is ``None``,
        coordinates are pooled unchanged, and ``coordinates`` is not consumed (no extent is needed).

        Args:
            coordinates: The coordinate stream the pool will hold, e.g.
                :meth:`CityModel.coordinates`. Consumed once, only when ``precision`` is given.
            precision: Decimal places to quantize to, or ``None`` for a lossless pool.

        Returns:
            An empty :class:`VertexPool`: its :attr:`transform` is ``None`` for a lossless pool, or
            ``{"scale", "translate"}`` sized to the coordinates' extent for a quantizing one.

        Examples:
            - A lossless pool has no transform and starts empty:
                ```python
                >>> pool = VertexPool.for_coordinates([(0.0, 0.0), (1.0, 1.0)])
                >>> pool.transform is None
                True
                >>> pool.vertices
                []

                ```
            - A quantizing pool derives ``translate`` from the coordinates' per-axis minimum:
                ```python
                >>> pool = VertexPool.for_coordinates([(10.0, 20.0), (11.0, 22.0)], precision=2)
                >>> pool.transform
                {'scale': [0.01, 0.01], 'translate': [10.0, 20.0]}

                ```
        """
        if precision is None:
            return cls(transform=None)
        minima: list[float] = []
        dimension: int | None = None
        for coordinate in coordinates:
            dimension = _require_uniform_dimension(dimension, len(coordinate))
            for axis, component in enumerate(coordinate):
                value = float(component)
                if axis < len(minima):
                    minima[axis] = min(minima[axis], value)
                else:
                    minima.append(value)
        scale = round(10.0**-precision, precision)
        return cls(
            transform={"scale": [scale] * len(minima), "translate": list(minima)}
        )

    def quantize(self, coordinate: Sequence[float]) -> tuple[float, ...]:
        """Map a real coordinate to its stored form — a quantized integer tuple, or floats as-is.

        Args:
            coordinate: A real coordinate to convert to the pool's stored representation.

        Returns:
            Integer steps from the ``transform``'s ``translate`` origin when a transform is set,
            otherwise the coordinate's components as floats.

        Examples:
            - Without a transform, components are returned as floats:
                ```python
                >>> VertexPool().quantize((1, 2, 3))
                (1.0, 2.0, 3.0)

                ```
            - With a transform, the coordinate becomes integer steps from ``translate``:
                ```python
                >>> pool = VertexPool(transform={"scale": [0.01, 0.01], "translate": [100.0, 200.0]})
                >>> pool.quantize((100.5, 200.25))
                (50, 25)

                ```
        """
        if self.transform is None:
            return tuple(float(component) for component in coordinate)
        scale = self.transform["scale"]
        translate = self.transform["translate"]
        return tuple(
            int(round((float(coordinate[axis]) - translate[axis]) / scale[axis]))
            for axis in range(len(coordinate))
        )

    def dequantize(self, vertex: Sequence[float]) -> tuple[float, ...]:
        """Map a stored vertex back to a real coordinate — the inverse of :meth:`quantize`.

        Args:
            vertex: A stored vertex (an integer tuple when a transform is set, else floats).

        Returns:
            The real coordinate: ``vertex * scale + translate`` per axis when a transform is set,
            otherwise the components as floats.

        Examples:
            - Without a transform, the vertex is returned unchanged as floats:
                ```python
                >>> VertexPool().dequantize((1, 2, 3))
                (1.0, 2.0, 3.0)

                ```
            - With a transform, integer steps are scaled back to a real coordinate:
                ```python
                >>> pool = VertexPool(transform={"scale": [0.01, 0.01], "translate": [100.0, 200.0]})
                >>> pool.dequantize((50, 25))
                (100.5, 200.25)

                ```
        """
        if self.transform is None:
            return tuple(float(component) for component in vertex)
        scale = self.transform["scale"]
        translate = self.transform["translate"]
        return tuple(
            vertex[axis] * scale[axis] + translate[axis] for axis in range(len(vertex))
        )

    def index_of(self, coordinate: Sequence[float]) -> int:
        """Return ``coordinate``'s index in the pool, appending it (quantized) when first seen.

        Args:
            coordinate: A real coordinate to look up or add; it is quantized by :meth:`quantize`
                before de-duplication, so two coordinates that quantize alike share one index.

        Returns:
            The coordinate's integer index in :attr:`vertices` — an existing one when already
            present, otherwise the index of the freshly appended vertex.

        Examples:
            - New coordinates are appended; a repeat returns its existing index:
                ```python
                >>> pool = VertexPool()
                >>> pool.index_of((5.0, 6.0))
                0
                >>> pool.index_of((7.0, 8.0))
                1
                >>> pool.index_of((5.0, 6.0))
                0
                >>> len(pool.vertices)
                2

                ```
        """
        key = self.quantize(coordinate)
        self._dimension = _require_uniform_dimension(self._dimension, len(key))
        existing = self._index.get(key)
        if existing is not None:
            return existing
        position = len(self.vertices)
        self.vertices.append(key)
        self._index[key] = position
        return position

    def dereference(self, indexed: Any) -> Any:
        """Invert :meth:`Geometry.index_into`: a nested index structure → coordinate tuples.

        Each leaf integer becomes its (dequantized) coordinate; the nesting is preserved, so the
        result equals the geometry's original ``boundaries`` for a lossless pool, and equals them to
        within the quantization step for a pool built with a ``precision``.

        Args:
            indexed: A vertex index, or an arbitrarily nested list of them (a geometry's indexed
                boundaries, as :meth:`Geometry.index_into` returns).

        Returns:
            The same structure with each index replaced by its coordinate tuple — a single tuple
            when ``indexed`` is one index.

        Examples:
            - Dereference a geometry's indexed boundaries back to coordinates:
                ```python
                >>> pool = VertexPool(
                ...     vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)])
                >>> pool.dereference([[[0, 1, 2]]])
                [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]]

                ```
            - A lone index dereferences to a single coordinate:
                ```python
                >>> VertexPool(vertices=[(2.0, 3.0)]).dereference(0)
                (2.0, 3.0)

                ```
        """
        if isinstance(indexed, int):
            return self.dequantize(self.vertices[indexed])
        return [self.dereference(child) for child in indexed]


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


def _require_uniform_dimension(seen: int | None, got: int) -> int:
    """Return the pool's coordinate dimension, raising if ``got`` disagrees with ``seen``.

    A CityJSON ``vertices`` array must have one dimension throughout; a model can carry
    mixed-dimension coordinates (e.g. a CityGML source with per-element ``srsDimension``), so the
    pool rejects them rather than emit a non-uniform, non-serializable array.
    """
    if seen is not None and got != seen:
        raise CityModelError(
            f"cannot build a vertex pool from mixed-dimension coordinates ({seen}D and {got}D); "
            "every CityJSON vertex must have the same dimension"
        )
    return seen if seen is not None else got


def _iter_coordinates(element: Any) -> Iterator[tuple[float, ...]]:
    """Walk a boundaries structure to coordinate level, yielding each coordinate tuple."""
    if _is_coordinate(element):
        yield tuple(float(component) for component in element)
    elif isinstance(element, list):
        for child in element:
            yield from _iter_coordinates(child)


def _map_coordinates(element: Any, replace: Any) -> Any:
    """Rebuild a boundaries structure with ``replace`` applied to each coordinate, nesting intact."""
    if _is_coordinate(element):
        return replace(element)
    return [_map_coordinates(child, replace) for child in element]


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

    Examples:
        - Walk a ``MultiSurface`` and read each surface's semantic value:
            ```python
            >>> surface_a = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> surface_b = [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
            >>> [value for _surface, value in iter_surfaces([surface_a, surface_b], [0, 1])]
            [0, 1]

            ```
        - Descend a ``Solid``'s shell to reach its surfaces, nesting the values in step:
            ```python
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> pairs = list(iter_surfaces([[surface, surface]], [[7, 9]]))
            >>> [value for _surface, value in pairs]
            [7, 9]

            ```
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
