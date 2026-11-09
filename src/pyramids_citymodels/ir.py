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

import copy
import re
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pyramids_citymodels.errors import CityModelError

if TYPE_CHECKING:
    from pyramids_citymodels.options import ObjectFilter

__all__ = [
    "CityModel",
    "CityObject",
    "Geometry",
    "SemanticSurface",
    "VertexPool",
    "crs_from_identifier",
    "is_coordinate",
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

        Raises:
            TypeError: If :attr:`boundaries` is a bare coordinate rather than a nested list (a real
                geometry's boundaries are always nested, so this signals an out-of-contract value).
            CityModelError: Propagated from :meth:`VertexPool.index_of` if a coordinate's dimension
                disagrees with the pool's.

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
        indexed = _map_coordinates(self.boundaries, pool.index_of)
        # boundaries is a nested list (surface/shell of rings), so _map_coordinates returns a list;
        # a bare-coordinate boundaries is out of contract and would make the list[Any] return a lie.
        # Raise (rather than assert, which -O strips) so the contract holds in every interpreter mode.
        if not isinstance(indexed, list):
            raise TypeError(
                "geometry boundaries must be a nested list, not a bare coordinate"
            )
        return indexed


@dataclass
class CityObject:
    """One city object — a feature in the city model.

    Attributes:
        id: The object's identifier, unique within the file.
        type: The object class, e.g. ``"Building"``, ``"BuildingPart"``, ``"Bridge"``.
        attributes: The object's attributes, passed through verbatim.
        attribute_units: Optional unit of measure (``uom``) per attribute name, e.g.
            ``{"measuredHeight": "m"}`` — a side map so ``attributes`` stays a plain name→value dict.
        geometries: The object's geometries (often one per level of detail).
        parents: Ids of parent objects (e.g. a ``BuildingPart``'s ``Building``).
        children: Ids of child objects.
    """

    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    attribute_units: dict[str, str] = field(default_factory=dict)
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

    @classmethod
    def from_file(
        cls, path: Any, selector: "ObjectFilter | None" = None
    ) -> "CityModel":
        """Parse a city-model file into a :class:`CityModel`, choosing the reader by content.

        The symmetric counterpart of :meth:`to_file`: it dispatches to the CityJSON, CityGML or
        3D Tiles reader so conversion is a single expression that never touches the lossy
        ``FeatureCollection`` form (ADR-0005)::

            CityModel.from_file("building.gml").to_file("building.city.json")

        A ``.gml`` / ``.xml`` / ``.citygml`` suffix goes straight to the CityGML reader; any other
        file (``.json`` is shared by CityJSON, a 3D Tiles tileset and GeoJSON) is content-sniffed —
        CityJSON, then 3D Tiles, then CityGML.

        Args:
            path: A path to a CityJSON, CityGML or 3D Tiles file.
            selector: An optional :class:`~pyramids_citymodels.options.ObjectFilter` read-time
                subset. The CityJSON and CityGML readers apply it while parsing (skipping an
                object's geometry when it does not match); the 3D Tiles reader prunes tile subtrees
                by bbox and filters the decoded features, since its objects are synthesized from
                tiles. ``None`` reads everything.

        Returns:
            The parsed city model, containing only the objects the ``selector`` keeps.

        Raises:
            CityModelError: If no reader recognizes the file.

        Examples:
            - Read CityJSON content (chosen by content sniff) and inspect the model:
                ```python
                >>> doc = (
                ...     '{"type": "CityJSON", "version": "2.0", '
                ...     '"CityObjects": {"b": {"type": "Building"}}, "vertices": []}'
                ... )
                >>> model = CityModel.from_file(doc)
                >>> model.source_format
                'CityJSON'
                >>> list(model.objects)
                ['b']

                ```
            - CityGML content is recognized by its namespace, not a suffix:
                ```python
                >>> xml = '<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0"/>'
                >>> CityModel.from_file(xml).source_format
                'CityGML'

                ```
            - Subset to chosen ids at read time with an
              :class:`~pyramids_citymodels.options.ObjectFilter`:
                ```python
                >>> from pyramids_citymodels.options import ObjectFilter
                >>> doc = (
                ...     '{"type": "CityJSON", "version": "2.0", "vertices": [], '
                ...     '"CityObjects": {"a": {"type": "Building"}, "b": {"type": "Building"}}}'
                ... )
                >>> model = CityModel.from_file(doc, ObjectFilter.coerce(ids="a"))
                >>> list(model.objects)
                ['a']

                ```
        """
        from pyramids_citymodels.errors import CityModelError
        from pyramids_citymodels.readers.citygml import (
            looks_like_citygml,
            parse_citygml,
        )
        from pyramids_citymodels.readers.cityjson import (
            looks_like_cityjson,
            parse_cityjson,
        )
        from pyramids_citymodels.readers.threedtiles import (
            looks_like_3dtiles,
            parse_3dtiles,
        )

        name = str(path).lower()
        if name.endswith((".gml", ".xml", ".citygml")):
            return parse_citygml(path, selector)
        if looks_like_cityjson(path):
            return parse_cityjson(path, selector)
        if looks_like_3dtiles(path):
            # 3D Tiles objects are synthesized from decoded tiles rather than read from named
            # features, so the reader prunes tile subtrees by bbox and applies the rest of the
            # subset to the decoded features itself (see parse_3dtiles).
            return parse_3dtiles(path, selector)
        if looks_like_citygml(path):
            return parse_citygml(path, selector)
        raise CityModelError(
            f"no city-model reader recognizes {path!r} (not CityJSON, 3D Tiles or CityGML)"
        )

    def to_cityjson(self, path: Any = None) -> "dict[str, Any] | None":
        """Serialize this model to CityJSON 2.0 (see :func:`.writers.cityjson.write_cityjson`).

        Returns the CityJSON document ``dict`` when ``path`` is ``None``, else writes it and
        returns ``None``.

        Examples:
            - Serialize to a dict and read a city object straight out of it:
                ```python
                >>> doc = (
                ...     '{"type": "CityJSON", "version": "2.0", '
                ...     '"CityObjects": {"b": {"type": "Building"}}, "vertices": []}'
                ... )
                >>> out = CityModel.from_file(doc).to_cityjson()
                >>> out["type"]
                'CityJSON'
                >>> out["CityObjects"]["b"]["type"]
                'Building'

                ```
        """
        from pyramids_citymodels.writers.cityjson import write_cityjson

        return write_cityjson(self, path)

    def to_cityjsonseq(self, path: Any = None) -> str | None:
        """Serialize this model to a CityJSON Text Sequence (``.city.jsonl``).

        See :func:`.writers.cityjson.write_cityjsonseq`. Returns the sequence as a ``str`` when
        ``path`` is ``None``, else writes it and returns ``None``.

        Examples:
            - The first line is a CityJSON header, each following line a ``CityJSONFeature``:
                ```python
                >>> import json
                >>> doc = (
                ...     '{"type": "CityJSON", "version": "2.0", '
                ...     '"CityObjects": {"b": {"type": "Building"}}, "vertices": []}'
                ... )
                >>> lines = CityModel.from_file(doc).to_cityjsonseq().splitlines()
                >>> json.loads(lines[0])["type"]
                'CityJSON'
                >>> json.loads(lines[1])["id"]
                'b'

                ```
        """
        from pyramids_citymodels.writers.cityjson import write_cityjsonseq

        return write_cityjsonseq(self, path)

    def to_citygml(self, path: Any = None) -> str | None:
        """Serialize this model to CityGML 2.0 (see :func:`.writers.citygml.write_citygml`).

        Returns the document as a ``str`` when ``path`` is ``None``, else writes it and returns
        ``None``. See the writer's module docstring for the round-trip fidelity contract.

        Examples:
            - Each object's id is written as a ``gml:id`` inside a ``cityObjectMember``:
                ```python
                >>> doc = (
                ...     '{"type": "CityJSON", "version": "2.0", '
                ...     '"CityObjects": {"H1": {"type": "Building"}}, "vertices": []}'
                ... )
                >>> xml = CityModel.from_file(doc).to_citygml()
                >>> "cityObjectMember" in xml
                True
                >>> 'gml:id="H1"' in xml
                True

                ```
        """
        from pyramids_citymodels.writers.citygml import write_citygml

        return write_citygml(self, path)

    def to_jsonfg(self, path: Any = None) -> "dict[str, Any] | None":
        """Serialize this model to OGC Features and Geometries JSON (JSON-FG).

        See :func:`.writers.jsonfg.write_jsonfg`. A one-way GIS-exchange projection, not a
        round-trip. Returns the document ``dict`` when ``path`` is ``None``, else writes it.

        Examples:
            - Each city object becomes one JSON-FG feature carrying its type as ``featureType``:
                ```python
                >>> doc = (
                ...     '{"type": "CityJSON", "version": "2.0", '
                ...     '"CityObjects": {"b": {"type": "Building"}}, "vertices": []}'
                ... )
                >>> fg = CityModel.from_file(doc).to_jsonfg()
                >>> fg["type"]
                'FeatureCollection'
                >>> fg["features"][0]["featureType"]
                'Building'

                ```
        """
        from pyramids_citymodels.writers.jsonfg import write_jsonfg

        return write_jsonfg(self, path)

    def to_obj(self, path: Any = None) -> str | None:
        """Serialize this model to Wavefront OBJ (see :func:`.writers.obj.write_obj`).

        A plain-geometry mesh export (feature #14): each surface's exterior ring becomes one face.
        Returns the OBJ text as a ``str`` when ``path`` is ``None``, else writes it.

        Examples:
            - Each object's faces are grouped under an ``o`` marker:
                ```python
                >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
                ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
                >>> "o b" in model.to_obj().splitlines()
                True

                ```
        """
        from pyramids_citymodels.writers.obj import write_obj

        return write_obj(self, path)

    def to_gltf(self, path: Any = None) -> "dict[str, Any] | None":
        """Serialize this model to glTF 2.0 (see :func:`.writers.gltf.write_gltf`).

        A triangulated, positions-only mesh export (feature #14), embedded as a self-contained
        ``.gltf``. Returns the document ``dict`` when ``path`` is ``None``, else writes it.

        Examples:
            - A one-triangle model exports one triangle (three indices):
                ```python
                >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
                ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
                >>> model.to_gltf()["accessors"][1]["count"]
                3

                ```
        """
        from pyramids_citymodels.writers.gltf import write_gltf

        return write_gltf(self, path)

    def translate(self, offset: "tuple[float, ...]") -> "CityModel":
        """A new model with every coordinate shifted by ``offset`` per axis (geometry cleanup, #17).

        See :func:`.transform.translate`. Semantics and appearance are preserved.

        Examples:
            - Shift a surface's heights up by 10:
                ```python
                >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
                ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
                >>> model.translate((0.0, 0.0, 10.0)).objects["b"].geometries[0].boundaries[0][0][0]
                (0.0, 0.0, 10.0)

                ```
        """
        from pyramids_citymodels.transform import translate

        return translate(self, offset)

    def quantize(self, digits: int = 3) -> "CityModel":
        """A new model with every coordinate rounded to ``digits`` decimals (see :func:`.transform.quantize`)."""
        from pyramids_citymodels.transform import quantize

        return quantize(self, digits)

    def clean_rings(self) -> "CityModel":
        """A new model with consecutive duplicate ring vertices removed (see :func:`.transform.clean_rings`)."""
        from pyramids_citymodels.transform import clean_rings

        return clean_rings(self)

    def triangulate(self) -> "CityModel":
        """A new model with every surface fan-triangulated into triangles (see :func:`.transform.triangulate`).

        Keeps each triangle's semantic value; drops interior rings (holes) and per-geometry appearance.

        Examples:
            - A quad surface becomes two triangles:
                ```python
                >>> quad = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]]
                >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
                ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[quad])])})
                >>> len(model.triangulate().objects["b"].geometries[0].boundaries)
                2

                ```
        """
        from pyramids_citymodels.transform import triangulate

        return triangulate(self)

    def to_collada(self, path: Any = None) -> str | None:
        """Serialize this model to COLLADA 1.4.1 ``.dae`` (see :func:`.writers.collada.write_collada`).

        A triangulated, positions-only mesh export (feature #14). Returns the document string when
        ``path`` is ``None``, else writes it.

        Examples:
            - A one-triangle model yields one COLLADA triangle:
                ```python
                >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
                ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
                >>> 'triangles count="1"' in model.to_collada()
                True

                ```
        """
        from pyramids_citymodels.writers.collada import write_collada

        return write_collada(self, path)

    def to_kml(self, path: Any = None) -> str | None:
        """Serialize this model to KML (see :func:`.writers.kml.write_kml`).

        A Placemark per object, each a MultiGeometry of one Polygon per surface (feature #14).
        Coordinates are the model's own (no reprojection). Returns the KML string when ``path`` is
        ``None``, else writes it.

        Examples:
            - A one-surface building becomes a Placemark with a Polygon:
                ```python
                >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
                ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
                >>> "<Placemark>" in model.to_kml()
                True

                ```
        """
        from pyramids_citymodels.writers.kml import write_kml

        return write_kml(self, path)

    def to_3dtiles(
        self, directory: Any, max_objects_per_tile: int | None = None
    ) -> "dict[str, Any]":
        """Write this model as a 3D Tiles tileset into ``directory``.

        Writes ``tileset.json`` plus batched ``b3dm`` content (one batch per city object, with a batch
        table of each object's id/type/attributes) — the tiling/packaging export (features #15/#51).
        By default a single tile; with ``max_objects_per_tile`` set, a model with more objects is
        subdivided into an XY quadtree of tiles. See :func:`.writers.threedtiles.write_3dtiles` for the
        format and its documented limits. 3D Tiles is multi-file, so a directory is required.

        Args:
            directory: The output directory (created if missing).
            max_objects_per_tile: When set, subdivide into tiles of at most this many objects;
                ``None`` (default) writes a single tile.

        Returns:
            The written ``tileset.json`` document.

        Examples:
            - A one-building model writes a tileset pointing at its b3dm content:
                ```python
                >>> import tempfile
                >>> from pathlib import Path
                >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
                ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
                >>> tileset = model.to_3dtiles(Path(tempfile.mkdtemp()))
                >>> tileset["root"]["content"]["uri"]
                'content.b3dm'
                >>> tileset["asset"]["gltfUpAxis"]  # IR is z-up
                'Z'

                ```
        """
        from pyramids_citymodels.writers.threedtiles import write_3dtiles

        return write_3dtiles(self, directory, max_objects_per_tile)

    def to_file(self, path: Any) -> "dict[str, Any] | str | None":
        """Serialize this model to ``path``, choosing the writer by the path's suffix.

        The symmetric counterpart of :meth:`from_file`:

        - ``.city.json`` / ``.cityjson`` / ``.json`` → CityJSON (:meth:`to_cityjson`)
        - ``.city.jsonl`` / ``.cityjsonseq`` / ``.jsonl`` → CityJSON Text Sequence
          (:meth:`to_cityjsonseq`)
        - ``.gml`` / ``.xml`` / ``.citygml`` → CityGML (:meth:`to_citygml`)
        - ``.fg.json`` / ``.jsonfg`` → JSON-FG (:meth:`to_jsonfg`)
        - ``.obj`` → Wavefront OBJ (:meth:`to_obj`)
        - ``.gltf`` → glTF 2.0 (:meth:`to_gltf`)
        - ``.dae`` → COLLADA 1.4.1 (:meth:`to_collada`)
        - ``.kml`` → KML (:meth:`to_kml`)

        The CityJSON and CityGML suffixes are symmetric with :meth:`from_file`. The CityJSON Text
        Sequence (``.city.jsonl`` / ``.cityjsonseq`` / ``.jsonl``), JSON-FG (``.fg.json`` /
        ``.jsonfg``), OBJ (``.obj``), glTF (``.gltf``), COLLADA (``.dae``) and KML (``.kml``) suffixes
        are **write-only**: this package ships no reader for them, so :meth:`from_file` cannot read
        them back.

        Args:
            path: The output path; its suffix selects the format.

        Returns:
            ``None`` (the model is written to ``path``).

        Raises:
            CityModelError: If no writer matches the suffix.

        Examples:
            - Convert a CityJSON model to CityGML on disk and read it back, all by suffix:
                ```python
                >>> import os, tempfile
                >>> doc = (
                ...     '{"type": "CityJSON", "version": "2.0", '
                ...     '"CityObjects": {"b": {"type": "Building"}}, "vertices": []}'
                ... )
                >>> path = os.path.join(tempfile.mkdtemp(), "building.gml")
                >>> CityModel.from_file(doc).to_file(path) is None
                True
                >>> CityModel.from_file(path).objects["b"].type
                'Building'

                ```
            - An unknown suffix is rejected:
                ```python
                >>> from pyramids_citymodels.errors import CityModelError
                >>> try:
                ...     CityModel().to_file("model.dwg")
                ... except CityModelError as error:
                ...     print("cannot infer" in str(error))
                True

                ```
        """
        from pyramids_citymodels.errors import CityModelError

        name = str(path).lower()
        if name.endswith((".city.jsonl", ".cityjsonseq", ".jsonl")):
            return self.to_cityjsonseq(path)
        if name.endswith((".fg.json", ".jsonfg")):
            return self.to_jsonfg(path)
        if name.endswith((".gml", ".xml", ".citygml")):
            return self.to_citygml(path)
        if name.endswith(".obj"):
            return self.to_obj(path)
        if name.endswith(".gltf"):
            return self.to_gltf(path)
        if name.endswith(".dae"):
            return self.to_collada(path)
        if name.endswith(".kml"):
            return self.to_kml(path)
        if name.endswith((".city.json", ".cityjson", ".json")):
            return self.to_cityjson(path)
        raise CityModelError(
            f"cannot infer a city-model writer from the suffix of {path!r}; expected one of "
            ".city.json/.cityjson/.json, .city.jsonl/.cityjsonseq/.jsonl, .gml/.xml/.citygml, "
            ".fg.json/.jsonfg, .obj, .gltf, .dae, .kml"
        )

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

    def merge(self, *others: "CityModel") -> "CityModel":
        """Combine this model with ``others`` into one new model (feature #15).

        An IR-level union — pure format-level aggregation, no GIS — that returns a **new**
        :class:`CityModel` and mutates none of the inputs. Objects from every model are gathered in
        order (``self`` first). An id that collides with one already taken is re-keyed with a
        ``#2`` (``#3``, …) suffix, and that model's ``parents``/``children`` links are rewritten to
        the new ids so the hierarchy stays internally consistent.

        The CRS and version are taken from the first model that declares one; a later model whose
        declared CRS differs is a conflict — its objects are still merged, but a warning is raised
        because coordinates are **not** reprojected here (that is ``pyramids-core``'s job).
        ``source_format`` becomes the common value, or ``"mixed"`` when the inputs disagree.
        ``metadata`` and ``extensions`` are shallow-merged with earlier models winning on a key
        clash. Appearance pools are index-based and cannot be concatenated without re-indexing every
        geometry, so the merged model keeps the **first** non-empty ``appearances`` and warns if a
        later model's appearances are dropped.

        Args:
            *others: Further models to merge into a copy of this one, in order.

        Returns:
            A new :class:`CityModel` holding every input's objects under unique ids.

        Examples:
            - Merge two one-object models; the objects coexist under their own ids:
                ```python
                >>> a = CityModel(objects={"a": CityObject(id="a", type="Building")})
                >>> b = CityModel(objects={"b": CityObject(id="b", type="Bridge")})
                >>> sorted(a.merge(b).objects)
                ['a', 'b']

                ```
            - A colliding id is re-keyed and the clone's child link follows it:
                ```python
                >>> a = CityModel(objects={"x": CityObject(id="x", type="Building")})
                >>> b = CityModel(objects={
                ...     "x": CityObject(id="x", type="Building", children=["p"]),
                ...     "p": CityObject(id="p", type="BuildingPart", parents=["x"])})
                >>> merged = a.merge(b)
                >>> sorted(merged.objects)
                ['p', 'x', 'x#2']
                >>> merged.objects["x#2"].children
                ['p']
                >>> merged.objects["p"].parents
                ['x#2']

                ```
        """
        merged = CityModel(
            crs=self.crs,
            metadata=dict(self.metadata),
            version=self.version,
            source_format=self.source_format,
            # Deep-copy so the merged model is fully independent — consistent with the deep-copied
            # geometries that index into this pool (the nested materials/textures lists are shared
            # by a shallow dict() copy otherwise).
            appearances=copy.deepcopy(self.appearances),
            extensions=dict(self.extensions),
        )
        for model in (self, *others):
            remap = _merge_ids(model.objects, into=merged.objects)
            for object_id, city_object in model.objects.items():
                merged.objects[remap[object_id]] = CityObject(
                    id=remap[object_id],
                    type=city_object.type,
                    attributes=dict(city_object.attributes),
                    # Deep-copy so the merged model is fully independent: a later in-place edit to a
                    # merged geometry's boundaries/semantics must not leak into the source model.
                    geometries=copy.deepcopy(city_object.geometries),
                    parents=[remap.get(p, p) for p in city_object.parents],
                    children=[remap.get(c, c) for c in city_object.children],
                )
            if model is self:
                continue
            _reconcile_metadata(merged, model)
        return merged


def _merge_ids(
    objects: dict[str, "CityObject"], into: dict[str, "CityObject"]
) -> dict[str, str]:
    """Map each id in ``objects`` to a unique id not already in ``into`` (suffixing a clash).

    A clash gets the lowest free ``<id>#<n>`` (``#2``, ``#3``, …). The whole map is computed up
    front — reserving each chosen id against both the existing keys and ids assigned earlier in this
    same model — so links can be rewritten consistently before any object is inserted.
    """
    remap: dict[str, str] = {}
    taken = set(into)
    for object_id in objects:
        new_id = object_id
        suffix = 2
        while new_id in taken:
            new_id = f"{object_id}#{suffix}"
            suffix += 1
        remap[object_id] = new_id
        taken.add(new_id)
    return remap


def _reconcile_metadata(merged: "CityModel", model: "CityModel") -> None:
    """Fold a later model's CRS/version/format/metadata/extensions/appearances into ``merged``."""
    if merged.crs is None:
        merged.crs = model.crs
    elif model.crs is not None and model.crs != merged.crs:
        warnings.warn(
            f"merging models with different CRS ({merged.crs!r} and {model.crs!r}); coordinates "
            f"are not reprojected here, so the merged model keeps {merged.crs!r}",
            UserWarning,
            stacklevel=3,
        )
    if merged.version is None:
        merged.version = model.version
    if model.source_format != merged.source_format:
        merged.source_format = "mixed"
    for key, value in model.metadata.items():
        merged.metadata.setdefault(key, value)
    for key, value in model.extensions.items():
        merged.extensions.setdefault(key, value)
    if model.appearances and not merged.appearances:
        merged.appearances = copy.deepcopy(model.appearances)
    elif model.appearances:
        warnings.warn(
            "merging models that both carry appearances; appearance pools are index-based and are "
            "not concatenated, so the merged model keeps only the first model's appearances",
            UserWarning,
            stacklevel=3,
        )


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
        # pool's single coordinate dimension too: a transform fixes it up front (its scale/translate
        # length), and any preloaded vertices must agree with it and each other.
        self._index: dict[tuple[float, ...], int] = {
            vertex: position for position, vertex in enumerate(self.vertices)
        }
        self._dimension: int | None = (
            len(self.transform["scale"]) if self.transform is not None else None
        )
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
        small. With ``precision`` ``None`` the pool is lossless: :attr:`transform` is ``None`` and
        coordinates are pooled unchanged. Either way ``coordinates`` is scanned once to validate a
        single uniform dimension, so a mixed-dimension model fails fast here rather than later.

        Args:
            coordinates: The coordinate stream the pool will hold, e.g.
                :meth:`CityModel.coordinates`. Consumed once (for the dimension check, and to size
                the ``transform`` when ``precision`` is given).
            precision: Decimal places to quantize to (a non-negative integer), or ``None`` for a
                lossless pool.

        Raises:
            TypeError: If ``precision`` is neither ``None`` nor an ``int`` (a ``bool`` is rejected
                too, so ``True`` is not read as ``1``).
            ValueError: If ``precision`` is a negative integer.
            CityModelError: If ``coordinates`` mixes dimensions, since a CityJSON ``vertices`` array
                must be uniform.

        Returns:
            An empty :class:`VertexPool`: its :attr:`transform` is ``None`` for a lossless pool (or
            for a model with no coordinates), or ``{"scale", "translate"}`` sized to the
            coordinates' extent for a quantizing one.

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
        _validate_precision(precision)
        # Always scan the coordinates for a single uniform dimension, so a mixed-dimension model
        # fails fast at build time on both the lossless and the quantizing paths (not only later,
        # in index_of). The per-axis minimum is accumulated only when quantizing needs it.
        minima: list[float] = []
        dimension: int | None = None
        for coordinate in coordinates:
            dimension = _require_uniform_dimension(dimension, len(coordinate))
            if precision is not None:
                _accumulate_minimum(minima, coordinate)
        if precision is None or not minima:
            # Lossless, or no coordinates to quantize (an empty/geometry-less model): a transform
            # with empty scale/translate is invalid CityJSON, so carry none — stay lossless.
            return cls(transform=None)
        # The scale is the inexact binary double for 10**-precision; that is the CityJSON convention
        # and is exactly what quantize/dequantize invert, so it is not rounded (rounding here was a
        # no-op: round(10.0**-p, p) == 10.0**-p for every p).
        scale = 10.0**-precision
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

        Raises:
            CityModelError: If ``coordinate`` has a different dimension than vertices already in the
                pool; a CityJSON ``vertices`` array must be uniform.

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
        # Validate the dimension against the raw coordinate BEFORE quantizing, so a mismatch always
        # surfaces as CityModelError rather than a raw IndexError (over-dimension indexing the
        # transform's scale/translate) or a silently stored wrong-dimension vertex (under-dimension).
        self._dimension = _require_uniform_dimension(self._dimension, len(coordinate))
        key = self.quantize(coordinate)
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


def is_coordinate(value: Any) -> bool:
    """True when ``value`` is a single coordinate — a short (2–4) sequence of numbers.

    The one place the "is this a coordinate leaf, not a nesting level?" test lives, shared by the
    IR traversal here and by the writers (which both re-index boundaries and walk them for bounds).

    Examples:
        - A 3-number sequence is a coordinate; a list of coordinates (a ring) is not:
            ```python
            >>> is_coordinate((1.0, 2.0, 3.0))
            True
            >>> is_coordinate([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
            False

            ```
    """
    return (
        isinstance(value, (list, tuple))
        and 2 <= len(value) <= 4
        and all(isinstance(component, (int, float)) for component in value)
    )


def _is_ring(value: Any) -> bool:
    """True when ``value`` is a ring — a non-empty list whose first element is a coordinate."""
    return isinstance(value, list) and len(value) > 0 and is_coordinate(value[0])


def _is_surface(value: Any) -> bool:
    """True when ``value`` is a surface — a non-empty list whose first element is a ring."""
    return isinstance(value, list) and len(value) > 0 and _is_ring(value[0])


def _validate_precision(precision: int | None) -> None:
    """Reject a ``precision`` that is not ``None`` or a non-negative, non-bool ``int``.

    Raises:
        TypeError: If ``precision`` is neither ``None`` nor an ``int`` (``bool`` excluded).
        ValueError: If ``precision`` is a negative integer.
    """
    if precision is None:
        return
    # bool is an int subclass, so exclude it explicitly: precision=True must not mean 1 decimal.
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise TypeError(
            f"precision must be an int (decimal places) or None, got {type(precision).__name__}"
        )
    if precision < 0:
        raise ValueError(
            f"precision must be a non-negative integer (decimal places), got {precision}"
        )


def _accumulate_minimum(minima: list[float], coordinate: Sequence[float]) -> None:
    """Fold ``coordinate`` into ``minima`` in place, keeping the per-axis running minimum."""
    for axis, component in enumerate(coordinate):
        value = float(component)
        if axis < len(minima):
            minima[axis] = min(minima[axis], value)
        else:
            minima.append(value)


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


# Both walks treat a leaf as a coordinate via `is_coordinate`, whose ``2 <= len <= 4`` bound is a
# deliberate CityJSON assumption (2-D or 3-D points, optionally with a 4th component): a 1-component or
# >=5-component leaf is not a coordinate this package produces, so it is walked into / skipped rather
# than flagged. This also distinguishes a 2-element *ring* (its elements are tuples) from a 2-D
# *coordinate* (its elements are numbers).
def _iter_coordinates(element: Any) -> Iterator[tuple[float, ...]]:
    """Walk a boundaries structure to coordinate level, yielding each coordinate tuple."""
    if is_coordinate(element):
        yield tuple(float(component) for component in element)
    elif isinstance(element, list):
        for child in element:
            yield from _iter_coordinates(child)


def _map_coordinates(element: Any, replace: Any) -> Any:
    """Rebuild a boundaries structure with ``replace`` applied to each coordinate, nesting intact."""
    if is_coordinate(element):
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
