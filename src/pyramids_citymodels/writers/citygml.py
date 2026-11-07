"""Serialize the IR to CityGML 2.0 (issue #26) — the mirror of :mod:`.readers.citygml`.

This is the hard writer. CityGML is schema-heavy GML/XML, and the reader is deliberately lossy
(issue #24): it reconstructs the LOD2 *semantic* model — ``Building → BuildingPart → boundary
surface → gml:Polygon`` — matching elements by local name across CityGML 1.0/2.0/3.0, and in doing
so drops ``gml:id``\\ s it does not reuse, attribute ``uom`` units and the source XML structure.
The writer targets exactly what that reader rebuilds, so the round-trip fidelity contract is:

- **Preserved:** each object's id (written as ``gml:id``), its type (``Building`` / ``BuildingPart``),
  the parent/child hierarchy (children nested in ``bldg:consistsOfBuildingPart``), every boundary
  surface's geometry (rings, holes, coordinates) across ``MultiSurface`` and ``Solid``/``MultiSolid``
  alike, its semantic type, and the CRS. A plain (non-semantic) ``Solid`` round-trips **as a**
  ``Solid`` (emitted as ``bldg:lod{n}Solid``), and a surface's **material** (``app:X3DMaterial``, per
  ``app:theme``) is written as an ``app:Appearance`` targeting that polygon's assigned ``gml:id``, so
  materials read from CityJSON or CityGML survive the round-trip. The **version** round-trips for a
  ``1.0`` or ``2.0`` source
  (whose LOD2 building structure this writer matches element-for-element): the source version's
  namespaces are emitted. A ``3.0`` or unknown source is emitted as ``2.0`` (its version does *not*
  round-trip) — 3.0 restructures the geometry model, which this writer does not reproduce.
- **Best-effort:** attribute *values* keep their Python type via ``gen:*Attribute`` kinds
  (``int`` → ``gen:intAttribute``, ``float`` → ``gen:doubleAttribute``, else
  ``gen:stringAttribute``); the known numeric building attributes (``measuredHeight``, …) are
  written as their native ``bldg:`` elements. Rings are emitted closed (first position repeated) so
  the ``gml:LinearRing`` is schema-valid even though the IR (CityJSON-style) carries open rings. A
  CityGML→CityGML round-trip is therefore exact only for a **spec-conformant source, whose rings are
  already closed** (re-closing is then a no-op); an open-ring source gains the required closing
  position, so its re-parsed rings carry one extra coordinate. Real GML ``LinearRing``\\ s are always
  closed, so this affects only hand-authored open-ring input, not conformant CityGML.
- **Lossy:** attribute ``uom`` units, original ``gml:id``\\ s of surfaces/polygons (regenerated),
  a geometry whose surfaces carry *no* semantic type (written as a plain building geometry, which
  this package's semantic reader does not read back), and any non-``Building``/``BuildingPart``
  object type. Such a type (``Bridge``, ``CityFurniture``, … — now common since the CityGML reader
  reads them) is written under the ``bldg:`` namespace with a warning, e.g. ``<bldg:CityFurniture>``,
  which is **not schema-valid** (city furniture belongs to the ``frn:`` namespace): the geometry and
  attributes are preserved for a CityJSON/JSON-FG/mesh target, but a CityGML→CityGML round-trip of a
  non-building object is best-effort, not conformant (emitting the correct thematic namespace per
  type is a follow-up). CityGML **textures** (``app:ParameterizedTexture``) are not written yet
  (materials are) — a documented follow-up (#44). A ``MultiSolid`` is written as its surfaces (its
  N-solid partitioning is not preserved), though it is never mislabelled as a single ``Solid``.

It writes with the standard-library :mod:`xml.etree.ElementTree` (no ``lxml``, matching the reader),
so there is no new dependency and no optional ``extra``.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import warnings
from typing import Any
from xml.etree import ElementTree

from pyramids_citymodels.ir import (
    CityModel,
    CityObject,
    Geometry,
    is_coordinate,
    iter_surfaces,
)

__all__ = ["write_citygml"]

#: CityGML versions whose LOD2 building structure this writer matches element-for-element. A source
#: in one of these is re-emitted under its own namespaces (so the version string round-trips); any
#: other version (e.g. 3.0, which restructures geometry) is emitted as 2.0.
_MATCHED_VERSIONS = frozenset({"1.0", "2.0"})

#: Object types the CityGML reader reconstructs. A top-level object of any other type is written but
#: not read back by this package's semantic reader, so writing one warns rather than losing it silently.
_READABLE_OBJECT_TYPES = frozenset({"Building", "BuildingPart"})

#: The GML namespace, constant across CityGML versions this writer emits.
_GML_NS = "http://www.opengis.net/gml"


def _namespaces(version: str) -> dict[str, str]:
    """The ``prefix → URI`` map for a CityGML ``version`` (``core``/``bldg``/``gen`` carry it).

    Returned fresh per call and threaded through element construction, so the writer holds **no**
    mutable module state — concurrent writes of different-version models cannot corrupt each other.
    """
    return {
        "core": f"http://www.opengis.net/citygml/{version}",
        "bldg": f"http://www.opengis.net/citygml/building/{version}",
        "gen": f"http://www.opengis.net/citygml/generics/{version}",
        "app": f"http://www.opengis.net/citygml/appearance/{version}",
        "gml": _GML_NS,
    }


#: The default (CityGML 2.0) namespace map. Its prefixes are registered with ElementTree once at
#: import (never per call), so a 2.0 document serializes with friendly ``core:``/``bldg:``/``gml:``
#: prefixes; a 1.0 document carries the same (correct) URIs under auto-assigned prefixes.
_NS = _namespaces("2.0")
for _prefix, _uri in _NS.items():
    ElementTree.register_namespace(_prefix, _uri)

#: Known numeric building attributes written as native ``bldg:`` elements (the reader coerces these
#: back to numbers by name), mirroring the reader's ``_FLOAT``/``_INT_BUILDING_ATTRIBUTES`` sets.
_NATIVE_BUILDING_ATTRIBUTES = frozenset(
    {
        "measuredHeight",
        "yearOfConstruction",
        "storeysAboveGround",
        "storeysBelowGround",
    }
)


def _q(ns: dict[str, str], prefix: str, local: str) -> str:
    """A Clark-notation ``{namespace}local`` tag, resolving ``prefix`` against the per-call map ``ns``."""
    return f"{{{ns[prefix]}}}{local}"


def _coordinate(point: Any, dimension: int) -> list[float]:
    """Normalize a coordinate to exactly ``dimension`` components (pad with 0.0, truncate extra)."""
    values = [float(component) for component in point]
    if len(values) < dimension:
        values = values + [0.0] * (dimension - len(values))
    return values[:dimension]


#: X3DMaterial colour members ([r, g, b] floats) and scalar members, mirroring the reader's sets.
_MATERIAL_COLORS = ("diffuseColor", "emissiveColor", "specularColor")
_MATERIAL_SCALARS = ("ambientIntensity", "shininess", "transparency")


class _MaterialTargets:
    """Collects, while geometry is emitted, which polygon ``gml:id``\\ s each material paints per theme.

    The CityGML writer assigns every emitted ``gml:Polygon`` a ``gml:id`` and, for a painted surface,
    records that id under the material's theme; :meth:`appearance_members` then builds the
    ``app:Appearance`` elements that target those ids — so a CityJSON material survives ``to_citygml``.
    """

    def __init__(self, materials: list[dict[str, Any]]):
        self._materials = materials
        self._targets: dict[str, dict[int, list[str]]] = {}
        self._counter = 0

    def paint(
        self,
        ns: dict[str, str],
        polygon: ElementTree.Element | None,
        material_by_theme: dict[str, int],
    ) -> None:
        """Give ``polygon`` a fresh ``gml:id`` and record it as a target of each theme's material."""
        if polygon is None:
            return
        self._counter += 1
        polygon_id = f"poly_{self._counter}"
        polygon.set(_q(ns, "gml", "id"), polygon_id)
        for theme, index in material_by_theme.items():
            self._targets.setdefault(theme, {}).setdefault(index, []).append(polygon_id)

    def appearance_members(self, ns: dict[str, str]) -> list[ElementTree.Element]:
        """One ``app:appearanceMember``/``app:Appearance`` element per painted theme (``[]`` if none)."""
        members: list[ElementTree.Element] = []
        for theme, by_index in self._targets.items():
            member = ElementTree.Element(_q(ns, "app", "appearanceMember"))
            appearance = ElementTree.SubElement(member, _q(ns, "app", "Appearance"))
            theme_element = ElementTree.SubElement(appearance, _q(ns, "app", "theme"))
            theme_element.text = theme
            for index, polygon_ids in by_index.items():
                material = (
                    self._materials[index] if index < len(self._materials) else {}
                )
                surface_data = ElementTree.SubElement(
                    appearance, _q(ns, "app", "surfaceDataMember")
                )
                self._x3d_material(ns, surface_data, material, polygon_ids)
            members.append(member)
        return members

    def _x3d_material(
        self,
        ns: dict[str, str],
        surface_data: ElementTree.Element,
        material: dict[str, Any],
        polygon_ids: list[str],
    ) -> None:
        """Build one ``app:X3DMaterial`` with ``material``'s members and an ``app:target`` per id."""
        x3d = ElementTree.SubElement(surface_data, _q(ns, "app", "X3DMaterial"))
        name = material.get("name")
        if name:
            x3d.set(_q(ns, "gml", "id"), str(name))
        for member_name in _MATERIAL_COLORS:
            if member_name in material:
                element = ElementTree.SubElement(x3d, _q(ns, "app", member_name))
                element.text = " ".join(repr(float(c)) for c in material[member_name])
        for member_name in _MATERIAL_SCALARS:
            if member_name in material:
                element = ElementTree.SubElement(x3d, _q(ns, "app", member_name))
                element.text = repr(float(material[member_name]))
        if "isSmooth" in material:
            element = ElementTree.SubElement(x3d, _q(ns, "app", "isSmooth"))
            element.text = "true" if material["isSmooth"] else "false"
        for polygon_id in polygon_ids:
            target = ElementTree.SubElement(x3d, _q(ns, "app", "target"))
            target.text = f"#{polygon_id}"


def _surface_materials(geometry: Geometry) -> list[dict[str, int]]:
    """Per-surface ``{theme: material index}`` for a geometry, in :func:`iter_surfaces` order.

    Walks each theme's material ``values`` (nested parallel to the boundaries, like the semantic
    values) so the material index for surface *i* lines up with the *i*-th surface the writer emits;
    an unpainted (``None``) surface contributes no entry.
    """
    material_map = (geometry.appearance or {}).get("material") or {}
    count = sum(1 for _ in iter_surfaces(geometry.boundaries, geometry.semantic_values))
    per_surface: list[dict[str, int]] = [{} for _ in range(count)]
    for theme, spec in material_map.items():
        values = (spec or {}).get("values")
        if values is None:
            continue
        for position, (_surface, index) in enumerate(
            iter_surfaces(geometry.boundaries, values)
        ):
            if position < count and index is not None:
                per_surface[position][theme] = index
    return per_surface


def _pos_list(ring: list[Any]) -> tuple[str, int]:
    """The ``gml:posList`` text for a ring and its ``srsDimension``.

    The dimension is taken from the ring's first coordinate, and every coordinate is normalized to
    it, so the whitespace-separated token stream chunks back into equal-length coordinates exactly
    as the reader expects. The ring is **closed** — a GML ``LinearRing`` requires its first position
    to be repeated as its last — so a CityJSON ring (never closed by spec) and a bare triangle (3
    positions) are emitted as valid closed rings of at least four positions.
    """
    dimension = len(ring[0]) if ring and len(ring[0]) >= 2 else 3
    coordinates = [_coordinate(point, dimension) for point in ring]
    if len(coordinates) >= 3 and coordinates[0] != coordinates[-1]:
        coordinates.append(list(coordinates[0]))
    tokens = [repr(component) for coordinate in coordinates for component in coordinate]
    return " ".join(tokens), dimension


def _polygon_element(
    ns: dict[str, str], surface: list[Any]
) -> ElementTree.Element | None:
    """Build a ``gml:Polygon`` (exterior ring first, then interior rings) from an IR surface.

    A ring needs at least three distinct positions to bound an area; such a ring is emitted closed
    (first position repeated), giving the four-position minimum a GML ``LinearRing`` requires.
    """
    rings = [ring for ring in surface if ring and len(ring) >= 3]
    if not rings:
        return None
    polygon = ElementTree.Element(_q(ns, "gml", "Polygon"))
    for position, ring in enumerate(rings):
        boundary = ElementTree.SubElement(
            polygon, _q(ns, "gml", "exterior" if position == 0 else "interior")
        )
        linear_ring = ElementTree.SubElement(boundary, _q(ns, "gml", "LinearRing"))
        pos_list = ElementTree.SubElement(linear_ring, _q(ns, "gml", "posList"))
        text, dimension = _pos_list(ring)
        pos_list.set("srsDimension", str(dimension))
        pos_list.text = text
    return polygon


def _bounded_by(
    ns: dict[str, str],
    semantic_type: str,
    surface: list[Any],
    lod: int,
    targets: _MaterialTargets,
    material_by_theme: dict[str, int],
) -> ElementTree.Element | None:
    """A ``bldg:boundedBy`` wrapping one boundary surface of ``semantic_type`` and its polygon."""
    polygon = _polygon_element(ns, surface)
    if polygon is None:
        return None
    targets.paint(ns, polygon, material_by_theme)
    bounded_by = ElementTree.Element(_q(ns, "bldg", "boundedBy"))
    boundary_surface = ElementTree.SubElement(bounded_by, _q(ns, "bldg", semantic_type))
    multi_surface_property = ElementTree.SubElement(
        boundary_surface, _q(ns, "bldg", f"lod{lod}MultiSurface")
    )
    multi_surface = ElementTree.SubElement(
        multi_surface_property, _q(ns, "gml", "MultiSurface")
    )
    surface_member = ElementTree.SubElement(
        multi_surface, _q(ns, "gml", "surfaceMember")
    )
    surface_member.append(polygon)
    return bounded_by


def _lod_number(geometry: Geometry) -> int:
    """The integer LOD for the ``lod{n}MultiSurface`` wrapper (``"2.2"`` → ``2``); default ``2``."""
    if geometry.lod is None:
        return 2
    try:
        return int(float(geometry.lod))
    except (TypeError, ValueError):
        return 2


def _add_attributes(
    ns: dict[str, str], feature: ElementTree.Element, attributes: dict[str, Any]
) -> None:
    """Append an object's attributes: native ``bldg:`` elements for the known numeric ones, else
    a typed ``gen:*Attribute`` that preserves the Python value type on read."""
    for name, value in attributes.items():
        if name in _NATIVE_BUILDING_ATTRIBUTES:
            element = ElementTree.SubElement(feature, _q(ns, "bldg", name))
            element.text = repr(value) if isinstance(value, float) else str(value)
            continue
        # bool is an int subclass — keep it as a string, not an intAttribute.
        if isinstance(value, bool):
            kind = "stringAttribute"
        elif isinstance(value, int):
            kind = "intAttribute"
        elif isinstance(value, float):
            kind = "doubleAttribute"
        else:
            kind = "stringAttribute"
        generic = ElementTree.SubElement(feature, _q(ns, "gen", kind))
        generic.set("name", name)
        value_element = ElementTree.SubElement(generic, _q(ns, "gen", "value"))
        value_element.text = repr(value) if isinstance(value, float) else str(value)


def _append_plain_surface(
    ns: dict[str, str],
    feature: ElementTree.Element,
    surface: list[Any],
    lod: int,
    targets: _MaterialTargets,
    material_by_theme: dict[str, int],
) -> None:
    """Append a surface with no semantic type as a bare ``bldg:lod{n}MultiSurface`` on the feature."""
    plain = _polygon_element(ns, surface)
    if plain is None:
        return
    targets.paint(ns, plain, material_by_theme)
    multi_property = ElementTree.SubElement(
        feature, _q(ns, "bldg", f"lod{lod}MultiSurface")
    )
    multi = ElementTree.SubElement(multi_property, _q(ns, "gml", "MultiSurface"))
    member = ElementTree.SubElement(multi, _q(ns, "gml", "surfaceMember"))
    member.append(plain)


def _append_solid(
    ns: dict[str, str],
    feature: ElementTree.Element,
    geometry: Geometry,
    lod: int,
    targets: _MaterialTargets,
    surface_materials: list[dict[str, int]],
) -> None:
    """Append a (non-semantic) ``Solid`` as a ``bldg:lod{n}Solid`` of one ``gml:CompositeSurface``.

    Emitting the surfaces inside a ``gml:Solid`` (rather than as separate plain ``MultiSurface``
    properties) preserves the geometry *type*, so a CityJSON ``Solid`` round-trips as a ``Solid``
    through CityGML (the reader reads a ``lodXSolid`` back as ``type="Solid"`` — §3 gap 2). Only a
    single ``Solid`` is handled here: a ``MultiSolid`` is **not** routed through this path (that would
    collapse its N solids into one shell and mislabel the type), so it falls through to the
    per-surface path in :func:`_append_geometry` and is written as its surfaces.
    """
    polygons: list[ElementTree.Element] = []
    for position, (surface, _value) in enumerate(
        iter_surfaces(geometry.boundaries, geometry.semantic_values)
    ):
        polygon = _polygon_element(ns, surface)
        if polygon is None:
            continue
        material = (
            surface_materials[position] if position < len(surface_materials) else {}
        )
        targets.paint(ns, polygon, material)
        polygons.append(polygon)
    if not polygons:
        return
    solid_property = ElementTree.SubElement(feature, _q(ns, "bldg", f"lod{lod}Solid"))
    solid = ElementTree.SubElement(solid_property, _q(ns, "gml", "Solid"))
    exterior = ElementTree.SubElement(solid, _q(ns, "gml", "exterior"))
    composite = ElementTree.SubElement(exterior, _q(ns, "gml", "CompositeSurface"))
    for polygon in polygons:
        member = ElementTree.SubElement(composite, _q(ns, "gml", "surfaceMember"))
        member.append(polygon)


def _append_geometry(
    ns: dict[str, str],
    feature: ElementTree.Element,
    geometry: Geometry,
    targets: _MaterialTargets,
) -> None:
    """Append one IR geometry to ``feature`` — a ``gml:Solid`` when it is a plain solid, else surfaces.

    A non-semantic ``Solid`` is emitted as a ``bldg:lod{n}Solid`` so its type survives the round-trip
    (§3 gap 2). Every other geometry is walked to surface level for ANY type via the shared IR
    traversal, each surface paired with its semantic value — so a semantic ``Solid``'s surfaces (nested
    one shell deeper) are still reached and emitted as ``bldg:boundedBy`` boundary surfaces, and a
    surface with no semantic type becomes a plain building ``MultiSurface``. A ``MultiSolid`` is **not**
    emitted as a single ``gml:Solid`` (that would collapse its solids and mislabel the type); it takes
    this per-surface path, so its type is not preserved (a documented limit) but it is never mislabelled.
    Each emitted polygon is registered with ``targets`` (its ``gml:id`` + any material painting it).
    """
    lod = _lod_number(geometry)
    surface_materials = _surface_materials(geometry)
    if geometry.type == "Solid" and not geometry.semantics:
        _append_solid(ns, feature, geometry, lod, targets, surface_materials)
        return
    for position, (surface, semantic_value) in enumerate(
        iter_surfaces(geometry.boundaries, geometry.semantic_values)
    ):
        material = (
            surface_materials[position] if position < len(surface_materials) else {}
        )
        surface_object = geometry.semantic_for(semantic_value)
        if surface_object is None:
            _append_plain_surface(ns, feature, surface, lod, targets, material)
            continue
        bounded_by = _bounded_by(
            ns, surface_object.type, surface, lod, targets, material
        )
        if bounded_by is not None:
            feature.append(bounded_by)


def _append_children(
    ns: dict[str, str],
    feature: ElementTree.Element,
    city_object: CityObject,
    model: CityModel,
    visited: set[str],
    targets: _MaterialTargets,
) -> None:
    """Nest each child object: a ``BuildingPart`` under ``bldg:consistsOfBuildingPart``, else attached."""
    for child_id in city_object.children:
        child = model.objects.get(child_id)
        if child is None or child_id in visited:
            continue
        child_element = _feature_element(ns, child, model, visited, targets)
        if child.type == "BuildingPart":
            consists = ElementTree.SubElement(
                feature, _q(ns, "bldg", "consistsOfBuildingPart")
            )
            consists.append(child_element)
        else:
            # A non-part child cannot nest in a building; attach it so its geometry is not lost,
            # though this package's reader will not re-link it as a child.
            feature.append(child_element)


def _feature_element(
    ns: dict[str, str],
    city_object: CityObject,
    model: CityModel,
    visited: set[str],
    targets: _MaterialTargets,
) -> ElementTree.Element:
    """Build one ``bldg:<type>`` feature element: its id, attributes, surfaces and nested children."""
    feature = ElementTree.Element(_q(ns, "bldg", city_object.type))
    feature.set(_q(ns, "gml", "id"), city_object.id)
    visited.add(city_object.id)
    _add_attributes(ns, feature, city_object.attributes)
    for geometry in city_object.geometries:
        _append_geometry(ns, feature, geometry, targets)
    _append_children(ns, feature, city_object, model, visited, targets)
    return feature


def _collect_points(element: Any, points: list[list[float]]) -> None:
    """Append every coordinate found anywhere in a boundaries structure to ``points``."""
    if is_coordinate(element):
        points.append([float(component) for component in element])
        return
    if isinstance(element, list):
        for child in element:
            _collect_points(child, points)


def _bounds(model: CityModel) -> tuple[list[float], list[float], int] | None:
    """The (lower, upper, dimension) bounding box over every coordinate in the model, or ``None``."""
    points: list[list[float]] = []
    for city_object in model.objects.values():
        for geometry in city_object.geometries:
            _collect_points(geometry.boundaries, points)
    if not points:
        return None
    dimension = max(len(point) for point in points)
    lower = [
        min(point[axis] for point in points if axis < len(point))
        for axis in range(dimension)
    ]
    upper = [
        max(point[axis] for point in points if axis < len(point))
        for axis in range(dimension)
    ]
    return lower, upper, dimension


def _add_envelope(
    ns: dict[str, str], root: ElementTree.Element, model: CityModel
) -> None:
    """Append a ``gml:boundedBy`` envelope carrying the CRS (as ``srsName``) and the bounding box."""
    bounds = _bounds(model)
    if bounds is None and not model.crs:
        return
    bounded_by = ElementTree.SubElement(root, _q(ns, "gml", "boundedBy"))
    envelope = ElementTree.SubElement(bounded_by, _q(ns, "gml", "Envelope"))
    if model.crs:
        envelope.set("srsName", model.crs)
    if bounds is not None:
        lower, upper, dimension = bounds
        envelope.set("srsDimension", str(dimension))
        lower_corner = ElementTree.SubElement(envelope, _q(ns, "gml", "lowerCorner"))
        lower_corner.text = " ".join(repr(component) for component in lower)
        upper_corner = ElementTree.SubElement(envelope, _q(ns, "gml", "upperCorner"))
        upper_corner.text = " ".join(repr(component) for component in upper)


def write_citygml(model: CityModel, path: Any = None) -> str | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to a CityGML 2.0 document.

    Reconstructs the LOD2 semantic model the reader understands: a ``core:CityModel`` of
    ``core:cityObjectMember`` buildings, each with its boundary surfaces grouped under
    ``bldg:boundedBy`` by semantic type and its parts nested under ``bldg:consistsOfBuildingPart``.
    See the module docstring for the full round-trip fidelity contract.

    Args:
        model: The city model to serialize.
        path: A filesystem path to write the document to, or ``None`` to return it as a string.

    Returns:
        The CityGML document as a ``str`` when ``path`` is ``None``; otherwise ``None``.

    Examples:
        - Round-trip the geometry and semantic type of a parsed CityGML building:
            ```python
            >>> from pyramids_citymodels.readers.citygml import parse_citygml
            >>> xml = '''<core:CityModel
            ...   xmlns:core="http://www.opengis.net/citygml/2.0"
            ...   xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
            ...   xmlns:gml="http://www.opengis.net/gml">
            ...   <core:cityObjectMember><bldg:Building gml:id="B">
            ...     <bldg:boundedBy><bldg:RoofSurface>
            ...       <bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>
            ...         <gml:Polygon><gml:exterior><gml:LinearRing>
            ...           <gml:posList srsDimension="3">0 0 1 1 0 1 1 1 1 0 0 1</gml:posList>
            ...         </gml:LinearRing></gml:exterior></gml:Polygon>
            ...       </gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>
            ...     </bldg:RoofSurface></bldg:boundedBy>
            ...   </bldg:Building></core:cityObjectMember>
            ... </core:CityModel>'''
            >>> out = write_citygml(parse_citygml(xml))
            >>> reparsed = parse_citygml(out)
            >>> reparsed.objects["B"].geometries[0].semantics[0].type
            'RoofSurface'

            ```
        - An object's id is written as a ``gml:id`` on its ``bldg:Building`` element:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject
            >>> xml = write_citygml(CityModel(objects={"H1": CityObject(id="H1", type="Building")}))
            >>> 'gml:id="H1"' in xml
            True

            ```
    """
    # Emit the source version's namespaces for a version this writer matches (1.0/2.0), so the
    # version string round-trips through the reader; a 3.0 or unknown source falls back to 2.0. The
    # map is a per-call local threaded through construction — no module-global is mutated, so
    # concurrent writes of different-version models cannot corrupt each other.
    version = model.version if model.version in _MATCHED_VERSIONS else "2.0"
    if model.version is not None and model.version not in _MATCHED_VERSIONS:
        # CityGML 3.0 restructures the geometry model, so emitting 3.0 namespaces with this writer's
        # 2.0 element structure would be invalid 3.0. The reader reads 3.0 (version-agnostic by local
        # name), but on write the model is downgraded to valid 2.0 — the version does not round-trip.
        warnings.warn(
            f"CityGML writer: a {model.version} source is written as valid CityGML 2.0 (this writer "
            "emits the 2.0 building structure), so the CityGML version does not round-trip",
            stacklevel=2,
        )
    ns = _namespaces(version)

    unreadable = sorted(
        {
            city_object.type
            for city_object in model.objects.values()
            if city_object.type not in _READABLE_OBJECT_TYPES
        }
    )
    if unreadable:
        warnings.warn(
            "CityGML writer: object type(s) "
            f"{', '.join(unreadable)} are written but not reconstructed by this package's "
            "CityGML reader (it reads only Building/BuildingPart), so they will not round-trip",
            stacklevel=2,
        )

    root = ElementTree.Element(_q(ns, "core", "CityModel"))
    _add_envelope(ns, root, model)

    materials = (model.appearances or {}).get("materials") or []
    targets = _MaterialTargets(materials)
    visited: set[str] = set()
    for city_object in model.objects.values():
        if city_object.parents:
            continue  # emitted inside its parent
        member = ElementTree.SubElement(root, _q(ns, "core", "cityObjectMember"))
        member.append(_feature_element(ns, city_object, model, visited, targets))

    # Appearances come after the city objects: each paints polygons by the gml:ids assigned above.
    for appearance_member in targets.appearance_members(ns):
        root.append(appearance_member)

    ElementTree.indent(root)
    document = ElementTree.tostring(root, encoding="unicode", xml_declaration=True)

    if path is None:
        return document
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return None
