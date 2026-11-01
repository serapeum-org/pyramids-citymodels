"""Serialize the IR to CityGML 2.0 (issue #26) — the mirror of :mod:`.readers.citygml`.

This is the hard writer. CityGML is schema-heavy GML/XML, and the reader is deliberately lossy
(issue #24): it reconstructs the LOD2 *semantic* model — ``Building → BuildingPart → boundary
surface → gml:Polygon`` — matching elements by local name across CityGML 1.0/2.0/3.0, and in doing
so drops ``gml:id``\\ s it does not reuse, attribute ``uom`` units and the source XML structure.
The writer targets exactly what that reader rebuilds, so the round-trip fidelity contract is:

- **Preserved:** each object's id (written as ``gml:id``), its type (``Building`` / ``BuildingPart``),
  the parent/child hierarchy (children nested in ``bldg:consistsOfBuildingPart``), every boundary
  surface's geometry (rings, holes, coordinates) and its semantic type, the CRS and the version.
- **Best-effort:** attribute *values* keep their Python type via ``gen:*Attribute`` kinds
  (``int`` → ``gen:intAttribute``, ``float`` → ``gen:doubleAttribute``, else
  ``gen:stringAttribute``); the known numeric building attributes (``measuredHeight``, …) are
  written as their native ``bldg:`` elements.
- **Lossy:** attribute ``uom`` units, original ``gml:id``\\ s of surfaces/polygons (regenerated),
  a geometry whose surfaces carry *no* semantic type (written as a plain building geometry, which
  this package's semantic reader does not read back), and any non-``Building`` object type.

It writes with the standard-library :mod:`xml.etree.ElementTree` (no ``lxml``, matching the reader),
so there is no new dependency and no optional ``extra``.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any
from xml.etree import ElementTree

from pyramids_citymodels.ir import CityModel, CityObject, Geometry

__all__ = ["write_citygml"]

#: CityGML 2.0 namespace URIs, by the prefix this writer registers for them.
_NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gen": "http://www.opengis.net/citygml/generics/2.0",
    "gml": "http://www.opengis.net/gml",
}

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


def _q(prefix: str, local: str) -> str:
    """A Clark-notation ``{namespace}local`` tag for an element in one of the known namespaces."""
    return f"{{{_NS[prefix]}}}{local}"


def _coordinate(point: Any, dimension: int) -> list[float]:
    """Normalize a coordinate to exactly ``dimension`` components (pad with 0.0, truncate extra)."""
    values = [float(component) for component in point]
    if len(values) < dimension:
        values = values + [0.0] * (dimension - len(values))
    return values[:dimension]


def _pos_list(ring: list[Any]) -> tuple[str, int]:
    """The ``gml:posList`` text for a ring and its ``srsDimension``.

    The dimension is taken from the ring's first coordinate, and every coordinate is normalized to
    it, so the whitespace-separated token stream chunks back into equal-length coordinates exactly
    as the reader expects.
    """
    dimension = len(ring[0]) if ring and len(ring[0]) >= 2 else 3
    tokens: list[str] = []
    for point in ring:
        tokens.extend(repr(component) for component in _coordinate(point, dimension))
    return " ".join(tokens), dimension


def _polygon_element(surface: list[Any]) -> ElementTree.Element | None:
    """Build a ``gml:Polygon`` (exterior ring first, then interior rings) from an IR surface."""
    rings = [ring for ring in surface if ring and len(ring) >= 3]
    if not rings:
        return None
    polygon = ElementTree.Element(_q("gml", "Polygon"))
    for position, ring in enumerate(rings):
        boundary = ElementTree.SubElement(
            polygon, _q("gml", "exterior" if position == 0 else "interior")
        )
        linear_ring = ElementTree.SubElement(boundary, _q("gml", "LinearRing"))
        pos_list = ElementTree.SubElement(linear_ring, _q("gml", "posList"))
        text, dimension = _pos_list(ring)
        pos_list.set("srsDimension", str(dimension))
        pos_list.text = text
    return polygon


def _bounded_by(
    semantic_type: str, surface: list[Any], lod: int
) -> ElementTree.Element | None:
    """A ``bldg:boundedBy`` wrapping one boundary surface of ``semantic_type`` and its polygon."""
    polygon = _polygon_element(surface)
    if polygon is None:
        return None
    bounded_by = ElementTree.Element(_q("bldg", "boundedBy"))
    boundary_surface = ElementTree.SubElement(bounded_by, _q("bldg", semantic_type))
    multi_surface_property = ElementTree.SubElement(
        boundary_surface, _q("bldg", f"lod{lod}MultiSurface")
    )
    multi_surface = ElementTree.SubElement(
        multi_surface_property, _q("gml", "MultiSurface")
    )
    surface_member = ElementTree.SubElement(multi_surface, _q("gml", "surfaceMember"))
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


def _add_attributes(feature: ElementTree.Element, attributes: dict[str, Any]) -> None:
    """Append an object's attributes: native ``bldg:`` elements for the known numeric ones, else
    a typed ``gen:*Attribute`` that preserves the Python value type on read."""
    for name, value in attributes.items():
        if name in _NATIVE_BUILDING_ATTRIBUTES:
            element = ElementTree.SubElement(feature, _q("bldg", name))
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
        generic = ElementTree.SubElement(feature, _q("gen", kind))
        generic.set("name", name)
        value_element = ElementTree.SubElement(generic, _q("gen", "value"))
        value_element.text = repr(value) if isinstance(value, float) else str(value)


def _feature_element(
    city_object: CityObject, model: CityModel, visited: set[str]
) -> ElementTree.Element:
    """Build one ``bldg:<type>`` feature element: its id, attributes, surfaces and nested children."""
    feature = ElementTree.Element(_q("bldg", city_object.type))
    feature.set(_q("gml", "id"), city_object.id)
    visited.add(city_object.id)

    _add_attributes(feature, city_object.attributes)

    for geometry in city_object.geometries:
        lod = _lod_number(geometry)
        values = geometry.semantic_values
        for index, surface in enumerate(geometry.boundaries):
            semantic_value = (
                values[index]
                if isinstance(values, list) and index < len(values)
                else None
            )
            surface_object = geometry.semantic_for(semantic_value)
            if surface_object is None:
                # No semantic type — this package's semantic reader cannot place it; emit it as a
                # plain building geometry (valid CityGML, documented as not round-tripped here).
                plain = _polygon_element(surface)
                if plain is not None:
                    multi_property = ElementTree.SubElement(
                        feature, _q("bldg", f"lod{lod}MultiSurface")
                    )
                    multi = ElementTree.SubElement(
                        multi_property, _q("gml", "MultiSurface")
                    )
                    member = ElementTree.SubElement(multi, _q("gml", "surfaceMember"))
                    member.append(plain)
                continue
            bounded_by = _bounded_by(surface_object.type, surface, lod)
            if bounded_by is not None:
                feature.append(bounded_by)

    for child_id in city_object.children:
        child = model.objects.get(child_id)
        if child is None or child_id in visited:
            continue
        child_element = _feature_element(child, model, visited)
        if child.type == "BuildingPart":
            consists = ElementTree.SubElement(
                feature, _q("bldg", "consistsOfBuildingPart")
            )
            consists.append(child_element)
        else:
            # A non-part child cannot nest in a building; attach it so its geometry is not lost,
            # though this package's reader will not re-link it as a child.
            feature.append(child_element)
    return feature


def _collect_points(element: Any, points: list[list[float]]) -> None:
    """Append every coordinate found anywhere in a boundaries structure to ``points``."""
    if (
        isinstance(element, (list, tuple))
        and 2 <= len(element) <= 4
        and all(isinstance(component, (int, float)) for component in element)
    ):
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


def _add_envelope(root: ElementTree.Element, model: CityModel) -> None:
    """Append a ``gml:boundedBy`` envelope carrying the CRS (as ``srsName``) and the bounding box."""
    bounds = _bounds(model)
    if bounds is None and not model.crs:
        return
    bounded_by = ElementTree.SubElement(root, _q("gml", "boundedBy"))
    envelope = ElementTree.SubElement(bounded_by, _q("gml", "Envelope"))
    if model.crs:
        envelope.set("srsName", model.crs)
    if bounds is not None:
        lower, upper, dimension = bounds
        envelope.set("srsDimension", str(dimension))
        lower_corner = ElementTree.SubElement(envelope, _q("gml", "lowerCorner"))
        lower_corner.text = " ".join(repr(component) for component in lower)
        upper_corner = ElementTree.SubElement(envelope, _q("gml", "upperCorner"))
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
    """
    for prefix, uri in _NS.items():
        ElementTree.register_namespace(prefix, uri)

    root = ElementTree.Element(_q("core", "CityModel"))
    _add_envelope(root, model)

    visited: set[str] = set()
    for city_object in model.objects.values():
        if city_object.parents:
            continue  # emitted inside its parent
        member = ElementTree.SubElement(root, _q("core", "cityObjectMember"))
        member.append(_feature_element(city_object, model, visited))

    ElementTree.indent(root)
    document = ElementTree.tostring(root, encoding="unicode", xml_declaration=True)

    if path is None:
        return document
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return None
