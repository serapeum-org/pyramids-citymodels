"""Parse CityGML LOD2 semantic buildings into the shared IR (issue #7).

CityGML is schema-heavy GML/XML: a ``bldg:Building`` has optional ``bldg:BuildingPart`` children,
each carrying **boundary surfaces** (``WallSurface``, ``RoofSurface``, ``GroundSurface``, …) that
in turn carry ``gml:Polygon`` rings. This reader reconstructs that LOD2 *semantic* model —
``Building → BuildingPart → boundary surface → polygon`` — because the raw geometry is already
reachable through GDAL's GML driver; the semantics are what is missing (issue #1384).

It parses with the standard-library ``xml.etree.ElementTree`` (no ``lxml``, no new dependency) and
matches elements by **local name**, ignoring the namespace, so one code path handles CityGML 1.0
(``.../1.0``), 2.0 (``.../2.0``) and 3.0 (``.../3.0``) rather than hardcoding a version's
namespace URIs. A building (or part) becomes one :class:`~pyramids_citymodels.ir.Geometry` of
type ``"MultiSurface"`` whose surfaces are its boundary polygons, each tagged with the local name
of its boundary-surface element as the semantic ``type``. A ``BuildingPart`` becomes its own city
object, linked to its parent building.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import io
import os
from typing import Any
from xml.etree import ElementTree

from pyramids_citymodels.ir import (
    CityModel,
    CityObject,
    Geometry,
    SemanticSurface,
    crs_from_identifier,
)

__all__ = ["looks_like_citygml", "parse_citygml"]

#: Local names of ``bldg`` boundary-surface elements that carry a semantic surface type.
_BOUNDARY_SURFACE_NAMES = frozenset(
    {
        "WallSurface",
        "RoofSurface",
        "GroundSurface",
        "ClosureSurface",
        "OuterCeilingSurface",
        "OuterFloorSurface",
        "CeilingSurface",
        "FloorSurface",
        "InteriorWallSurface",
    }
)

#: Building attributes coerced from text to a number so the IR carries the same types the
#: CityJSON reader does (which keeps JSON ``float``/``int``), rather than raw strings (SF5).
_FLOAT_BUILDING_ATTRIBUTES = frozenset({"measuredHeight"})
_INT_BUILDING_ATTRIBUTES = frozenset(
    {"yearOfConstruction", "storeysAboveGround", "storeysBelowGround"}
)


def _coerce_number(text: str, to: type) -> Any:
    """Parse ``text`` with ``to`` (``int``/``float``); return the raw string if it does not parse."""
    try:
        return to(text)
    except (TypeError, ValueError):
        return text


def _local(tag: Any) -> str:
    """The local name of an element tag, dropping any ``{namespace}`` prefix."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _detect_version(root: ElementTree.Element) -> str | None:
    """Detect the CityGML version (``"1.0"`` / ``"2.0"`` / ``"3.0"``) from the namespace.

    Reads the trailing segment of the first ``citygml`` namespace seen (e.g.
    ``http://www.opengis.net/citygml/2.0`` or ``.../citygml/building/2.0`` → ``"2.0"``), so the
    version is carried through the IR without hardcoding a release's assumptions (#13).
    """
    for element in root.iter():
        tag = element.tag
        if isinstance(tag, str) and tag.startswith("{"):
            namespace = tag[1:].split("}", 1)[0]
            # Match the official OGC CityGML namespace only, not any namespace that merely
            # contains the substring "citygml": a third-party ADE can embed its own version
            # (e.g. .../citygml/2.0/energy/1.0) and would otherwise be read as the CityGML
            # version (INF2). The trailing segment is the version for both core
            # (.../citygml/2.0) and module (.../citygml/building/2.0) namespaces.
            if "opengis.net/citygml/" in namespace.lower():
                tail = namespace.rstrip("/").rsplit("/", 1)[-1]
                if tail[:1].isdigit():
                    return tail
    return None


def _gml_id(element: ElementTree.Element) -> str | None:
    """The ``gml:id`` attribute of an element, matched by local name (version-agnostic)."""
    for key, value in element.attrib.items():
        if _local(key) == "id":
            return value
    return None


def _href(element: ElementTree.Element) -> str | None:
    """The ``xlink:href`` of an element, matched by local name, with any leading ``#`` stripped."""
    for key, value in element.attrib.items():
        if _local(key) == "href":
            return value.lstrip("#")
    return None


def looks_like_citygml(source: Any) -> bool:
    """Return ``True`` when ``source`` is a CityGML document.

    Sniffs the XML for a CityGML namespace or a ``CityModel`` / ``cityObjectMember`` element, so
    the driver's ``can_open`` claims real CityGML and declines a plain GML or arbitrary XML file.
    Never raises — a non-XML or unreadable file returns ``False``.

    Examples:
        - A CityGML-namespaced document is recognized; plain XML is declined:
            ```python
            >>> looks_like_citygml('<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0"/>')
            True
            >>> looks_like_citygml("<root><child/></root>")
            False

            ```
    """
    try:
        for _event, element in _iterparse_start(source, limit=200):
            tag = _local(element.tag)
            # ``CityModel`` / ``cityObjectMember`` are CityGML-specific element names; ``Building``
            # is deliberately NOT a marker, because a plain (non-CityGML) GML feature collection can
            # carry a ``Building`` and must not be hijacked from OGR (SF2). A ``bldg``-namespaced
            # Building is still recognized below, since its namespace contains ``citygml``.
            if tag in ("CityModel", "cityObjectMember"):
                return True
            namespace = (
                element.tag[1:].split("}", 1)[0] if element.tag.startswith("{") else ""
            )
            if "citygml" in namespace.lower():
                return True
    except (OSError, ElementTree.ParseError, TypeError, ValueError):
        return False
    return False


def _iterparse_start(source: Any, limit: int | None = None):
    """Yield ``("start", element)`` pairs, from a path or an open/stream source, up to ``limit``."""
    if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
        iterator = ElementTree.iterparse(source, events=("start",))
    elif hasattr(source, "read"):
        iterator = ElementTree.iterparse(source, events=("start",))
    elif isinstance(source, (bytes, bytearray)):
        # Accept bytes too, so can_open agrees with read/_read_tree on what it can sniff (INF5).
        iterator = ElementTree.iterparse(io.BytesIO(source), events=("start",))
    elif isinstance(source, str):
        root = ElementTree.fromstring(source)
        for index, element in enumerate(root.iter()):
            yield ("start", element)
            if limit is not None and index + 1 >= limit:
                return
        return
    else:
        raise TypeError(f"cannot read CityGML from {type(source).__name__}")
    for index, (event, element) in enumerate(iterator):
        yield (event, element)
        if limit is not None and index + 1 >= limit:
            return


def _read_tree(source: Any) -> ElementTree.Element:
    """Parse the whole document to a root element from a path, open file, bytes or string."""
    if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
        return ElementTree.parse(source).getroot()
    if hasattr(source, "read"):
        return ElementTree.parse(source).getroot()
    if isinstance(source, (bytes, bytearray)):
        return ElementTree.fromstring(source)
    if isinstance(source, str):
        return ElementTree.fromstring(source)
    raise TypeError(f"cannot read CityGML from {type(source).__name__}")


def _srs_dimension(element: ElementTree.Element, default: int) -> int:
    """The ``srsDimension`` attribute on an element, or ``default``."""
    for key, value in element.attrib.items():
        if _local(key) == "srsDimension":
            try:
                return int(value)
            except ValueError:
                return default
    return default


def _ring_from_linear_ring(
    linear_ring: ElementTree.Element, dimension: int
) -> list[tuple[float, ...]]:
    """Extract a ring's coordinates from a ``gml:LinearRing`` (``posList`` or repeated ``pos``)."""
    for child in linear_ring.iter():
        if _local(child.tag) == "posList" and child.text:
            dim = _srs_dimension(child, dimension)
            numbers = [float(token) for token in child.text.split()]
            return [tuple(numbers[i : i + dim]) for i in range(0, len(numbers), dim)]
    coordinates: list[tuple[float, ...]] = []
    for child in linear_ring.iter():
        if _local(child.tag) == "pos" and child.text:
            dim = _srs_dimension(child, dimension)
            numbers = [float(token) for token in child.text.split()]
            coordinates.append(tuple(numbers[:dim]))
    return coordinates


def _surfaces_from_polygon(
    polygon: ElementTree.Element, dimension: int
) -> list[list[tuple[float, ...]]]:
    """Return one surface (exterior ring first, then interior rings) from a ``gml:Polygon``."""
    exterior: list[tuple[float, ...]] = []
    interiors: list[list[tuple[float, ...]]] = []
    for child in polygon:
        local = _local(child.tag)
        if local == "exterior":
            for linear_ring in child.iter():
                if _local(linear_ring.tag) == "LinearRing":
                    exterior = _ring_from_linear_ring(linear_ring, dimension)
                    break
        elif local == "interior":
            for linear_ring in child.iter():
                if _local(linear_ring.tag) == "LinearRing":
                    interiors.append(_ring_from_linear_ring(linear_ring, dimension))
                    break
    return [exterior, *interiors] if exterior else []


def _collect_building_attributes(element: ElementTree.Element) -> dict[str, Any]:
    """Lift a building's attributes off its direct children.

    Every leaf (childless) element with text is captured under its local name, so unknown and
    third-party **ADE** attributes survive rather than being dropped by a fixed whitelist (#13);
    the known-numeric names are coerced to numbers to match the CityJSON reader's types. Elements
    with child elements are structural (``boundedBy``, ``lod2Solid``, …) and skipped, except the
    generic ``gen:*Attribute`` wrappers, which are read as name/value pairs below.

    The numeric coercion is best-effort: a value that does not parse (``measuredHeight`` of
    ``"tall"``, a ``gYear`` carrying a timezone, a ``storeys*`` of ``"3.0"``) is kept as the raw
    string, so type parity with the CityJSON reader is *approximate* — a number when parseable,
    else the original text. Only element text is captured; attribute metadata such as a
    ``measuredHeight`` ``uom`` unit is not. A **repeated** property (CityGML allows several
    ``bldg:function`` / ``bldg:usage``) keeps only its **last** occurrence — attributes are a flat
    ``dict``, not a multimap.
    """
    attributes: dict[str, Any] = {}
    for child in element:
        local = _local(child.tag)
        is_leaf = len(child) == 0
        if is_leaf and child.text and child.text.strip():
            text = child.text.strip()
            if local in _FLOAT_BUILDING_ATTRIBUTES:
                attributes[local] = _coerce_number(text, float)
            elif local in _INT_BUILDING_ATTRIBUTES:
                attributes[local] = _coerce_number(text, int)
            else:
                attributes[local] = text
        elif local in (
            "stringAttribute",
            "intAttribute",
            "doubleAttribute",
            "genericAttribute",
        ):
            name = child.attrib.get("name") or _gml_id(child)
            value_element = next(
                (
                    grandchild
                    for grandchild in child
                    if _local(grandchild.tag) == "value"
                ),
                None,
            )
            if name and value_element is not None and value_element.text:
                raw = value_element.text.strip()
                if local == "intAttribute":
                    attributes[name] = _coerce_number(raw, int)
                elif local == "doubleAttribute":
                    attributes[name] = _coerce_number(raw, float)
                else:
                    attributes[name] = raw
    return attributes


def _nearest_feature_id(
    element: ElementTree.Element,
    parent_of: dict[ElementTree.Element, ElementTree.Element],
    feature_ids: dict[ElementTree.Element, str],
) -> str | None:
    """Walk ancestors to the nearest Building/BuildingPart element and return its assigned id."""
    current = parent_of.get(element)
    while current is not None:
        if current in feature_ids:
            return feature_ids[current]
        current = parent_of.get(current)
    return None


def parse_citygml(source: Any) -> CityModel:
    """Parse a CityGML document's LOD2 buildings into a :class:`~pyramids_citymodels.ir.CityModel`.

    Args:
        source: A path to a ``.gml``/``.xml`` file, an open file, or an XML string/bytes.

    Returns:
        A city model with one :class:`~pyramids_citymodels.ir.CityObject` per ``Building`` and per
        ``BuildingPart``; each carries a single ``"MultiSurface"`` geometry whose surfaces are its
        boundary polygons, tagged with their boundary-surface semantic type.

    Note:
        Shared geometry (``xlink:href``) is resolved against a document-wide ``gml:id`` index, so a
        polygon referenced by href is attributed to **every** referencing feature (legitimate CityGML
        shared geometry). Within one feature a polygon reachable from two boundary surfaces is taken
        **once**, keeping the first (document-order) surface's semantic type.

    Examples:
        - Parse an in-memory CityGML building and read its reconstructed semantic surface:
            ```python
            >>> xml = '''<core:CityModel
            ...   xmlns:core="http://www.opengis.net/citygml/2.0"
            ...   xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
            ...   xmlns:gml="http://www.opengis.net/gml">
            ...   <core:cityObjectMember><bldg:Building gml:id="B">
            ...     <bldg:boundedBy><bldg:RoofSurface>
            ...       <bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>
            ...         <gml:Polygon><gml:exterior><gml:LinearRing>
            ...           <gml:posList srsDimension="3">0 0 1 1 0 1 1 1 1</gml:posList>
            ...         </gml:LinearRing></gml:exterior></gml:Polygon>
            ...       </gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>
            ...     </bldg:RoofSurface></bldg:boundedBy>
            ...   </bldg:Building></core:cityObjectMember>
            ... </core:CityModel>'''
            >>> model = parse_citygml(xml)
            >>> model.objects["B"].geometries[0].semantics[0].type
            'RoofSurface'

            ```
    """
    root = _read_tree(source)
    parent_of = {child: parent for parent in root.iter() for child in parent}

    # CRS: the first srsName seen anywhere (envelope or geometry).
    crs = None
    for element in root.iter():
        for key, value in element.attrib.items():
            if _local(key) == "srsName":
                crs = crs_from_identifier(value)
                break
        if crs:
            break

    default_dimension = 3

    # Assign a stable id to every Building / BuildingPart up front, so boundary surfaces can be
    # attributed to the nearest one by walking ancestors.
    feature_ids: dict[ElementTree.Element, str] = {}
    anonymous = 0
    for element in root.iter():
        if _local(element.tag) in ("Building", "BuildingPart"):
            identifier = _gml_id(element)
            if not identifier:
                identifier = f"{_local(element.tag)}_{anonymous}"
                anonymous += 1
            feature_ids[element] = identifier

    model = CityModel(crs=crs, source_format="CityGML", version=_detect_version(root))

    # Create a city object per feature, with parent/child links between a part and its building.
    for element, identifier in feature_ids.items():
        parent_feature_id = _nearest_feature_id(element, parent_of, feature_ids)
        city_object = CityObject(
            id=identifier,
            type=_local(element.tag),
            attributes=_collect_building_attributes(element),
            parents=[parent_feature_id] if parent_feature_id else [],
        )
        model.objects[identifier] = city_object
        if parent_feature_id:
            model.objects[parent_feature_id].children.append(identifier)

    # Index every polygon by its gml:id so a boundary surface that references its geometry by
    # xlink:href (shared-geometry / XLink graph) can resolve it instead of silently dropping it.
    polygons_by_id: dict[str, ElementTree.Element] = {}
    for node in root.iter():
        if _local(node.tag) == "Polygon":
            polygon_id = _gml_id(node)
            if polygon_id:
                polygons_by_id[polygon_id] = node

    # Gather boundary surfaces, attributing each to its nearest feature and tagging its semantics.
    surfaces_by_feature: dict[
        str, tuple[list[Any], list[SemanticSurface], list[int]]
    ] = {identifier: ([], [], []) for identifier in feature_ids.values()}
    # One polygon reachable from two boundary surfaces of a feature (inline in one, referenced by
    # xlink:href from another) must be collected ONCE, keeping the first (document-order) surface's
    # semantic type — so the seen-set is scoped to the feature, not reset per boundary surface (SF1).
    seen_by_feature: dict[str, set[int]] = {
        identifier: set() for identifier in feature_ids.values()
    }
    for element in root.iter():
        local = _local(element.tag)
        if local not in _BOUNDARY_SURFACE_NAMES:
            continue
        feature_id = _nearest_feature_id(element, parent_of, feature_ids)
        if feature_id is None:
            continue
        boundaries, semantics, values = surfaces_by_feature[feature_id]
        seen = seen_by_feature[feature_id]
        for node in element.iter():
            polygon: ElementTree.Element | None = None
            if _local(node.tag) == "Polygon":
                polygon = node
            else:
                referenced = _href(node)
                if referenced:
                    polygon = polygons_by_id.get(referenced)
            if polygon is None or id(polygon) in seen:
                continue
            seen.add(id(polygon))
            surface = _surfaces_from_polygon(polygon, default_dimension)
            if not surface or len(surface[0]) < 3:
                continue
            values.append(len(semantics))
            semantics.append(SemanticSurface(type=local))
            boundaries.append(surface)

    for identifier, (boundaries, semantics, values) in surfaces_by_feature.items():
        if boundaries:
            model.objects[identifier].geometries.append(
                Geometry(
                    type="MultiSurface",
                    lod="2",
                    boundaries=boundaries,
                    semantics=semantics,
                    semantic_values=values,
                )
            )

    return model
