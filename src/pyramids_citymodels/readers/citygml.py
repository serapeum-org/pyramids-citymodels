"""Parse CityGML city objects into the shared IR (issues #7, #40 reader breadth).

CityGML is schema-heavy GML/XML: a feature such as a ``bldg:Building`` (with optional
``bldg:BuildingPart`` children) carries geometry either through **boundary surfaces**
(``WallSurface``, ``RoofSurface``, ``GroundSurface``, …) that hold ``gml:Polygon`` rings, or
**directly** on the feature in a ``lodXMultiSurface`` / ``lodXSolid`` / ``lodXGeometry`` container.
This reader reconstructs both, because the semantics (and, for many object types, the geometry
itself) are what is missing once a file is flattened (issue #1384).

It parses with the standard-library ``xml.etree.ElementTree`` (no ``lxml``, no new dependency) and
matches elements by **local name**, ignoring the namespace, so one code path handles CityGML 1.0
(``.../1.0``), 2.0 (``.../2.0``) and 3.0 (``.../3.0``) rather than hardcoding a version's namespace
URIs. A feature becomes one :class:`~pyramids_citymodels.ir.Geometry` of type ``"MultiSurface"``
whose surfaces are its polygons: those under a boundary surface are tagged with that surface's local
name as the semantic ``type`` (and the geometry is LOD 2, the semantic-surface LOD); those taken
directly from a ``lodX`` container carry no semantic type and the geometry's LOD is read from the
container (``lod1Solid`` → ``"1"``). A nested part (``BuildingPart`` / ``BridgePart`` /
``TunnelPart``) becomes its own city object, linked to its parent.

Beyond buildings, the reader recognizes the common top-level city objects — bridges, tunnels, city
furniture, vegetation, water bodies, land use, relief and transportation (:data:`_CITY_OBJECT_NAMES`)
— so a whole city file, not only its LOD2 buildings, reaches the IR.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import io
import os
import re
from typing import Any
from xml.etree import ElementTree

from pyramids_citymodels.ir import (
    CityModel,
    CityObject,
    Geometry,
    SemanticSurface,
    crs_from_identifier,
)
from pyramids_citymodels.options import ObjectFilter

__all__ = ["looks_like_citygml", "parse_citygml"]

#: Default coordinate dimension assumed when a geometry declares no ``srsDimension``.
_DEFAULT_SRS_DIMENSION = 3

#: Local names of the top-level CityGML city objects this reader turns into city objects — buildings
#: and their relatives across the thematic modules. A nested part (``BuildingPart`` / ``BridgePart``
#: / ``TunnelPart``) is included so it becomes its own object linked to its parent. Boundary-surface
#: elements (:data:`_BOUNDARY_SURFACE_NAMES`) are deliberately *not* here: they are geometry carriers
#: of a feature, not features in their own right.
_CITY_OBJECT_NAMES = frozenset(
    {
        "Building",
        "BuildingPart",
        "Bridge",
        "BridgePart",
        "Tunnel",
        "TunnelPart",
        "CityFurniture",
        "SolitaryVegetationObject",
        "PlantCover",
        "WaterBody",
        "LandUse",
        "ReliefFeature",
        "TINRelief",
        "TransportationComplex",
        "Road",
        "Railway",
        "Track",
        "Square",
        "GenericCityObject",
        "CityObjectGroup",
    }
)

#: Local names of boundary-surface elements that carry a semantic surface type — building/bridge/
#: tunnel thematic surfaces plus the water-body ones. A polygon under one of these is tagged with
#: the surface's local name; a polygon reached directly from a ``lodX`` container is not.
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
        "WaterSurface",
        "WaterGroundSurface",
        "WaterClosureSurface",
    }
)

#: A ``lodX...`` geometry-container local name (``lod1Solid``, ``lod2MultiSurface``, …); group 1 is
#: the LOD digit, used to label a feature's directly-held geometry.
_LOD_CONTAINER_RE = re.compile(r"^lod(\d)", re.IGNORECASE)

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


def _envelope_corners(
    envelope: ElementTree.Element,
) -> "tuple[float, float, float, float] | None":
    """Read a ``gml:Envelope``'s ``lowerCorner``/``upperCorner`` as 2D ``(xmin, ymin, xmax, ymax)``.

    Returns ``None`` when either corner is missing or unparseable, so the bbox prefilter keeps the
    feature rather than acting on a half-read extent.
    """
    lower: str | None = None
    upper: str | None = None
    for node in envelope.iter():
        name = _local(node.tag)
        if name == "lowerCorner":
            lower = node.text
        elif name == "upperCorner":
            upper = node.text
    if not lower or not upper:
        return None
    try:
        low = [float(value) for value in lower.split()]
        high = [float(value) for value in upper.split()]
    except ValueError:
        return None
    if len(low) < 2 or len(high) < 2:
        return None
    return (low[0], low[1], high[0], high[1])


def _feature_envelope(
    element: ElementTree.Element,
) -> "tuple[float, float, float, float] | None":
    """The feature's *own* declared extent from a direct ``gml:boundedBy/gml:Envelope``, or ``None``.

    Only a ``boundedBy`` that is a direct child of ``element`` is consulted, so a nested
    ``BuildingPart``'s envelope is never mistaken for its parent's (which would be smaller and could
    wrongly exclude the parent). A ``bldg:boundedBy`` carrying boundary *surfaces* rather than an
    ``Envelope`` is skipped, since it contains no ``Envelope``.
    """
    for child in element:
        if _local(child.tag) != "boundedBy":
            continue
        for node in child.iter():
            if _local(node.tag) == "Envelope":
                return _envelope_corners(node)
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


def _detect_crs(root: ElementTree.Element) -> Any:
    """The CRS from the first ``srsName`` attribute seen anywhere (envelope or geometry), or ``None``."""
    for element in root.iter():
        for key, value in element.attrib.items():
            if _local(key) == "srsName":
                return crs_from_identifier(value)
    return None


def _citygml_candidate_features(
    root: ElementTree.Element, selector: "ObjectFilter | None"
) -> "tuple[list[tuple[ElementTree.Element, str | None]], dict[ElementTree.Element, dict[str, Any]]]":
    """The Building/BuildingPart features passing the metadata + bbox subset, with their attributes.

    A feature passes when it matches by id/type/attributes and is not proven outside the bbox by its
    own ``gml:Envelope`` (one with no envelope is kept and refined downstream), before any polygon
    is parsed (feature #5, #32). An anonymous feature (no ``gml:id``) is matched on an empty id, so
    an id filter naturally excludes it. Each feature's attributes are collected **once** here and
    returned for reuse when the ``CityObject`` is built, so a kept feature is never scanned twice.

    Returns:
        ``(candidates, attributes_by_element)`` — the surviving ``(element, gml_id)`` pairs in
        document order, and every candidate element mapped to its collected attributes.
    """
    candidates: list[tuple[ElementTree.Element, str | None]] = []
    attributes_by_element: dict[ElementTree.Element, dict[str, Any]] = {}
    for element in root.iter():
        if _local(element.tag) not in _CITY_OBJECT_NAMES:
            continue
        gml_identifier = _gml_id(element)
        attributes = _collect_building_attributes(element)
        if selector is not None and (
            not selector.matches(gml_identifier or "", _local(element.tag), attributes)
            or selector.rejects_extent(_feature_envelope(element))
        ):
            continue
        candidates.append((element, gml_identifier))
        attributes_by_element[element] = attributes
    return candidates, attributes_by_element


def _assign_feature_ids(
    root: ElementTree.Element, selector: "ObjectFilter | None"
) -> "tuple[dict[ElementTree.Element, str], dict[ElementTree.Element, dict[str, Any]]]":
    """Assign a stable id to every *kept* Building/BuildingPart, applying the read-time subset.

    Gathers the candidate features (:func:`_citygml_candidate_features`), thins them by any random
    sample keyed on document position so an anonymous feature participates too, then assigns ids to
    the survivors in document order. The anonymous counter advances only for kept features, so
    synthesized ids are stable for a *given* selector, not across selectors — a different
    sample/bbox can renumber an anonymous feature (which has no source identity).

    Returns:
        ``(feature_ids, attributes_by_element)`` — the kept elements mapped to their assigned id,
        and every candidate element mapped to its collected attributes.
    """
    candidates, attributes_by_element = _citygml_candidate_features(root, selector)
    if selector is None:
        kept_positions = set(range(len(candidates)))
    else:
        kept_positions = selector.sample_ids(range(len(candidates)))
    feature_ids: dict[ElementTree.Element, str] = {}
    anonymous = 0
    for position, (element, gml_identifier) in enumerate(candidates):
        if position not in kept_positions:
            continue
        identifier = gml_identifier
        if not identifier:
            identifier = f"{_local(element.tag)}_{anonymous}"
            anonymous += 1
        feature_ids[element] = identifier
    return feature_ids, attributes_by_element


def _index_polygons(
    root: ElementTree.Element,
) -> dict[str, ElementTree.Element]:
    """Index every ``gml:Polygon`` by its ``gml:id``, so an ``xlink:href`` boundary surface can resolve it."""
    polygons_by_id: dict[str, ElementTree.Element] = {}
    for node in root.iter():
        if _local(node.tag) == "Polygon":
            polygon_id = _gml_id(node)
            if polygon_id:
                polygons_by_id[polygon_id] = node
    return polygons_by_id


def _resolve_polygon(
    node: ElementTree.Element, polygons_by_id: dict[str, ElementTree.Element]
) -> "ElementTree.Element | None":
    """The ``gml:Polygon`` a node *is*, or the one it references via ``xlink:href``, or ``None``."""
    if _local(node.tag) == "Polygon":
        return node
    referenced = _href(node)
    return polygons_by_id.get(referenced) if referenced else None


def _under_boundary_surface(
    element: ElementTree.Element,
    parent_of: dict[ElementTree.Element, ElementTree.Element],
    feature_ids: dict[ElementTree.Element, str],
) -> bool:
    """Whether ``element`` sits inside a boundary-surface element, within its own feature.

    Walks ancestors up to (but not past) the nearest feature; returns ``True`` if a boundary-surface
    element is met first. Used to tell a feature's *direct* geometry (no boundary surface) from
    geometry owned by a semantic boundary surface, which is gathered separately.
    """
    current = parent_of.get(element)
    while current is not None and current not in feature_ids:
        if _local(current.tag) in _BOUNDARY_SURFACE_NAMES:
            return True
        current = parent_of.get(current)
    return False


def _lod_from_ancestors(
    element: ElementTree.Element,
    parent_of: dict[ElementTree.Element, ElementTree.Element],
    feature_ids: dict[ElementTree.Element, str],
) -> str | None:
    """The LOD digit of the nearest ``lodX`` container above ``element`` (within its feature), or ``None``."""
    current = parent_of.get(element)
    while current is not None and current not in feature_ids:
        match = _LOD_CONTAINER_RE.match(_local(current.tag))
        if match:
            return match.group(1)
        current = parent_of.get(current)
    return None


def _gather_surfaces(
    root: ElementTree.Element,
    parent_of: dict[ElementTree.Element, ElementTree.Element],
    feature_ids: dict[ElementTree.Element, str],
    polygons_by_id: dict[str, ElementTree.Element],
) -> "tuple[dict[str, tuple[list[Any], list[SemanticSurface], list[int | None]]], dict[str, str], dict[str, list[str | None]]]":
    """Collect each feature's polygons — boundary-surface (semantic) and direct (geometry) alike.

    Two passes. **Pass 1** walks boundary-surface elements and, resolving ``xlink:href`` against the
    polygon index, attributes each polygon to its nearest feature with that surface's local name as
    its semantic type (LOD 2, the semantic-surface level). **Pass 2** walks every other polygon — a
    feature's *directly*-held geometry (``lodXMultiSurface`` / ``lodXSolid`` / ``lodXGeometry`` on
    city furniture, vegetation, water, relief, …) — attributing it to its nearest feature with no
    semantic type and recording the feature's LOD from the ``lodX`` container. A polygon is collected
    once per feature (``seen`` is feature-scoped, so a polygon shared between two boundary surfaces
    keeps the first surface's type — SF1); pass 2 skips polygons pass 1 already took.

    Returns:
        ``(surfaces_by_feature, lod_by_feature, polygon_ids_by_feature)`` — each feature's
        ``(boundaries, semantics, values)`` (``values`` entries are an index into ``semantics`` or
        ``None`` for a direct surface), the LOD inferred for features that carry only direct
        geometry, and the source ``gml:id`` of each collected polygon (parallel to ``boundaries``,
        ``None`` for an anonymous one) so appearance targets can be matched to surfaces.
    """
    surfaces_by_feature: dict[
        str, tuple[list[Any], list[SemanticSurface], list[int | None]]
    ] = {identifier: ([], [], []) for identifier in feature_ids.values()}
    seen_by_feature: dict[str, set[int]] = {
        identifier: set() for identifier in feature_ids.values()
    }
    polygon_ids_by_feature: dict[str, list[str | None]] = {
        identifier: [] for identifier in feature_ids.values()
    }
    lod_by_feature: dict[str, str] = {}

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
            polygon = _resolve_polygon(node, polygons_by_id)
            if polygon is None or id(polygon) in seen:
                continue
            seen.add(id(polygon))
            surface = _surfaces_from_polygon(polygon, _DEFAULT_SRS_DIMENSION)
            if not surface or len(surface[0]) < 3:
                continue
            values.append(len(semantics))
            semantics.append(SemanticSurface(type=local))
            boundaries.append(surface)
            polygon_ids_by_feature[feature_id].append(_gml_id(polygon))

    for polygon in root.iter():
        if _local(polygon.tag) != "Polygon":
            continue
        if _under_boundary_surface(polygon, parent_of, feature_ids):
            continue  # pass 1 owns boundary-surface polygons
        feature_id = _nearest_feature_id(polygon, parent_of, feature_ids)
        if feature_id is None:
            continue
        boundaries, semantics, values = surfaces_by_feature[feature_id]
        seen = seen_by_feature[feature_id]
        if id(polygon) in seen:
            continue
        seen.add(id(polygon))
        surface = _surfaces_from_polygon(polygon, _DEFAULT_SRS_DIMENSION)
        if not surface or len(surface[0]) < 3:
            continue
        values.append(None)  # direct geometry carries no semantic surface
        boundaries.append(surface)
        polygon_ids_by_feature[feature_id].append(_gml_id(polygon))
        lod_by_feature.setdefault(
            feature_id,
            _lod_from_ancestors(polygon, parent_of, feature_ids) or "",
        )
    return surfaces_by_feature, lod_by_feature, polygon_ids_by_feature


#: X3DMaterial children read as colour vectors ([r, g, b] floats) and as scalar floats.
_MATERIAL_COLORS = frozenset({"diffuseColor", "emissiveColor", "specularColor"})
_MATERIAL_SCALARS = frozenset({"ambientIntensity", "shininess", "transparency"})


def _parse_x3d_material(
    element: ElementTree.Element,
) -> "tuple[dict[str, Any], list[str]]":
    """Read a ``app:X3DMaterial`` into a CityJSON-shaped material dict plus its target polygon ids.

    Colour members become ``[r, g, b]`` float lists and the scalar members floats, matching the
    CityJSON material object; ``app:target`` references (``#poly_id``) become the surfaces the
    material paints. The element's ``gml:id`` seeds the material ``name`` when present.
    """
    material: dict[str, Any] = {}
    targets: list[str] = []
    for child in element:
        name = _local(child.tag)
        text = (child.text or "").strip()
        if name == "target" and text:
            targets.append(text.lstrip("#"))
        elif name in _MATERIAL_COLORS and text:
            try:
                material[name] = [float(component) for component in text.split()]
            except ValueError:
                pass
        elif name in _MATERIAL_SCALARS and text:
            material[name] = _coerce_number(text, float)
        elif name == "isSmooth" and text:
            material["isSmooth"] = text.lower() in ("true", "1")
    gml_identifier = _gml_id(element)
    if gml_identifier:
        material.setdefault("name", gml_identifier)
    return material, targets


def _gather_appearances(
    root: ElementTree.Element,
    polygon_ids_by_feature: dict[str, list[str | None]],
) -> "tuple[dict[str, Any], dict[str, dict[str, Any]]]":
    """Read CityGML ``app:X3DMaterial`` appearances into a CityJSON appearance pool + per-geometry maps.

    Builds a document ``materials`` list and, matching each material's ``app:target`` polygon
    ``gml:id`` to the surfaces gathered for each feature, a per-feature
    ``{"material": {<theme>: {"values": [...]}}}`` map whose values are parallel to that feature's
    surfaces (a material index, or ``None`` for an unpainted surface). Textures
    (``app:ParameterizedTexture``) are **not** read yet — their UV coordinates need the
    ``vertices-texture`` pool rebuilt from ``textureCoordinates`` (a documented follow-up); CityJSON
    textures, by contrast, already round-trip (they ride along in the appearance pool).

    Returns:
        ``(appearance_pool, appearance_by_feature)`` — both empty when the document has no
        X3DMaterial appearances.
    """
    materials: list[dict[str, Any]] = []
    target_to_material: dict[str, int] = {}
    theme_of: dict[int, str] = {}
    for appearance in root.iter():
        if _local(appearance.tag) != "Appearance":
            continue
        theme = next(
            (
                (child.text or "").strip()
                for child in appearance
                if _local(child.tag) == "theme" and (child.text or "").strip()
            ),
            "default",
        )
        for node in appearance.iter():
            if _local(node.tag) != "X3DMaterial":
                continue
            material, targets = _parse_x3d_material(node)
            index = len(materials)
            materials.append(material)
            theme_of[index] = theme
            for target in targets:
                target_to_material[target] = index

    if not materials:
        return {}, {}

    appearance_by_feature: dict[str, dict[str, Any]] = {}
    for feature_id, polygon_ids in polygon_ids_by_feature.items():
        values = [
            target_to_material.get(polygon_id or "") for polygon_id in polygon_ids
        ]
        painted = [value for value in values if value is not None]
        if not painted:
            continue
        theme = theme_of[painted[0]]
        appearance_by_feature[feature_id] = {"material": {theme: {"values": values}}}
    return {"materials": materials}, appearance_by_feature


def parse_citygml(source: Any, selector: "ObjectFilter | None" = None) -> CityModel:
    """Parse a CityGML document's LOD2 buildings into a :class:`~pyramids_citymodels.ir.CityModel`.

    Args:
        source: A path to a ``.gml``/``.xml`` file, an open file, or an XML string/bytes.
        selector: An optional :class:`~pyramids_citymodels.options.ObjectFilter`. When given, only
            matching ``Building``/``BuildingPart`` features (by id/type/attributes, and — for a
            feature with a ``gml:Envelope`` — a bbox prefilter) are read; a dropped feature's
            boundary surfaces are not parsed, and — per the filter's contract — any surfaces under
            it roll up to the nearest *kept* ancestor feature. ``None`` reads every feature.

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
    feature_ids, attributes_by_element = _assign_feature_ids(root, selector)

    model = CityModel(
        crs=_detect_crs(root), source_format="CityGML", version=_detect_version(root)
    )

    # Create a city object per feature, with parent/child links between a part and its building.
    for element, identifier in feature_ids.items():
        parent_feature_id = _nearest_feature_id(element, parent_of, feature_ids)
        model.objects[identifier] = CityObject(
            id=identifier,
            type=_local(element.tag),
            attributes=attributes_by_element[element],
            parents=[parent_feature_id] if parent_feature_id else [],
        )
        if parent_feature_id:
            model.objects[parent_feature_id].children.append(identifier)

    surfaces_by_feature, lod_by_feature, polygon_ids_by_feature = _gather_surfaces(
        root, parent_of, feature_ids, _index_polygons(root)
    )
    appearance_pool, appearance_by_feature = _gather_appearances(
        root, polygon_ids_by_feature
    )
    if appearance_pool:
        model.appearances = appearance_pool
    for identifier, (boundaries, semantics, values) in surfaces_by_feature.items():
        if not boundaries:
            continue
        # A feature with boundary surfaces is LOD 2 (the semantic-surface level) and keeps its
        # values; a feature with only direct geometry takes the LOD read from its lodX container
        # (None when none was found) and carries no semantics.
        if semantics:
            lod: str | None = "2"
            semantic_values: list[int | None] | None = values
        else:
            lod = lod_by_feature.get(identifier) or None
            semantic_values = None
        model.objects[identifier].geometries.append(
            Geometry(
                type="MultiSurface",
                lod=lod,
                boundaries=boundaries,
                semantics=semantics,
                semantic_values=semantic_values,
                appearance=appearance_by_feature.get(identifier),
            )
        )

    return model
