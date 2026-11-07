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
  ``Solid`` (emitted as ``bldg:lod{n}Solid``), and a surface's **material** (``app:X3DMaterial``) and
  **texture** (``app:ParameterizedTexture`` — image plus per-ring ``textureCoordinates`` from the
  ``vertices-texture`` pool, each ``app:theme`` an ``app:Appearance`` targeting the polygon's assigned
  ``gml:id``) — so the **painted colour / texture per surface** survives the round-trip (and the
  material ``name``, carried in a free-form ``gml:name``). What does **not** survive is the appearance
  *pool* itself: a material/texture is inlined once per theme that paints it, so an unpainted one is
  dropped and the indices are recompacted on re-read (the per-surface appearance is unchanged, but the
  pool composition and index values are not). The **version** round-trips for a ``1.0``, ``2.0`` or
  ``3.0`` source: the source version's namespaces are emitted. 1.0/2.0 use the ``bldg:boundedBy``
  building structure; **3.0** uses ``bldg:boundary`` and the ``gml/3.2`` namespace for the same
  building + common-object semantic/geometry/appearance model this writer covers (3.0's wider
  space/versioning features are out of scope, and a 3.0 document's namespace prefixes are
  auto-assigned — valid, and read back by local name). An unknown version falls back to ``2.0`` with
  a warning.
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
  type is a follow-up). A ``MultiSolid`` is written as its surfaces (its N-solid partitioning is not
  preserved), though it is never mislabelled as a single ``Solid``.

It writes with the standard-library :mod:`xml.etree.ElementTree` (no ``lxml``, matching the reader),
so there is no new dependency and no optional ``extra``.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import warnings
from collections.abc import Iterable
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

#: CityGML versions this writer emits natively (so the version string round-trips). 1.0/2.0 share the
#: ``bldg:boundedBy`` building structure; 3.0 uses ``bldg:boundary`` and the ``gml/3.2`` namespace (see
#: :func:`_namespaces` / :func:`_boundary_tag`). Any other version is emitted as 2.0 with a warning.
_MATCHED_VERSIONS = frozenset({"1.0", "2.0", "3.0"})

#: The GML namespace, constant across CityGML versions this writer emits.
_GML_NS = "http://www.opengis.net/gml"

#: Thematic module prefix → the module path segment of its ``.../citygml/<module>/<version>`` URI.
_MODULE_PATHS = {
    "bldg": "building",
    "gen": "generics",
    "app": "appearance",
    "brid": "bridge",
    "tun": "tunnel",
    "frn": "cityfurniture",
    "veg": "vegetation",
    "wtr": "waterbody",
    "luse": "landuse",
    "dem": "relief",
    "tran": "transportation",
    "grp": "cityobjectgroup",
}

#: City-object type → the thematic-module prefix its element and ``consistsOf…`` wrapper belong to, so
#: a non-Building object is emitted in its correct (schema-valid) namespace, e.g. ``frn:CityFurniture``
#: rather than an invalid ``bldg:CityFurniture``. Any type absent here falls back to ``bldg`` + a warning.
_TYPE_NAMESPACE_PREFIX = {
    "Building": "bldg",
    "BuildingPart": "bldg",
    "Bridge": "brid",
    "BridgePart": "brid",
    "Tunnel": "tun",
    "TunnelPart": "tun",
    "CityFurniture": "frn",
    "SolitaryVegetationObject": "veg",
    "PlantCover": "veg",
    "WaterBody": "wtr",
    "LandUse": "luse",
    "ReliefFeature": "dem",
    "TINRelief": "dem",
    "TransportationComplex": "tran",
    "Road": "tran",
    "Railway": "tran",
    "Track": "tran",
    "Square": "tran",
    "GenericCityObject": "gen",
    "CityObjectGroup": "grp",
}

#: Object types this writer emits in a schema-valid namespace (and the reader reads back). A type not
#: here is written under ``bldg:`` with a warning (best-effort, not conformant).
_READABLE_OBJECT_TYPES = frozenset(_TYPE_NAMESPACE_PREFIX)


def _namespaces(version: str) -> dict[str, str]:
    """The ``prefix → URI`` map for a CityGML ``version`` (``core`` and every thematic module).

    Returned fresh per call and threaded through element construction, so the writer holds **no**
    mutable module state — concurrent writes of different-version models cannot corrupt each other.
    Unused namespaces are not serialized (ElementTree emits an ``xmlns`` only for a namespace a
    document actually uses), so declaring every module here does not bloat the output.
    """
    namespaces = {
        prefix: f"http://www.opengis.net/citygml/{module}/{version}"
        for prefix, module in _MODULE_PATHS.items()
    }
    namespaces["core"] = f"http://www.opengis.net/citygml/{version}"
    # CityGML 3.0 is built on GML 3.2; 1.0/2.0 on the older GML namespace.
    namespaces["gml"] = (
        "http://www.opengis.net/gml/3.2" if version == "3.0" else _GML_NS
    )
    return namespaces


def _boundary_tag(ns: dict[str, str]) -> str:
    """The semantic-boundary wrapper element for this version: 3.0's ``boundary`` else ``boundedBy``."""
    return "boundary" if ns["core"].rstrip("/").endswith("/3.0") else "boundedBy"


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


class _UniqueIds:
    """Hands out ``gml:id``\\ s unique within one document.

    ``gml:id`` is an ``xsd:ID`` and must be unique per document, so the generated polygon ids and the
    material-``name`` ids are drawn from here, seeded with the ids already in use (the city-object
    ids). A preferred id that is taken gets a ``_2``/``_3``/… suffix.
    """

    def __init__(self, reserved: Iterable[str]):
        self._used: set[str] = set(reserved)

    def allocate(self, preferred: str) -> str:
        """Return ``preferred`` if free, else ``preferred_2``/``_3``/…; the result is then reserved."""
        candidate = preferred
        suffix = 2
        while candidate in self._used:
            candidate = f"{preferred}_{suffix}"
            suffix += 1
        self._used.add(candidate)
        return candidate


#: A texture target: the painted polygon's id and, per painted ring, its ``gml:id`` + the closed
#: ``textureCoordinates`` text.
_TextureTarget = tuple[str, list[tuple[str, str]]]


class _Appearances:
    """Collects, while geometry is emitted, the material and texture appearances to write per theme.

    The CityGML writer assigns every emitted ``gml:Polygon`` a document-unique ``gml:id`` and, for a
    painted surface, records that id under the appearance's theme (a texture also assigns each painted
    ``gml:LinearRing`` an id and carries its UV coordinates); :meth:`appearance_members` then builds
    one ``app:Appearance`` per theme with the ``app:X3DMaterial`` and ``app:ParameterizedTexture``
    elements that target those ids — so a CityJSON material/texture survives ``to_citygml``. All ids
    come from a shared :class:`_UniqueIds`, so the document stays valid even when an appearance is
    reused across themes or a name/object-id would otherwise collide.
    """

    def __init__(
        self,
        materials: list[dict[str, Any]],
        textures: list[dict[str, Any]],
        vertices_texture: list[list[float]],
        ids: _UniqueIds,
    ):
        self._materials = materials
        self._textures = textures
        self._vertices_texture = vertices_texture
        self._ids = ids
        self._material_targets: dict[str, dict[int, list[str]]] = {}
        self._texture_targets: dict[str, dict[int, list[_TextureTarget]]] = {}

    def paint(
        self,
        ns: dict[str, str],
        polygon: ElementTree.Element,
        surface: list[Any],
        material_by_theme: dict[str, int],
        texture_by_theme: dict[str, list[Any]],
    ) -> None:
        """Give ``polygon`` a document-unique ``gml:id`` and record the materials/textures painting it."""
        polygon_id = self._ids.allocate("poly")
        polygon.set(_q(ns, "gml", "id"), polygon_id)
        for theme, index in material_by_theme.items():
            self._material_targets.setdefault(theme, {}).setdefault(index, []).append(
                polygon_id
            )
        if texture_by_theme:
            self._paint_textures(ns, polygon, surface, polygon_id, texture_by_theme)

    def _paint_textures(
        self,
        ns: dict[str, str],
        polygon: ElementTree.Element,
        surface: list[Any],
        polygon_id: str,
        texture_by_theme: dict[str, list[Any]],
    ) -> None:
        """Assign painted-ring ids and record each theme's texture target for ``polygon``.

        ``_polygon_element`` drops rings with fewer than three vertices, so the emitted
        ``gml:LinearRing`` elements correspond to the surface's **kept** rings, not every ring. The
        texture ``values`` are indexed over *all* surface rings, so the kept-ring indices bridge the
        two — without this the UVs would shift onto the wrong ring when a degenerate ring is dropped.
        ``polygon.iter()`` yields the LinearRings in document order (exterior then interiors), the same
        order ``_polygon_element`` emitted the kept rings.
        """
        kept = [index for index, ring in enumerate(surface) if ring and len(ring) >= 3]
        linear_rings = [
            element
            for element in polygon.iter()
            if element.tag == _q(ns, "gml", "LinearRing")
        ]
        # Each kept ring's gml:id is allocated once and shared across themes: a ring textured under
        # two themes must keep ONE id — otherwise the second theme overwrites the element's gml:id and
        # the first theme's textureCoordinates reference a dangling ring (review R2-H1).
        ring_ids: dict[int, str] = {}
        for theme, surface_value in texture_by_theme.items():
            by_texture = self._ring_targets(
                ns, linear_rings, kept, surface, surface_value, ring_ids
            )
            for texture_index, ring_coords in by_texture.items():
                self._texture_targets.setdefault(theme, {}).setdefault(
                    texture_index, []
                ).append((polygon_id, ring_coords))

    def _ring_targets(
        self,
        ns: dict[str, str],
        linear_rings: list[ElementTree.Element],
        kept: list[int],
        surface: list[Any],
        surface_value: list[Any],
        ring_ids: dict[int, str],
    ) -> dict[int, list[tuple[str, str]]]:
        """Map texture index → ``[(ring_id, coords)]`` for one theme's value over ``linear_rings``.

        ``kept[position]`` is the surface-ring index of the ``position``-th emitted ``LinearRing``;
        ``ring_ids`` memoizes each ring's assigned ``gml:id`` so it is shared across themes.
        """
        by_texture: dict[int, list[tuple[str, str]]] = {}
        for position, ring_element in enumerate(linear_rings):
            if position >= len(kept):
                break
            ring_index = kept[position]
            if ring_index >= len(surface_value):
                continue
            resolved = self._ring_texture(
                surface_value[ring_index], surface[ring_index]
            )
            if resolved is None:
                continue
            texture_index, coordinates = resolved
            if position not in ring_ids:
                ring_ids[position] = self._ids.allocate("ring")
                ring_element.set(_q(ns, "gml", "id"), ring_ids[position])
            by_texture.setdefault(texture_index, []).append(
                (ring_ids[position], coordinates)
            )
        return by_texture

    def _ring_texture(
        self, ring_value: Any, ring: list[Any]
    ) -> "tuple[int, str] | None":
        """A painted ring's ``(texture index, closed textureCoordinates text)``, or ``None`` to skip.

        An unpainted ring — ``[None]`` or an index with no UV coordinates — is skipped **silently**
        (an unpainted ring is not an error). A **malformed** ``ring_value`` (not a list) or an
        **invalid** texture/UV index (not an int, or out of range of its pool) is skipped **with a
        warning** (never crashes ``to_citygml``). The UV coordinates are re-closed to match the
        emitted ``gml:LinearRing`` — closed exactly when :func:`_pos_list` closes the ``ring`` (an open
        ring of ≥3 vertices), so the texture-coordinate count equals the ring's position count.
        """
        if not isinstance(ring_value, (list, tuple)):
            warnings.warn(
                f"CityGML writer: malformed texture ring value {ring_value!r} is ignored",
                stacklevel=2,
            )
            return None
        if not ring_value or ring_value[0] is None:
            return None
        texture_index = ring_value[0]
        if not isinstance(texture_index, int) or isinstance(texture_index, bool):
            warnings.warn(
                f"CityGML writer: non-integer texture index {texture_index!r} is ignored",
                stacklevel=2,
            )
            return None
        if not 0 <= texture_index < len(self._textures):
            warnings.warn(
                f"CityGML writer: texture index {texture_index} is out of range for the "
                f"{len(self._textures)}-texture pool; that ring is left untextured",
                stacklevel=2,
            )
            return None
        coordinates: list[list[float]] = []
        for uv in ring_value[1:]:
            if not isinstance(uv, int) or isinstance(uv, bool):
                warnings.warn(
                    f"CityGML writer: non-integer UV index {uv!r} is ignored; ring untextured",
                    stacklevel=2,
                )
                return None
            if not 0 <= uv < len(self._vertices_texture):
                warnings.warn(
                    f"CityGML writer: UV index {uv} is out of range for the "
                    f"{len(self._vertices_texture)}-entry vertices-texture pool; ring untextured",
                    stacklevel=2,
                )
                return None
            coordinates.append(self._vertices_texture[uv])
        if not coordinates:
            return None
        # Close the UV list exactly when _pos_list closes the geometry ring (an open ring of ≥3
        # vertices), so textureCoordinates has one pair per emitted ring position — regardless of
        # whether the first and last UV coordinates happen to coincide (review R2-L1).
        if len(ring) >= 3 and ring[0] != ring[-1]:
            coordinates = [*coordinates, coordinates[0]]
        text = " ".join(repr(float(c)) for pair in coordinates for c in pair)
        return texture_index, text

    def appearance_members(self, ns: dict[str, str]) -> list[ElementTree.Element]:
        """One ``app:appearanceMember``/``app:Appearance`` per painted theme (``[]`` if none)."""
        members: list[ElementTree.Element] = []
        themes = list(dict.fromkeys([*self._material_targets, *self._texture_targets]))
        for theme in themes:
            member = ElementTree.Element(_q(ns, "app", "appearanceMember"))
            appearance = ElementTree.SubElement(member, _q(ns, "app", "Appearance"))
            theme_element = ElementTree.SubElement(appearance, _q(ns, "app", "theme"))
            theme_element.text = theme
            painted = self._emit_materials(ns, appearance, theme)
            painted = self._emit_textures(ns, appearance, theme) or painted
            # Only emit a theme's Appearance if it painted something — a theme whose every index was
            # out of range must not leave an empty app:Appearance behind.
            if painted:
                members.append(member)
        return members

    def _emit_materials(
        self, ns: dict[str, str], appearance: ElementTree.Element, theme: str
    ) -> bool:
        """Append this theme's ``app:X3DMaterial`` surface-data members; return whether any was added."""
        painted = False
        for index, polygon_ids in self._material_targets.get(theme, {}).items():
            if not 0 <= index < len(self._materials):
                warnings.warn(
                    f"CityGML writer: material index {index} is out of range for the "
                    f"{len(self._materials)}-material pool; those surfaces are left unpainted",
                    stacklevel=2,
                )
                continue
            surface_data = ElementTree.SubElement(
                appearance, _q(ns, "app", "surfaceDataMember")
            )
            self._x3d_material(ns, surface_data, self._materials[index], polygon_ids)
            painted = True
        return painted

    def _emit_textures(
        self, ns: dict[str, str], appearance: ElementTree.Element, theme: str
    ) -> bool:
        """Append this theme's ``app:ParameterizedTexture`` surface-data members; return whether any."""
        painted = False
        for texture_index, target_list in self._texture_targets.get(theme, {}).items():
            texture = self._textures[texture_index]
            if not texture.get("image"):
                warnings.warn(
                    f"CityGML writer: texture {texture_index} has no image URI; its surfaces are "
                    "left untextured (an app:ParameterizedTexture needs an imageURI)",
                    stacklevel=2,
                )
                continue
            surface_data = ElementTree.SubElement(
                appearance, _q(ns, "app", "surfaceDataMember")
            )
            self._parameterized_texture(ns, surface_data, texture, target_list)
            painted = True
        return painted

    def _parameterized_texture(
        self,
        ns: dict[str, str],
        surface_data: ElementTree.Element,
        texture: dict[str, Any],
        target_list: list[_TextureTarget],
    ) -> None:
        """Build one ``app:ParameterizedTexture`` (image + per-polygon ``app:target`` TexCoordLists)."""
        parameterized = ElementTree.SubElement(
            surface_data, _q(ns, "app", "ParameterizedTexture")
        )
        parameterized.set(_q(ns, "gml", "id"), self._ids.allocate("tex"))
        image = ElementTree.SubElement(parameterized, _q(ns, "app", "imageURI"))
        image.text = str(texture.get("image", ""))
        for polygon_id, ring_coords in target_list:
            target = ElementTree.SubElement(parameterized, _q(ns, "app", "target"))
            target.set("uri", f"#{polygon_id}")
            coord_list = ElementTree.SubElement(target, _q(ns, "app", "TexCoordList"))
            for ring_id, coordinates in ring_coords:
                texture_coordinates = ElementTree.SubElement(
                    coord_list, _q(ns, "app", "textureCoordinates")
                )
                texture_coordinates.set("ring", f"#{ring_id}")
                texture_coordinates.text = coordinates

    def _x3d_material(
        self,
        ns: dict[str, str],
        surface_data: ElementTree.Element,
        material: dict[str, Any],
        polygon_ids: list[str],
    ) -> None:
        """Build one ``app:X3DMaterial`` with ``material``'s members and an ``app:target`` per id.

        The ``gml:id`` is a generated, document-unique ``mat`` id (always a valid ``NCName``); the
        material's ``name`` rides in a free-form ``gml:name`` child instead, so it round-trips
        verbatim even when two materials share a name or a name is not ``NCName``-safe.
        """
        x3d = ElementTree.SubElement(surface_data, _q(ns, "app", "X3DMaterial"))
        x3d.set(_q(ns, "gml", "id"), self._ids.allocate("mat"))
        name = material.get("name")
        if name is not None:
            name_element = ElementTree.SubElement(x3d, _q(ns, "gml", "name"))
            name_element.text = str(name)
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
        # iter_surfaces walks ``boundaries`` (the first arg), so it yields exactly ``count`` surfaces
        # regardless of ``values`` — ``position`` is always a valid index into ``per_surface``.
        for position, (_surface, index) in enumerate(
            iter_surfaces(geometry.boundaries, values)
        ):
            if index is None:
                continue
            # A material index must be a plain int (``bool`` is an int subclass — exclude it). A
            # malformed leaf (a list from mis-nested values, a str/float) is skipped with a warning
            # rather than crashing to_citygml() downstream (unhashable key / bad comparison).
            if not isinstance(index, int) or isinstance(index, bool):
                warnings.warn(
                    f"CityGML writer: non-integer material index {index!r} is ignored; "
                    "that surface is left unpainted",
                    stacklevel=2,
                )
                continue
            per_surface[position][theme] = index
    return per_surface


def _surface_textures(geometry: Geometry) -> list[dict[str, list[Any]]]:
    """Per-surface ``{theme: surface texture value}`` for a geometry, in :func:`iter_surfaces` order.

    The texture ``values`` nest parallel to the boundaries at *surface* level (each surface's value is
    its list of per-ring ``[texture_index, uv₀, …]`` / ``[None]`` entries), so the value for surface
    *i* lines up with the *i*-th surface the writer emits. An untextured (``None``) surface contributes
    no entry.
    """
    texture_map = (geometry.appearance or {}).get("texture") or {}
    count = sum(1 for _ in iter_surfaces(geometry.boundaries, geometry.semantic_values))
    per_surface: list[dict[str, list[Any]]] = [{} for _ in range(count)]
    for theme, spec in texture_map.items():
        values = (spec or {}).get("values")
        if values is None:
            continue
        for position, (_surface, surface_value) in enumerate(
            iter_surfaces(geometry.boundaries, values)
        ):
            if surface_value is not None:
                per_surface[position][theme] = surface_value
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
    targets: _Appearances,
    material_by_theme: dict[str, int],
    texture_by_theme: dict[str, list[Any]],
) -> ElementTree.Element | None:
    """A ``bldg:boundedBy`` wrapping one boundary surface of ``semantic_type`` and its polygon."""
    polygon = _polygon_element(ns, surface)
    if polygon is None:
        return None
    targets.paint(ns, polygon, surface, material_by_theme, texture_by_theme)
    bounded_by = ElementTree.Element(_q(ns, "bldg", _boundary_tag(ns)))
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
    ns: dict[str, str],
    feature: ElementTree.Element,
    attributes: dict[str, Any],
    units: dict[str, str] | None = None,
) -> None:
    """Append an object's attributes: native ``bldg:`` elements for the known numeric ones, else
    a typed ``gen:*Attribute`` that preserves the Python value type on read.

    A ``uom`` from ``units`` is re-emitted on the matching native element (e.g.
    ``<bldg:measuredHeight uom="m">``), so a unit read from CityGML round-trips.
    """
    units = units or {}
    for name, value in attributes.items():
        if name in _NATIVE_BUILDING_ATTRIBUTES:
            element = ElementTree.SubElement(feature, _q(ns, "bldg", name))
            element.text = repr(value) if isinstance(value, float) else str(value)
            if name in units:
                element.set("uom", units[name])
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
    targets: _Appearances,
    material_by_theme: dict[str, int],
    texture_by_theme: dict[str, list[Any]],
) -> None:
    """Append a surface with no semantic type as a bare ``bldg:lod{n}MultiSurface`` on the feature."""
    plain = _polygon_element(ns, surface)
    if plain is None:
        return
    targets.paint(ns, plain, surface, material_by_theme, texture_by_theme)
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
    targets: _Appearances,
    surface_materials: list[dict[str, int]],
    surface_textures: list[dict[str, list[Any]]],
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
        texture = surface_textures[position] if position < len(surface_textures) else {}
        targets.paint(ns, polygon, surface, material, texture)
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
    targets: _Appearances,
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
    surface_textures = _surface_textures(geometry)
    if geometry.type == "Solid" and not geometry.semantics:
        _append_solid(
            ns, feature, geometry, lod, targets, surface_materials, surface_textures
        )
        return
    for position, (surface, semantic_value) in enumerate(
        iter_surfaces(geometry.boundaries, geometry.semantic_values)
    ):
        material = (
            surface_materials[position] if position < len(surface_materials) else {}
        )
        texture = surface_textures[position] if position < len(surface_textures) else {}
        surface_object = geometry.semantic_for(semantic_value)
        if surface_object is None:
            _append_plain_surface(ns, feature, surface, lod, targets, material, texture)
            continue
        bounded_by = _bounded_by(
            ns, surface_object.type, surface, lod, targets, material, texture
        )
        if bounded_by is not None:
            feature.append(bounded_by)


def _append_children(
    ns: dict[str, str],
    feature: ElementTree.Element,
    city_object: CityObject,
    model: CityModel,
    visited: set[str],
    targets: _Appearances,
) -> None:
    """Nest a part child under its ``consistsOf…Part`` wrapper in the parent's module, else attach it."""
    for child_id in city_object.children:
        child = model.objects.get(child_id)
        if child is None or child_id in visited:
            continue
        child_element = _feature_element(ns, child, model, visited, targets)
        if child.type.endswith("Part"):
            # A <X>Part nests under <module>:consistsOf<X>Part in its own thematic module
            # (bldg:consistsOfBuildingPart, brid:consistsOfBridgePart, tun:consistsOfTunnelPart).
            prefix = _TYPE_NAMESPACE_PREFIX.get(child.type, "bldg")
            consists = ElementTree.SubElement(
                feature, _q(ns, prefix, f"consistsOf{child.type}")
            )
            consists.append(child_element)
        else:
            # A non-part child cannot nest; attach it so its geometry is not lost, though this
            # package's reader will not re-link it as a child.
            feature.append(child_element)


def _feature_element(
    ns: dict[str, str],
    city_object: CityObject,
    model: CityModel,
    visited: set[str],
    targets: _Appearances,
) -> ElementTree.Element:
    """Build one feature element in its thematic namespace: id, attributes, surfaces and children."""
    prefix = _TYPE_NAMESPACE_PREFIX.get(city_object.type, "bldg")
    feature = ElementTree.Element(_q(ns, prefix, city_object.type))
    feature.set(_q(ns, "gml", "id"), city_object.id)
    visited.add(city_object.id)
    _add_attributes(ns, feature, city_object.attributes, city_object.attribute_units)
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
        - A surface's material is written as an ``app:X3DMaterial`` and survives the round-trip:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> from pyramids_citymodels.readers.citygml import parse_citygml
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(
            ...     appearances={"materials": [{"name": "red", "diffuseColor": [1.0, 0.0, 0.0]}]},
            ...     objects={"b": CityObject(id="b", type="Building", geometries=[
            ...         Geometry(type="MultiSurface", lod="2", boundaries=[surface],
            ...                  appearance={"material": {"t": {"values": [0]}}})])})
            >>> "X3DMaterial" in write_citygml(model)
            True
            >>> back = parse_citygml(write_citygml(model))
            >>> back.appearances["materials"][0]["diffuseColor"]
            [1.0, 0.0, 0.0]

            ```
        - A surface's texture is written as an ``app:ParameterizedTexture`` and round-trips:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> from pyramids_citymodels.readers.citygml import parse_citygml
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(
            ...     appearances={"textures": [{"type": "PNG", "image": "wall.png"}],
            ...                  "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
            ...     objects={"b": CityObject(id="b", type="Building", geometries=[
            ...         Geometry(type="MultiSurface", lod="2", boundaries=[surface],
            ...                  appearance={"texture": {"t": {"values": [[[0, 0, 1, 2]]]}}})])})
            >>> "ParameterizedTexture" in write_citygml(model)
            True
            >>> parse_citygml(write_citygml(model)).appearances["textures"][0]["image"]
            'wall.png'

            ```
    """
    # Emit the source version's namespaces for a version this writer matches (1.0/2.0), so the
    # version string round-trips through the reader; an unknown version falls back to 2.0. The map is
    # a per-call local threaded through construction — no module-global is mutated, so concurrent
    # writes of different-version models cannot corrupt each other.
    version = model.version if model.version in _MATCHED_VERSIONS else "2.0"
    if model.version is not None and model.version not in _MATCHED_VERSIONS:
        # 1.0/2.0/3.0 are emitted natively; any other version has no known structure, so it is written
        # as valid 2.0 and its version does not round-trip.
        warnings.warn(
            f"CityGML writer: a {model.version} source has no known emission structure and is "
            "written as valid CityGML 2.0, so the CityGML version does not round-trip",
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
            f"{', '.join(unreadable)} have no known thematic namespace and are written under bldg: "
            "(not schema-valid), so they will not round-trip conformantly",
            stacklevel=2,
        )

    root = ElementTree.Element(_q(ns, "core", "CityModel"))
    _add_envelope(ns, root, model)

    materials = (model.appearances or {}).get("materials") or []
    textures = (model.appearances or {}).get("textures") or []
    vertices_texture = (model.appearances or {}).get("vertices-texture") or []
    # Seed the id allocator with the city-object ids so generated polygon/material ids never collide
    # with them (gml:id is an xsd:ID — unique per document).
    targets = _Appearances(
        materials, textures, vertices_texture, _UniqueIds(model.objects)
    )
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
