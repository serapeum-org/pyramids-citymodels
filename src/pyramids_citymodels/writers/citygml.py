"""Serialize the IR to CityGML 2.0 (issue #26) — the mirror of :mod:`.readers.citygml`.

This is the hard writer. CityGML is schema-heavy GML/XML, and the reader is deliberately lossy
(issue #24): it reconstructs the LOD2 *semantic* model — ``Building → BuildingPart → boundary
surface → gml:Polygon`` — matching elements by local name across CityGML 1.0/2.0/3.0. The writer
targets exactly what that reader rebuilds and emits **OGC-XSD-valid** CityGML (verified against the
official 2.0 and 3.0 schemas — see ``tests/test_xsd_conformance.py``). The round-trip fidelity
contract is:

- **Preserved:** each object's id (written as ``gml:id``), its type, the parent/child hierarchy
  (children nested in ``<module>:consistsOf…Part``), every boundary surface's geometry (rings, holes,
  coordinates) across ``MultiSurface`` and ``Solid``/``MultiSolid`` alike, its semantic type, and the
  CRS. Each object is emitted in its **own thematic namespace** — ``bldg:Building``, ``brid:Bridge``,
  ``tun:Tunnel``, ``frn:CityFurniture``, ``veg:…``, ``luse:LandUse``, ``tran:…``, ``gen:…`` — with its
  boundary surfaces and geometry in that module's own elements (``brid:boundedBy`` →
  ``brid:WallSurface`` → ``brid:lod{n}MultiSurface``; city furniture / generics use their
  ``lod{n}Geometry`` slot; land use / transportation their ``lod{n}MultiSurface``). A plain
  (non-semantic) ``Solid`` round-trips **as a** ``Solid`` (the module's ``lod{n}Solid``). Attribute
  ``uom`` **units** round-trip: a native building attribute re-emits its ``uom``
  (``<bldg:measuredHeight uom="m">``) and a generic attribute with a unit is written as a
  ``gen:measureAttribute`` (whose value legally carries the ``uom``). A surface's **material**
  (``app:X3DMaterial``) and **texture** (``app:ParameterizedTexture`` — image plus per-ring
  ``textureCoordinates``, each ``app:theme`` an ``app:Appearance`` targeting the polygon's ``gml:id``)
  survive, as does the material ``name`` (a free-form ``gml:name``). The appearance *pool* itself does
  not survive a CityGML round-trip — **inherently**, not as a bug: CityGML has no flat indexed
  appearance pool (a surface references its material by ``gml:id``, not by a pool index), so a
  material/texture is inlined once per theme that paints it, an **unpainted** pool entry has nothing to
  attach to and is dropped, and the indices are recompacted on re-read. (The CityJSON writer, whose
  format *does* have an indexed pool, re-emits it verbatim — composition and index values survive a
  CityJSON round-trip; see #60.) The **version** round-trips for a ``1.0``/``2.0``/``3.0``
  source. 1.0/2.0 use the ``<module>:boundedBy`` structure on GML 3.1.1; **3.0** emits valid CityGML
  3.0 — ``core:boundary`` → ``con:<Surface>`` (the Construction module) → ``core:lod{n}MultiSurface``
  on GML 3.2 — for the building + common-object model this writer covers (3.0's wider space/versioning
  features are out of scope, and a 3.0 document's namespace prefixes are auto-assigned: valid, read
  back by local name). An unknown version falls back to valid ``2.0`` with a warning.
- **Best-effort:** attribute *values* keep their Python type via ``gen:*Attribute`` kinds
  (``int`` → ``gen:intAttribute``, ``float`` → ``gen:doubleAttribute``, else ``gen:stringAttribute``).
  Rings are emitted closed (first position repeated) so the ``gml:LinearRing`` is schema-valid even
  though the IR (CityJSON-style) carries open rings — so a CityGML→CityGML round-trip is exact only for
  a **spec-conformant, already-closed-ring source** (re-closing is a no-op); an open-ring source gains
  the required closing position, so its re-parsed rings carry one extra coordinate. Real GML rings are
  always closed, so this affects only hand-authored open-ring input, not conformant CityGML.
- **Lossy:** original ``gml:id``\\ s of surfaces/polygons (regenerated); a geometry whose surfaces
  carry *no* semantic type (written as a plain module geometry, which this package's semantic reader
  does not read back); a semantic surface whose type the object's module cannot carry as a boundary
  surface (e.g. a ``RoofSurface`` on a ``LandUse``), which is demoted to plain geometry — its semantic
  type dropped — **with a warning**; and a few object types whose module has no slot for the IR's generic surface
  geometry — ``ReliefFeature``/``TINRelief`` (need a TIN structure), ``CityObjectGroup``, and a
  WaterBody *surface* (water boundary surfaces use a different ``lod{n}Surface`` geometry). Their
  feature element is still emitted in the correct namespace, but the geometry falls back to a ``bldg:``
  property **with a warning** (preserved for a CityJSON/JSON-FG/mesh target, not a conformant
  CityGML→CityGML round-trip). An entirely unknown object type likewise falls back to ``bldg:`` + a
  warning. A ``MultiSolid`` is written as its surfaces (its N-solid partitioning is not preserved,
  though it is never mislabelled as a single ``Solid``).

It writes with the standard-library :mod:`xml.etree.ElementTree` (no ``lxml`` at runtime, matching the
reader — ``lxml`` is a test-only dependency used by the XSD-conformance check), so there is no new
runtime dependency and no optional ``extra``.
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

#: The nine thematic boundary-surface element names shared by the building/bridge/tunnel modules
#: (each defines its own, in its own namespace; the local names are identical). Taken from the
#: official CityGML 2.0 XSDs — a semantic surface of one of these types is written as a
#: ``<module>:boundedBy`` → ``<module>:<Type>`` boundary surface; any other semantic type has no
#: ``boundedBy`` representation in its module and is written as plain geometry instead.
_CONSTRUCTION_SURFACES = frozenset(
    {
        "RoofSurface",
        "WallSurface",
        "GroundSurface",
        "ClosureSurface",
        "FloorSurface",
        "OuterFloorSurface",
        "InteriorWallSurface",
        "CeilingSurface",
        "OuterCeilingSurface",
    }
)


class _GeometryProfile:
    """How one city-object *type* may carry geometry in **CityGML 2.0**, per the official XSDs.

    The thematic modules do not share one geometry convention: building/bridge/tunnel have boundary
    surfaces (``boundedBy``) plus ``lod{n}MultiSurface``/``lod{n}Solid``/``lod{n}Geometry``; city
    furniture and generics have only a ``lod{n}Geometry`` (a ``gml:_Geometry`` slot); land use and
    transportation have ``lod{n}MultiSurface`` but no boundary surfaces; water has boundary surfaces
    and a solid but no plain ``MultiSurface``; relief and groups have no slot that accepts a generic
    surface at all. Each attribute is the *local* element name template (``{n}`` is the LOD integer),
    or ``None`` when that module has no such slot (then emission falls back, with a warning).
    """

    def __init__(
        self,
        surfaces: frozenset[str] = frozenset(),
        multisurface: str | None = None,
        solid: str | None = None,
        generic: str | None = None,
    ):
        self.surfaces = (
            surfaces  # semantic types that may be a boundedBy boundary surface
        )
        self.multisurface = (
            multisurface  # property for a plain MultiSurface, e.g. "lod{n}MultiSurface"
        )
        self.solid = solid  # property for a Solid, e.g. "lod{n}Solid"
        self.generic = generic  # gml:_Geometry property, e.g. "lod{n}Geometry"


#: Building-pattern profile (building/bridge/tunnel): boundary surfaces + MultiSurface + Solid + Geometry.
_CONSTRUCTION_PROFILE = _GeometryProfile(
    _CONSTRUCTION_SURFACES, "lod{n}MultiSurface", "lod{n}Solid", "lod{n}Geometry"
)
#: A ``gml:_Geometry``-only profile (city furniture, generics, solitary vegetation): ``lod{n}Geometry``.
_GEOMETRY_ONLY_PROFILE = _GeometryProfile(generic="lod{n}Geometry")
#: A plain-``MultiSurface`` profile, no boundary surfaces (land use, transportation, plant cover).
_MULTISURFACE_PROFILE = _GeometryProfile(multisurface="lod{n}MultiSurface")
#: WaterBody: only its ``lod{n}Solid`` body is cleanly representable from the IR's surface model. Its
#: boundary surfaces take a ``lod{n}Surface`` (single ``gml:Surface``, not ``MultiSurface``) geometry
#: and it has no plain ``MultiSurface`` slot, so a water *surface* or plain geometry falls back + warns.
_WATER_PROFILE = _GeometryProfile(solid="lod{n}Solid")
#: No slot that accepts a generic surface/solid (relief, city-object groups) — emission falls back.
_NO_GEOMETRY_PROFILE = _GeometryProfile()

#: City-object type → its CityGML 2.0 geometry profile. A type absent here uses the construction
#: profile under ``bldg:`` (the warned best-effort fallback in :func:`_feature_element`).
_TYPE_GEOMETRY_PROFILE = {
    "Building": _CONSTRUCTION_PROFILE,
    "BuildingPart": _CONSTRUCTION_PROFILE,
    "Bridge": _CONSTRUCTION_PROFILE,
    "BridgePart": _CONSTRUCTION_PROFILE,
    "Tunnel": _CONSTRUCTION_PROFILE,
    "TunnelPart": _CONSTRUCTION_PROFILE,
    "CityFurniture": _GEOMETRY_ONLY_PROFILE,
    "GenericCityObject": _GEOMETRY_ONLY_PROFILE,
    "SolitaryVegetationObject": _GEOMETRY_ONLY_PROFILE,
    "PlantCover": _MULTISURFACE_PROFILE,
    "LandUse": _MULTISURFACE_PROFILE,
    "TransportationComplex": _MULTISURFACE_PROFILE,
    "Road": _MULTISURFACE_PROFILE,
    "Railway": _MULTISURFACE_PROFILE,
    "Track": _MULTISURFACE_PROFILE,
    "Square": _MULTISURFACE_PROFILE,
    "WaterBody": _WATER_PROFILE,
    "ReliefFeature": _NO_GEOMETRY_PROFILE,
    "TINRelief": _NO_GEOMETRY_PROFILE,
    "CityObjectGroup": _NO_GEOMETRY_PROFILE,
}


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
    if version == "3.0":
        # CityGML 3.0 relocated the thematic boundary surfaces (RoofSurface, …) to the Construction
        # module; it has no 1.0/2.0 counterpart, so it is declared only here.
        namespaces["con"] = "http://www.opengis.net/citygml/construction/3.0"
    return namespaces


#: The CityGML 3.0 core namespace, compared exactly (not by URL suffix) to switch on the 3.0 geometry
#: model (``core:boundary`` → ``con:`` surfaces → ``core:lod{n}…``; see :class:`_GeometryPlan`).
_CORE_30 = "http://www.opengis.net/citygml/3.0"


class _WriteDiagnostics:
    """Feature-level notes gathered while a model is written, surfaced as one warning each at the end.

    ``fallbacks`` holds object types whose geometry had no valid slot in their module and used the
    ``bldg:`` fallback; ``demoted`` holds types where a semantic boundary surface could not be carried
    by the module and was written as plain geometry (its semantic type dropped). Both are warned once.
    """

    def __init__(self) -> None:
        self.fallbacks: set[str] = set()
        self.demoted: set[str] = set()


class _GeometryPlan:
    """The exact element QNames for emitting one object's geometry in a given version **and** module.

    Encapsulates the two things that vary: the CityGML **version** (2.0 keeps boundary surfaces and
    geometry in the object's own thematic module — ``bldg:boundedBy`` → ``bldg:WallSurface`` →
    ``bldg:lod2MultiSurface``; 3.0 moved them to core/construction — ``core:boundary`` →
    ``con:WallSurface`` → ``core:lod2MultiSurface``), and the **module**'s geometry
    :class:`_GeometryProfile` (which slots a given object type actually has). The ``_profile`` gates
    *whether* a slot exists in both versions (so a 3.0 relief/group with no slot still falls back and
    warns); the version only changes the *names*. A method returns ``None`` when the type+version has no
    valid slot for that geometry kind, so the caller can fall back.
    """

    def __init__(
        self,
        ns: dict[str, str],
        prefix: str,
        profile: _GeometryProfile,
        object_type: str,
        diagnostics: _WriteDiagnostics,
    ):
        self._ns = ns
        self._prefix = prefix
        self._profile = profile
        self._is_30 = ns["core"] == _CORE_30
        self.object_type = object_type
        self._diagnostics = diagnostics

    def note_fallback(self) -> None:
        """Record that this object's geometry had no valid slot and used the ``bldg:`` fallback."""
        self._diagnostics.fallbacks.add(self.object_type)

    def note_demoted(self) -> None:
        """Record that a semantic boundary surface was demoted to plain geometry (semantic dropped)."""
        self._diagnostics.demoted.add(self.object_type)

    def supports_surface(self, semantic_type: str) -> bool:
        """Whether ``semantic_type`` is a valid ``boundedBy`` boundary surface here.

        Only a module that has boundary surfaces at all (``profile.surfaces``) can carry one; in 3.0
        the surface *names* come from the Construction module, but a non-construction type (empty
        ``profile.surfaces``) still has none.
        """
        if not self._profile.surfaces:
            return False
        valid = _CONSTRUCTION_SURFACES if self._is_30 else self._profile.surfaces
        return semantic_type in valid

    def boundary_wrapper(self) -> str:
        """The boundary wrapper element (3.0 ``core:boundary``; else ``<module>:boundedBy``)."""
        if self._is_30:
            return _q(self._ns, "core", "boundary")
        return _q(self._ns, self._prefix, "boundedBy")

    def surface_element(self, semantic_type: str) -> str:
        """The boundary-surface element (3.0 ``con:<Type>``; else ``<module>:<Type>``)."""
        if self._is_30:
            return _q(self._ns, "con", semantic_type)
        return _q(self._ns, self._prefix, semantic_type)

    def surface_geometry(self, lod: int) -> str:
        """The ``lod{n}MultiSurface`` geometry inside a boundary surface (3.0 ``core:``; else module)."""
        prefix = "core" if self._is_30 else self._prefix
        return _q(self._ns, prefix, f"lod{lod}MultiSurface")

    def plain_property(self, lod: int) -> str | None:
        """The property for a plain ``MultiSurface``, or ``None`` when the type has no such slot.

        The ``_profile`` decides whether the type has a surface slot at all (so a relief/group with no
        slot returns ``None`` and falls back in 3.0 too); the version decides the element's name
        (3.0 ``core:lod{n}MultiSurface`` on the space, else the module's own slot).
        """
        template = self._profile.multisurface or self._profile.generic
        if template is None:
            return None
        if self._is_30:
            return _q(self._ns, "core", f"lod{lod}MultiSurface")
        return _q(self._ns, self._prefix, template.format(n=lod))

    def solid_property(self, lod: int) -> str | None:
        """The property for a ``Solid``, or ``None`` when the type has no such slot."""
        if not self._profile.solid:
            return None
        if self._is_30:
            return _q(self._ns, "core", f"lod{lod}Solid")
        return _q(self._ns, self._prefix, self._profile.solid.format(n=lod))

    def boundary_rank(self) -> tuple[int, int]:
        """The sort rank of the ``boundedBy``/``boundary`` group relative to the geometry properties.

        The content sequence differs by version: CityGML **2.0** places ``boundedBy`` *after* the
        LOD-2 geometry (``_BOUNDED_BY_RANK``), while CityGML **3.0** places ``core:boundary`` *before*
        the space's own ``core:lod{n}Solid``/``MultiSurface`` — so in 3.0 it sorts ahead of everything.
        """
        return (-1, 0) if self._is_30 else _BOUNDED_BY_RANK

    def solid_shell_tag(self) -> str:
        """The ``gml:Solid``'s exterior shell element: GML 3.2 ``gml:Shell`` (3.0) else ``CompositeSurface``.

        GML 3.1.1 wraps a solid's exterior surfaces in a ``gml:CompositeSurface``; GML 3.2 (CityGML 3.0)
        wraps them in a ``gml:Shell``. Both then hold ``gml:surfaceMember`` polygons.
        """
        return _q(self._ns, "gml", "Shell" if self._is_30 else "CompositeSurface")


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
    plan: _GeometryPlan,
    semantic_type: str,
    surface: list[Any],
    lod: int,
    targets: _Appearances,
    material_by_theme: dict[str, int],
    texture_by_theme: dict[str, list[Any]],
) -> ElementTree.Element | None:
    """A boundary wrapper (``<module>:boundedBy`` / 3.0 ``core:boundary``) around one boundary surface.

    The wrapper, surface element and geometry property all come from ``plan``, so the surface lands in
    its module's namespace (``bldg:``/``brid:``/``tun:``, or 3.0's ``con:`` + ``core:``), not a
    hardcoded ``bldg:``.
    """
    polygon = _polygon_element(ns, surface)
    if polygon is None:
        return None
    targets.paint(ns, polygon, surface, material_by_theme, texture_by_theme)
    bounded_by = ElementTree.Element(plan.boundary_wrapper())
    boundary_surface = ElementTree.SubElement(
        bounded_by, plan.surface_element(semantic_type)
    )
    multi_surface_property = ElementTree.SubElement(
        boundary_surface, plan.surface_geometry(lod)
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


def _text(value: Any) -> str:
    """The element text for an attribute value (``repr`` for a float, ``str`` otherwise)."""
    return repr(value) if isinstance(value, float) else str(value)


def _add_attributes(
    ns: dict[str, str],
    feature: ElementTree.Element,
    attributes: dict[str, Any],
    units: dict[str, str] | None = None,
    native_ok: bool = True,
) -> None:
    """Append an object's attributes to ``feature`` in schema order.

    The generic ``gen:*Attribute`` elements are inherited from ``AbstractCityObject`` and must precede
    the module's own (native) attributes in the content sequence, so **all generics are emitted first**,
    then the native ``bldg:`` numeric attributes. A native attribute is written as its ``bldg:`` element
    only for a building-family feature (``native_ok``); on any other object type ``bldg:measuredHeight``
    would be invalid, so it goes out as a generic instead. A non-native attribute that carries a unit is
    written as a ``gen:measureAttribute`` (whose ``gml:MeasureType`` value legally holds the ``uom`` — a
    plain ``gen:doubleAttribute`` cannot), so the unit round-trips; a native one re-emits its ``uom`` on
    the ``bldg:`` element.
    """
    units = units or {}
    native: list[tuple[str, Any]] = []
    for name, value in attributes.items():
        if native_ok and name in _NATIVE_BUILDING_ATTRIBUTES:
            native.append((name, value))
        else:
            _emit_generic_attribute(ns, feature, name, value, units.get(name))
    for name, value in native:
        element = ElementTree.SubElement(feature, _q(ns, "bldg", name))
        element.text = _text(value)
        if name in units:
            element.set("uom", units[name])


def _generic_attribute_kind(value: Any) -> str:
    """The ``gen:*Attribute`` element for a value's Python type (``bool`` stays a string)."""
    if isinstance(
        value, bool
    ):  # bool is an int subclass — keep it a string, not an intAttribute
        return "stringAttribute"
    if isinstance(value, int):
        return "intAttribute"
    if isinstance(value, float):
        return "doubleAttribute"
    return "stringAttribute"


def _emit_generic_attribute(
    ns: dict[str, str],
    feature: ElementTree.Element,
    name: str,
    value: Any,
    unit: str | None,
) -> None:
    """Append one ``gen:*Attribute``: a ``measureAttribute`` when a numeric value carries a unit, else
    the typed kind for its Python type."""
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    if unit is not None and numeric:
        # gen:measureAttribute is the only generic kind whose value (gml:MeasureType) holds a uom.
        kind, with_uom = "measureAttribute", True
    else:
        kind, with_uom = _generic_attribute_kind(value), False
    generic = ElementTree.SubElement(feature, _q(ns, "gen", kind))
    generic.set("name", name)
    value_element = ElementTree.SubElement(generic, _q(ns, "gen", "value"))
    if with_uom:
        value_element.set("uom", unit or "")
    value_element.text = _text(value)


#: The schema-sequence rank of a ``boundedBy`` group in a construction content model: it sits after the
#: LOD-2 geometry and before LOD-3, so it is ranked as the end of the LOD-2 group. Plain geometry at
#: LOD ≤ 2 then sorts before it, LOD ≥ 3 after it — matching ``AbstractBuildingType``.
_BOUNDED_BY_RANK = (2, 9)


def _build_plain_multisurface(
    ns: dict[str, str],
    plan: _GeometryPlan,
    polygons: list[ElementTree.Element],
    lod: int,
) -> ElementTree.Element:
    """Build **one** geometry property wrapping ``polygons`` in a single ``gml:MultiSurface``.

    All of a feature's plain (non-boundary) surfaces at one LOD share a single property — the module's
    ``lod{n}MultiSurface`` or (city furniture / generics) ``lod{n}Geometry`` slot, which are
    ``maxOccurs="1"``. When the type has no plain-geometry slot (relief, groups), it falls back to
    ``bldg:lod{n}MultiSurface`` and records the fallback so :func:`write_citygml` can warn.
    """
    property_tag = plan.plain_property(lod)
    if property_tag is None:
        plan.note_fallback()
        property_tag = _q(ns, "bldg", f"lod{lod}MultiSurface")
    multi_property = ElementTree.Element(property_tag)
    multi = ElementTree.SubElement(multi_property, _q(ns, "gml", "MultiSurface"))
    for polygon in polygons:
        member = ElementTree.SubElement(multi, _q(ns, "gml", "surfaceMember"))
        member.append(polygon)
    return multi_property


def _build_solid(
    ns: dict[str, str],
    plan: _GeometryPlan,
    geometry: Geometry,
    lod: int,
    targets: _Appearances,
    surface_materials: list[dict[str, int]],
    surface_textures: list[dict[str, list[Any]]],
) -> ElementTree.Element | None:
    """Build a (non-semantic) ``Solid`` as the module's ``lod{n}Solid``, or ``None`` to fall back.

    Emitting the surfaces inside a ``gml:Solid`` preserves the geometry *type*, so a CityJSON ``Solid``
    round-trips as a ``Solid`` (the reader reads a ``lodXSolid`` back as ``type="Solid"`` — §3 gap 2).
    Returns ``None`` when the module has no ``Solid`` slot or the solid has no valid polygon (then the
    caller writes the shell's surfaces as plain geometry instead). A ``MultiSolid`` is not routed here.
    """
    solid_tag = plan.solid_property(lod)
    if solid_tag is None:
        return None
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
        return None
    solid_property = ElementTree.Element(solid_tag)
    solid = ElementTree.SubElement(solid_property, _q(ns, "gml", "Solid"))
    exterior = ElementTree.SubElement(solid, _q(ns, "gml", "exterior"))
    composite = ElementTree.SubElement(exterior, plan.solid_shell_tag())
    for polygon in polygons:
        member = ElementTree.SubElement(composite, _q(ns, "gml", "surfaceMember"))
        member.append(polygon)
    return solid_property


def _collect_one_geometry(
    ns: dict[str, str],
    plan: _GeometryPlan,
    geometry: Geometry,
    targets: _Appearances,
) -> tuple[
    int,
    ElementTree.Element | None,
    list[ElementTree.Element],
    list[ElementTree.Element],
]:
    """Classify one geometry's surfaces: ``(lod, solid_or_None, boundary_surfaces, plain_polygons)``.

    A non-semantic ``Solid`` with a module solid slot yields the ``solid`` element (and no surfaces).
    Otherwise each surface is split: one the module can carry becomes a ``boundedBy`` boundary surface;
    any other (plain, or a demoted typed surface — recorded so the writer warns) becomes a plain
    polygon. :func:`_append_feature_geometry` aggregates and orders these across all of a feature's
    geometries.
    """
    lod = _lod_number(geometry)
    materials = _surface_materials(geometry)
    textures = _surface_textures(geometry)
    if geometry.type == "Solid" and not geometry.semantics:
        solid = _build_solid(ns, plan, geometry, lod, targets, materials, textures)
        if solid is not None:
            return lod, solid, [], []
        # No Solid slot — fall through and emit the shell's surfaces as plain geometry.
    bounded: list[ElementTree.Element] = []
    plain: list[ElementTree.Element] = []
    for position, (surface, semantic_value) in enumerate(
        iter_surfaces(geometry.boundaries, geometry.semantic_values)
    ):
        material = materials[position] if position < len(materials) else {}
        texture = textures[position] if position < len(textures) else {}
        surface_object = geometry.semantic_for(semantic_value)
        _classify_surface(
            ns,
            plan,
            surface,
            surface_object,
            lod,
            targets,
            material,
            texture,
            bounded,
            plain,
        )
    return lod, None, bounded, plain


def _classify_surface(
    ns: dict[str, str],
    plan: _GeometryPlan,
    surface: list[Any],
    surface_object: Any,
    lod: int,
    targets: _Appearances,
    material: dict[str, int],
    texture: dict[str, list[Any]],
    bounded: list[ElementTree.Element],
    plain: list[ElementTree.Element],
) -> None:
    """Route one surface: a ``boundedBy`` boundary surface the module can carry, else a plain polygon.

    A semantic surface the module cannot carry as a boundary surface is demoted to ``plain`` (its type
    recorded via :meth:`_GeometryPlan.note_demoted` so the writer warns).
    """
    if surface_object is not None and plan.supports_surface(surface_object.type):
        bounded_by = _bounded_by(
            ns, plan, surface_object.type, surface, lod, targets, material, texture
        )
        if bounded_by is not None:
            bounded.append(bounded_by)
        return
    if surface_object is not None:
        plan.note_demoted()  # a typed surface the module can't carry → plain, semantic lost
    polygon = _polygon_element(ns, surface)
    if polygon is None:
        return
    targets.paint(ns, polygon, surface, material, texture)
    plain.append(polygon)


def _append_feature_geometry(
    ns: dict[str, str],
    plan: _GeometryPlan,
    feature: ElementTree.Element,
    geometries: list[Geometry],
    targets: _Appearances,
) -> None:
    """Emit **all** of a feature's geometries, in CityGML's content-model order.

    CityGML's construction content model is a *sequence*: the ``lod{n}Solid`` / ``lod{n}MultiSurface``
    geometry properties come **before** the ``boundedBy`` boundary surfaces, and each plain property is
    ``maxOccurs="1"``. So this collects across every geometry — one aggregated plain property per LOD
    (never a duplicate), the solids, and the boundary surfaces — then appends them ordered by their
    schema rank (geometry by LOD, ``boundedBy`` after the LOD-2 geometry). A non-semantic ``Solid`` uses
    the module's ``lod{n}Solid`` when it has one (type preserved — §3 gap 2), else its shell is written
    as plain surfaces. A surface whose semantic type the module cannot carry as a boundary surface is
    demoted to plain geometry (its semantic type dropped — recorded so :func:`write_citygml` warns).
    """
    bounded: list[ElementTree.Element] = []
    plain_by_lod: dict[int, list[ElementTree.Element]] = {}
    solids: list[tuple[int, ElementTree.Element]] = []
    for geometry in geometries:
        lod, solid, geometry_bounded, geometry_plain = _collect_one_geometry(
            ns, plan, geometry, targets
        )
        if solid is not None:
            solids.append((lod, solid))
        bounded.extend(geometry_bounded)
        if geometry_plain:
            plain_by_lod.setdefault(lod, []).extend(geometry_plain)
    ordered: list[tuple[tuple[int, int], ElementTree.Element]] = [
        ((lod, 0), solid) for lod, solid in solids
    ]
    ordered += [
        ((lod, 1), _build_plain_multisurface(ns, plan, polygons, lod))
        for lod, polygons in plain_by_lod.items()
    ]
    boundary_rank = plan.boundary_rank()
    ordered += [(boundary_rank, bounded_by) for bounded_by in bounded]
    ordered.sort(
        key=lambda item: item[0]
    )  # stable → boundedBy keep their insertion order
    for _rank, element in ordered:
        feature.append(element)


def _append_children(
    ns: dict[str, str],
    feature: ElementTree.Element,
    city_object: CityObject,
    model: CityModel,
    visited: set[str],
    targets: _Appearances,
    diagnostics: _WriteDiagnostics,
) -> None:
    """Nest a part child under its ``consistsOf…Part`` wrapper in the parent's module, else attach it."""
    for child_id in city_object.children:
        child = model.objects.get(child_id)
        if child is None or child_id in visited:
            continue
        child_element = _feature_element(
            ns, child, model, visited, targets, diagnostics
        )
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
    diagnostics: _WriteDiagnostics,
) -> ElementTree.Element:
    """Build one feature element in its thematic namespace: id, attributes, surfaces and children."""
    prefix = _TYPE_NAMESPACE_PREFIX.get(city_object.type, "bldg")
    profile = _TYPE_GEOMETRY_PROFILE.get(city_object.type, _CONSTRUCTION_PROFILE)
    plan = _GeometryPlan(ns, prefix, profile, city_object.type, diagnostics)
    feature = ElementTree.Element(_q(ns, prefix, city_object.type))
    feature.set(_q(ns, "gml", "id"), city_object.id)
    visited.add(city_object.id)
    _add_attributes(
        ns,
        feature,
        city_object.attributes,
        city_object.attribute_units,
        native_ok=prefix == "bldg",
    )
    _append_feature_geometry(ns, plan, feature, city_object.geometries, targets)
    _append_children(ns, feature, city_object, model, visited, targets, diagnostics)
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


def _resolve_version(model: CityModel) -> str:
    """The CityGML version to emit, warning for an unknown version and for unreadable object types.

    A ``1.0``/``2.0``/``3.0`` source is emitted natively (its version round-trips); any other falls
    back to valid ``2.0`` with a warning. A separate warning lists object types with no known thematic
    namespace (written under ``bldg:``).
    """
    if model.version is not None and model.version not in _MATCHED_VERSIONS:
        warnings.warn(
            f"CityGML writer: a {model.version} source has no known emission structure and is "
            "written as valid CityGML 2.0, so the CityGML version does not round-trip",
            stacklevel=3,
        )
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
            stacklevel=3,
        )
    return model.version if model.version in _MATCHED_VERSIONS else "2.0"


def _warn_diagnostics(diagnostics: _WriteDiagnostics) -> None:
    """Emit one warning each for the geometry fallbacks and the demoted semantic surfaces collected."""
    if diagnostics.fallbacks:
        warnings.warn(
            "CityGML writer: object type(s) "
            f"{', '.join(sorted(diagnostics.fallbacks))} have no geometry slot in their thematic "
            "module for the supplied geometry and are written with a bldg: geometry property (not "
            "schema-valid), so their geometry will not round-trip conformantly",
            stacklevel=3,
        )
    if diagnostics.demoted:
        warnings.warn(
            "CityGML writer: object type(s) "
            f"{', '.join(sorted(diagnostics.demoted))} carry a semantic boundary surface their "
            "thematic module cannot represent; it is written as plain geometry and its semantic "
            "surface type is dropped",
            stacklevel=3,
        )


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
    # The map is a per-call local threaded through construction — no module-global is mutated, so
    # concurrent writes of different-version models cannot corrupt each other.
    version = _resolve_version(model)
    ns = _namespaces(version)
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
    # Per-feature notes collected during emission (geometry with no module slot → bldg: fallback; a
    # semantic surface a module can't carry → demoted to plain geometry), each warned once below.
    diagnostics = _WriteDiagnostics()
    for city_object in model.objects.values():
        if city_object.parents:
            continue  # emitted inside its parent
        member = ElementTree.SubElement(root, _q(ns, "core", "cityObjectMember"))
        member.append(
            _feature_element(ns, city_object, model, visited, targets, diagnostics)
        )
    _warn_diagnostics(diagnostics)

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
