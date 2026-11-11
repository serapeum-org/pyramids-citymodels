"""Serialize the IR to COLLADA 1.4.1 (`.dae`) (feature #14 — mesh / handoff export targets).

COLLADA is the DCC-tool interchange mesh format. This writer triangulates every surface's exterior
ring (a fan from its first vertex) into one positions-only mesh and emits a single `.dae`: a
`library_geometries` of one triangle mesh instanced from a `library_visual_scenes`. Only the standard
library (`xml.etree.ElementTree`) is used, so there is no new dependency.

Deliberate, documented limits (plain geometry, like the OBJ/glTF writers):

- **Exterior rings only**, fan-triangulated — interior rings (holes) are dropped, and a strongly
  non-convex polygon's fan can self-overlap (fan triangulation is not ear-clipping).
- **Positions only by default; opt-in materials/textures** (``materials=True``) — one ``<triangles>``
  per ``(material, texture)`` bound through ``library_materials`` / ``library_effects``, with the
  diffuse ``<color>`` from the CityJSON ``diffuseColor`` and, when a surface is textured, a ``TEXCOORD``
  input + a sampler over a ``library_images`` image referenced by URI. No normals. Off by default the
  mesh is positions-only; appearance also round-trips through the CityJSON/CityGML writers. Coordinates
  are the model's own (no reprojection).
- An empty model yields a valid `.dae` with a geometry-less scene and no `library_geometries` (that
  element is omitted rather than emitted empty, which the COLLADA 1.4.1 XSD would reject).
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from collections.abc import Callable
from typing import Any
from xml.etree import ElementTree

from pyramids_citymodels.ir import CityModel, iter_surfaces
from pyramids_citymodels.writers import _appearance
from pyramids_citymodels.writers._geometry import open_ring, xyz

__all__ = ["write_collada"]

#: The COLLADA 1.4.1 schema namespace, emitted as a literal ``xmlns`` on the root so elements carry
#: no prefix (a global default-namespace registration would collide with the KML writer's).
_DAE_NS = "http://www.collada.org/2005/11/COLLADASchema"


def _fan_surface(
    surface: list[Any],
    intern: "Callable[[tuple[float, float, float]], int]",
    triangles: list[int],
) -> None:
    """Fan-triangulate one surface's exterior ring, appending flat triangle indices to ``triangles``."""
    if not surface:
        return
    ring = [xyz(coordinate) for coordinate in open_ring(surface[0])]
    if len(ring) < 3:
        return
    indices = [intern(point) for point in ring]
    for apex in range(1, len(indices) - 1):  # fan from the first vertex
        triangles.extend((indices[0], indices[apex], indices[apex + 1]))


def _mesh(model: CityModel) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Fan-triangulate every surface's exterior ring into a (positions, flat triangle indices) pair."""
    positions: list[tuple[float, float, float]] = []
    pool: dict[tuple[float, float, float], int] = {}
    triangles: list[int] = []

    def intern(point: tuple[float, float, float]) -> int:
        index = pool.get(point)
        if index is None:
            index = len(positions)
            pool[point] = index
            positions.append(point)
        return index

    for city_object in model.objects.values():
        for geometry in city_object.geometries:
            for surface, _value in iter_surfaces(
                geometry.boundaries, geometry.semantic_values
            ):
                _fan_surface(surface, intern, triangles)
    return positions, triangles


def _geometry_library(
    collada: ElementTree.Element,
    positions: list[tuple[float, float, float]],
    triangles: list[int],
) -> None:
    """Append `library_geometries` with one positions mesh, and the instancing scene libraries."""
    # COLLADA's <COLLADA> sequence is asset, (library_*)*, scene — so every library element must
    # precede <scene>. The geometry library is emitted only for a non-empty model: the 1.4.1 XSD
    # requires <library_geometries> to hold at least one <geometry>, so an empty one would be invalid
    # (the library is optional, and a scene with a geometry-less node is valid on its own).
    if triangles:
        library = ElementTree.SubElement(collada, "library_geometries")
        geometry = ElementTree.SubElement(library, "geometry")
        geometry.set("id", "mesh0")
        mesh = ElementTree.SubElement(geometry, "mesh")
        source = ElementTree.SubElement(mesh, "source")
        source.set("id", "mesh0-positions")
        float_array = ElementTree.SubElement(source, "float_array")
        float_array.set("id", "mesh0-positions-array")
        float_array.set("count", str(len(positions) * 3))
        float_array.text = " ".join(
            repr(component) for point in positions for component in point
        )
        technique = ElementTree.SubElement(source, "technique_common")
        accessor = ElementTree.SubElement(technique, "accessor")
        accessor.set("source", "#mesh0-positions-array")
        accessor.set("count", str(len(positions)))
        accessor.set("stride", "3")
        for axis in ("X", "Y", "Z"):
            param = ElementTree.SubElement(accessor, "param")
            param.set("name", axis)
            param.set("type", "float")
        vertices = ElementTree.SubElement(mesh, "vertices")
        vertices.set("id", "mesh0-vertices")
        vertex_input = ElementTree.SubElement(vertices, "input")
        vertex_input.set("semantic", "POSITION")
        vertex_input.set("source", "#mesh0-positions")
        triangles_element = ElementTree.SubElement(mesh, "triangles")
        triangles_element.set("count", str(len(triangles) // 3))
        triangle_input = ElementTree.SubElement(triangles_element, "input")
        triangle_input.set("semantic", "VERTEX")
        triangle_input.set("source", "#mesh0-vertices")
        triangle_input.set("offset", "0")
        primitives = ElementTree.SubElement(triangles_element, "p")
        primitives.text = " ".join(str(index) for index in triangles)

    scenes = ElementTree.SubElement(collada, "library_visual_scenes")
    visual_scene = ElementTree.SubElement(scenes, "visual_scene")
    visual_scene.set("id", "scene0")
    node = ElementTree.SubElement(visual_scene, "node")
    node.set("id", "node0")
    if triangles:
        ElementTree.SubElement(node, "instance_geometry").set("url", "#mesh0")
    scene = ElementTree.SubElement(collada, "scene")
    ElementTree.SubElement(scene, "instance_visual_scene").set("url", "#scene0")


class _Group:
    """One ``(material, texture)`` appearance group's triangles + its material/texture binding."""

    def __init__(
        self, material_index: int | None, texture_index: int | None, image: str | None
    ) -> None:
        self.material_index = material_index
        self.texture_index = texture_index
        self.image = image  # the texture image URI, or None (untextured)
        self.symbol: str | None = (
            None  # the <triangles material=...> symbol, or None for plain geometry
        )
        self.corners: list[
            tuple[int, ...]
        ] = []  # per triangle-corner (pos,) or (pos, uv)

    @property
    def textured(self) -> bool:
        return self.image is not None


def _appearance_mesh(
    model: CityModel,
) -> tuple[list[tuple[float, float, float]], list[Any], list[_Group]]:
    """Fan-triangulate the model into shared ``positions`` + ``uvs`` and one :class:`_Group` per appearance.

    Positions and UVs are deduplicated across the whole model; each group records its triangle corners
    as ``(position_index[, uv_index])`` tuples and the material/texture it binds.
    """
    textures_pool = model.appearances.get("textures") or []
    uv_pool = model.appearances.get("vertices-texture") or []
    positions: list[tuple[float, float, float]] = []
    uvs: list[Any] = []
    pos_index: dict[tuple[float, float, float], int] = {}
    uv_index: dict[Any, int] = {}

    def position(coordinate: Any) -> int:
        key = xyz(coordinate)
        index = pos_index.get(key)
        if index is None:
            index = len(positions)
            pos_index[key] = index
            positions.append(key)
        return index

    def texcoord(uv: Any) -> int:
        index = uv_index.get(uv)
        if index is None:
            index = len(uvs)
            uv_index[uv] = index
            uvs.append(uv)
        return index

    groups: list[_Group] = []
    for (material_index, texture_index), rings in _appearance.collect_groups(
        model
    ).items():
        image = _appearance.texture_image(texture_index, textures_pool)
        group = _Group(material_index, texture_index, image)
        if material_index is not None or texture_index is not None:
            group.symbol = f"sym_{len(groups)}"
        for ring, uv_indices in rings:
            corners: list[tuple[int, ...]] = []
            for vertex, coordinate in enumerate(ring):
                if group.textured and uv_indices is not None:
                    uv = _appearance.ring_uv(uv_pool, uv_indices, vertex)
                    corners.append((position(coordinate), texcoord(uv)))
                else:
                    corners.append((position(coordinate),))
            for apex in range(1, len(corners) - 1):  # fan from the first vertex
                group.corners.extend((corners[0], corners[apex], corners[apex + 1]))
        groups.append(group)
    return positions, uvs, groups


def _float_source(
    mesh: ElementTree.Element, sid: str, values: list[Any], params: tuple[str, ...]
) -> None:
    """A COLLADA ``<source>`` of a flat float array with a ``technique_common`` accessor over ``params``."""
    stride = len(params)
    source = ElementTree.SubElement(mesh, "source")
    source.set("id", sid)
    array = ElementTree.SubElement(source, "float_array")
    array.set("id", f"{sid}-array")
    array.set("count", str(len(values) * stride))
    array.text = " ".join(repr(component) for value in values for component in value)
    accessor = ElementTree.SubElement(
        ElementTree.SubElement(source, "technique_common"), "accessor"
    )
    accessor.set("source", f"#{sid}-array")
    accessor.set("count", str(len(values)))
    accessor.set("stride", str(stride))
    for name in params:
        param = ElementTree.SubElement(accessor, "param")
        param.set("name", name)
        param.set("type", "float")


def _effect(effects: ElementTree.Element, group: _Group, model: CityModel) -> None:
    """One ``<effect>`` for a group: a lambert whose diffuse is the base colour, or the bound texture."""
    materials_pool = model.appearances.get("materials") or []
    color = _appearance.base_color(
        _appearance.diffuse_of(group.material_index, materials_pool)
    )
    effect = ElementTree.SubElement(effects, "effect")
    effect.set("id", f"eff_{group.symbol}")
    profile = ElementTree.SubElement(effect, "profile_COMMON")
    if group.textured:
        surface_param = ElementTree.SubElement(profile, "newparam")
        surface_param.set("sid", f"{group.symbol}-surface")
        surface = ElementTree.SubElement(surface_param, "surface")
        surface.set("type", "2D")
        ElementTree.SubElement(surface, "init_from").text = f"img_{group.texture_index}"
        sampler_param = ElementTree.SubElement(profile, "newparam")
        sampler_param.set("sid", f"{group.symbol}-sampler")
        sampler = ElementTree.SubElement(sampler_param, "sampler2D")
        ElementTree.SubElement(sampler, "source").text = f"{group.symbol}-surface"
    technique = ElementTree.SubElement(profile, "technique")
    technique.set("sid", "common")
    diffuse = ElementTree.SubElement(
        ElementTree.SubElement(technique, "lambert"), "diffuse"
    )
    if group.textured:
        texture = ElementTree.SubElement(diffuse, "texture")
        texture.set("texture", f"{group.symbol}-sampler")
        texture.set("texcoord", "UVSET0")
    else:
        ElementTree.SubElement(diffuse, "color").text = " ".join(repr(c) for c in color)


def _material_libraries(
    collada: ElementTree.Element, groups: list[_Group], model: CityModel
) -> None:
    """Append ``<library_images>`` / ``<library_effects>`` / ``<library_materials>`` for the groups."""
    appearance_groups = [group for group in groups if group.symbol is not None]
    if not appearance_groups:
        return
    images = {
        group.texture_index
        for group in appearance_groups
        if group.textured and group.texture_index is not None
    }
    if images:
        library_images = ElementTree.SubElement(collada, "library_images")
        textures_pool = model.appearances.get("textures") or []
        for texture_index in sorted(images):
            image = ElementTree.SubElement(library_images, "image")
            image.set("id", f"img_{texture_index}")
            ElementTree.SubElement(image, "init_from").text = _appearance.texture_image(
                texture_index, textures_pool
            )
    effects = ElementTree.SubElement(collada, "library_effects")
    materials = ElementTree.SubElement(collada, "library_materials")
    for group in appearance_groups:
        _effect(effects, group, model)
        material = ElementTree.SubElement(materials, "material")
        material.set("id", f"mat_{group.symbol}")
        ElementTree.SubElement(material, "instance_effect").set(
            "url", f"#eff_{group.symbol}"
        )


def _triangles_element(mesh: ElementTree.Element, group: _Group) -> None:
    """One ``<triangles>`` for a group — VERTEX (+ TEXCOORD when textured), bound to its material symbol."""
    triangles = ElementTree.SubElement(mesh, "triangles")
    triangles.set("count", str(len(group.corners) // 3))
    if group.symbol is not None:
        triangles.set("material", group.symbol)
    vertex_input = ElementTree.SubElement(triangles, "input")
    vertex_input.set("semantic", "VERTEX")
    vertex_input.set("source", "#mesh0-vertices")
    vertex_input.set("offset", "0")
    if group.textured:
        uv_input = ElementTree.SubElement(triangles, "input")
        uv_input.set("semantic", "TEXCOORD")
        uv_input.set("source", "#mesh0-uvs")
        uv_input.set("offset", "1")
        uv_input.set("set", "0")
    ElementTree.SubElement(triangles, "p").text = " ".join(
        str(index) for corner in group.corners for index in corner
    )


def _textured_geometry_library(
    collada: ElementTree.Element,
    positions: list[tuple[float, float, float]],
    uvs: list[Any],
    groups: list[_Group],
) -> None:
    """Append ``<library_geometries>`` (one mesh, per-group ``<triangles>``) and the instancing scene."""
    drawable = [group for group in groups if group.corners]
    if drawable:
        geometry = ElementTree.SubElement(
            ElementTree.SubElement(collada, "library_geometries"), "geometry"
        )
        geometry.set("id", "mesh0")
        mesh = ElementTree.SubElement(geometry, "mesh")
        _float_source(mesh, "mesh0-positions", positions, ("X", "Y", "Z"))
        if uvs:
            _float_source(mesh, "mesh0-uvs", uvs, ("S", "T"))
        vertices = ElementTree.SubElement(mesh, "vertices")
        vertices.set("id", "mesh0-vertices")
        position_input = ElementTree.SubElement(vertices, "input")
        position_input.set("semantic", "POSITION")
        position_input.set("source", "#mesh0-positions")
        for group in drawable:
            _triangles_element(mesh, group)

    scenes = ElementTree.SubElement(collada, "library_visual_scenes")
    visual_scene = ElementTree.SubElement(scenes, "visual_scene")
    visual_scene.set("id", "scene0")
    node = ElementTree.SubElement(visual_scene, "node")
    node.set("id", "node0")
    if drawable:
        instance = ElementTree.SubElement(node, "instance_geometry")
        instance.set("url", "#mesh0")
        _bind_material(instance, drawable)
    scene = ElementTree.SubElement(collada, "scene")
    ElementTree.SubElement(scene, "instance_visual_scene").set("url", "#scene0")


def _bind_material(instance: ElementTree.Element, groups: list[_Group]) -> None:
    """Bind each group's ``material`` symbol to its ``<material>`` under ``<instance_geometry>``."""
    bound = [group for group in groups if group.symbol is not None]
    if not bound:
        return
    technique = ElementTree.SubElement(
        ElementTree.SubElement(instance, "bind_material"), "technique_common"
    )
    for group in bound:
        symbol = (
            group.symbol or ""
        )  # bound groups have a symbol; `or ""` only narrows the type
        instance_material = ElementTree.SubElement(technique, "instance_material")
        instance_material.set("symbol", symbol)
        instance_material.set("target", f"#mat_{symbol}")
        if group.textured:
            bind = ElementTree.SubElement(instance_material, "bind_vertex_input")
            bind.set("semantic", "UVSET0")
            bind.set("input_semantic", "TEXCOORD")
            bind.set("input_set", "0")


def write_collada(
    model: CityModel, path: Any = None, materials: bool = False
) -> str | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to a COLLADA 1.4.1 ``.dae`` document.

    With ``materials=False`` (default) every surface's exterior ring is fan-triangulated into one
    positions-only mesh. With ``materials=True`` the IR appearances ride along: surfaces are grouped by
    ``(material, texture)`` into per-material ``<triangles>`` bound through ``<library_materials>`` /
    ``<library_effects>``, a material's diffuse ``<color>`` comes from the CityJSON ``diffuseColor``, and
    a textured group adds a ``TEXCOORD`` input (from the ``vertices-texture`` pool) plus a sampler over a
    ``<library_images>`` image referenced by its CityJSON ``image`` URI. A model with no appearances is
    unchanged. See the module docstring for the geometry limits.

    Args:
        model: The city model to serialize.
        path: A path to write the ``.dae`` to, or ``None`` to return the document string.
        materials: Carry IR appearances (materials + textures) instead of a positions-only mesh.

    Returns:
        The COLLADA document as a ``str`` when ``path`` is ``None``; otherwise ``None``.

    Examples:
        - A one-triangle model yields one COLLADA triangle:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> "<triangles count=\\"1\\"" in model.to_collada()
            True

            ```
    """
    collada = ElementTree.Element("COLLADA")
    collada.set("xmlns", _DAE_NS)
    collada.set("version", "1.4.1")
    asset = ElementTree.SubElement(collada, "asset")
    up_axis = ElementTree.SubElement(asset, "up_axis")
    up_axis.text = "Z_UP"
    if materials:
        positions, uvs, groups = _appearance_mesh(model)
        _material_libraries(collada, groups, model)
        _textured_geometry_library(collada, positions, uvs, groups)
    else:
        positions, triangles = _mesh(model)
        _geometry_library(collada, positions, triangles)

    ElementTree.indent(collada)
    text = ElementTree.tostring(collada, encoding="unicode", xml_declaration=True)
    if path is None:
        return text
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return None
