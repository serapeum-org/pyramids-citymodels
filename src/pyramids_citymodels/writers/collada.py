"""Serialize the IR to COLLADA 1.4.1 (`.dae`) (feature #14 — mesh / handoff export targets).

COLLADA is the DCC-tool interchange mesh format. This writer triangulates every surface's exterior
ring (a fan from its first vertex) into one positions-only mesh and emits a single `.dae`: a
`library_geometries` of one triangle mesh instanced from a `library_visual_scenes`. Only the standard
library (`xml.etree.ElementTree`) is used, so there is no new dependency.

Deliberate, documented limits (plain geometry, like the OBJ/glTF writers):

- **Exterior rings only**, fan-triangulated — interior rings (holes) are dropped, and a strongly
  non-convex polygon's fan can self-overlap (fan triangulation is not ear-clipping).
- **One mesh, positions only** — no normals, materials or textures; appearance round-trips through
  the CityJSON/CityGML writers, not here. Coordinates are the model's own (no reprojection).
- An empty model yields a valid `.dae` with an empty geometry library and an empty scene.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any
from xml.etree import ElementTree

from pyramids_citymodels.ir import CityModel, iter_surfaces
from pyramids_citymodels.writers._geometry import open_ring, xyz

__all__ = ["write_collada"]

#: The COLLADA 1.4.1 schema namespace, emitted as a literal ``xmlns`` on the root so elements carry
#: no prefix (a global default-namespace registration would collide with the KML writer's).
_DAE_NS = "http://www.collada.org/2005/11/COLLADASchema"


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
                if not surface:
                    continue
                ring = [xyz(coordinate) for coordinate in open_ring(surface[0])]
                if len(ring) < 3:
                    continue
                indices = [intern(point) for point in ring]
                for apex in range(1, len(indices) - 1):  # fan from the first vertex
                    triangles.extend((indices[0], indices[apex], indices[apex + 1]))
    return positions, triangles


def _geometry_library(
    collada: ElementTree.Element,
    positions: list[tuple[float, float, float]],
    triangles: list[int],
) -> None:
    """Append `library_geometries` with one positions mesh, and the instancing scene libraries."""
    library = ElementTree.SubElement(collada, "library_geometries")
    scenes = ElementTree.SubElement(collada, "library_visual_scenes")
    visual_scene = ElementTree.SubElement(scenes, "visual_scene")
    visual_scene.set("id", "scene0")
    node = ElementTree.SubElement(visual_scene, "node")
    node.set("id", "node0")
    scene = ElementTree.SubElement(collada, "scene")
    ElementTree.SubElement(scene, "instance_visual_scene").set("url", "#scene0")
    if not triangles:
        return  # empty model: a valid scene with no geometry
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
    ElementTree.SubElement(node, "instance_geometry").set("url", "#mesh0")


def write_collada(model: CityModel, path: Any = None) -> str | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to a COLLADA 1.4.1 ``.dae`` document.

    Every surface's exterior ring is fan-triangulated into one positions-only mesh. See the module
    docstring for the documented limits.

    Args:
        model: The city model to serialize.
        path: A path to write the ``.dae`` to, or ``None`` to return the document string.

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
    positions, triangles = _mesh(model)
    _geometry_library(collada, positions, triangles)

    ElementTree.indent(collada)
    text = ElementTree.tostring(collada, encoding="unicode", xml_declaration=True)
    if path is None:
        return text
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return None
