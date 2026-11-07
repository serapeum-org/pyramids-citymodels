"""Serialize the IR to KML (feature #14 — mesh / handoff export targets).

KML is the Google-Earth handoff format: each city object becomes a ``<Placemark>`` whose
``<MultiGeometry>`` holds one ``<Polygon>`` per surface (exterior ring in ``<outerBoundaryIs>``,
holes in ``<innerBoundaryIs>``). Only the standard library (``xml.etree.ElementTree``) is used, so
there is no new dependency — the same contract the CityGML writer keeps.

Deliberate, documented limits (plain geometry, like the OBJ/glTF writers):

- **No reprojection.** KML coordinates are ``lon,lat,alt``; this writer emits the model's own
  ``x,y,z`` verbatim (reprojection belongs to pyramids, ADR-0003). The output is correct for a
  geographic (lon/lat) CRS and a raw placement for a projected one.
- **No appearance** — geometry only; materials/textures round-trip through CityJSON/CityGML.
- An empty model yields a valid, empty ``<kml><Document/></kml>``.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any
from xml.etree import ElementTree

from pyramids_citymodels.ir import CityModel, iter_surfaces
from pyramids_citymodels.writers._geometry import open_ring, xyz

__all__ = ["write_kml"]

#: The KML 2.2 namespace, emitted as a literal ``xmlns`` on the root so elements carry no prefix
#: (ElementTree's global default-namespace registration would collide with the COLLADA writer's).
_KML_NS = "http://www.opengis.net/kml/2.2"


def _coordinates(ring: list[Any]) -> str:
    """A KML ``coordinates`` string for a ring — ``lon,lat,alt`` triples, closed (first repeated)."""
    points = [xyz(coordinate) for coordinate in open_ring(ring)]
    if len(points) >= 3:
        points = [*points, points[0]]  # KML LinearRings are closed
    return " ".join(f"{x},{y},{z}" for x, y, z in points)


def _append_polygon(multi: ElementTree.Element, surface: list[Any]) -> None:
    """Append one ``<Polygon>`` (exterior + interior rings) for an IR surface to a MultiGeometry."""
    rings = [ring for ring in surface if ring and len(ring) >= 3]
    if not rings:
        return
    polygon = ElementTree.SubElement(multi, "Polygon")
    for position, ring in enumerate(rings):
        boundary = ElementTree.SubElement(
            polygon, "outerBoundaryIs" if position == 0 else "innerBoundaryIs"
        )
        linear_ring = ElementTree.SubElement(boundary, "LinearRing")
        coordinates = ElementTree.SubElement(linear_ring, "coordinates")
        coordinates.text = _coordinates(ring)


def write_kml(model: CityModel, path: Any = None) -> str | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to KML.

    Each object becomes a ``<Placemark>`` (named by its id) whose ``<MultiGeometry>`` carries one
    ``<Polygon>`` per surface. See the module docstring for the documented limits.

    Args:
        model: The city model to serialize.
        path: A path to write the ``.kml`` to, or ``None`` to return the document string.

    Returns:
        The KML document as a ``str`` when ``path`` is ``None``; otherwise ``None``.

    Examples:
        - A one-surface building becomes a Placemark with a Polygon:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> kml = model.to_kml()
            >>> "<Placemark>" in kml and "<Polygon>" in kml
            True

            ```
    """
    kml = ElementTree.Element("kml")
    kml.set("xmlns", _KML_NS)
    document = ElementTree.SubElement(kml, "Document")
    for city_object in model.objects.values():
        placemark = ElementTree.SubElement(document, "Placemark")
        name = ElementTree.SubElement(placemark, "name")
        name.text = city_object.id
        multi = ElementTree.SubElement(placemark, "MultiGeometry")
        for geometry in city_object.geometries:
            for surface, _value in iter_surfaces(
                geometry.boundaries, geometry.semantic_values
            ):
                _append_polygon(multi, surface)

    ElementTree.indent(kml)
    text = ElementTree.tostring(kml, encoding="unicode", xml_declaration=True)
    if path is None:
        return text
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return None
