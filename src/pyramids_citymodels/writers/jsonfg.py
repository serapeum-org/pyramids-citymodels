"""Serialize the IR to OGC Features and Geometries JSON (JSON-FG), a thin add-on (issue #27).

JSON-FG is a GeoJSON profile (OGC 21-045) that adds first-class 3D geometry and a feature type: a
``FeatureCollection`` whose features carry a ``place`` member holding the real 3D geometry (GeoJSON
``geometry`` stays 2D or ``null``), a ``featureType`` and a ``coordRefSys``. This writer maps each
:class:`~pyramids_citymodels.ir.CityObject` to one JSON-FG feature whose ``place`` is a 3D
``MultiPolygon`` of the object's surfaces, with its attributes as ``properties``.

Unlike the CityJSON and CityGML writers, this is a **one-way** GIS-exchange projection, not a
round-trippable encoding: there is no JSON-FG reader, and collapsing the per-geometry/LOD structure
and semantics into one flat ``MultiPolygon`` is lossy. It is offered because JSON-FG is the
GeoJSON-compatible way to hand a city model's 3D footprints to a web/GIS consumer. Standard-library
``json`` only — no new dependency.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import json
from typing import Any

from pyramids_citymodels.ir import CityModel, CityObject, iter_surfaces

__all__ = ["write_jsonfg"]

#: The JSON-FG version this writer conforms to (OGC 21-045, the "core" + "3d" conformance classes).
_CONFORMS_TO = [
    "http://www.opengis.net/spec/json-fg-1/0.3",
    "http://www.opengis.net/spec/json-fg-1/0.3/conf/3d",
]


def _coord_ref_sys(model: CityModel) -> str | None:
    """The JSON-FG ``coordRefSys`` (an OGC CRS URI) for the model's CRS, or ``None``.

    An EPSG CRS becomes the OGC URI JSON-FG's examples use (note the ``http`` scheme, per the
    JSON-FG spec — the CityJSON writer uses ``https`` per CityJSON's; both are intentional). A
    non-EPSG identifier is passed through verbatim rather than forced into a malformed EPSG URI.
    """
    if not model.crs:
        return None
    if not model.crs.upper().startswith("EPSG:"):
        return model.crs
    code = model.crs.split(":")[-1]
    return f"http://www.opengis.net/def/crs/EPSG/0/{code}"


def _ring(coordinates: list[Any]) -> list[list[float]]:
    """A closed GeoJSON ring: each coordinate as a ``[x, y, z]`` list, first point repeated last."""
    ring = [[float(component) for component in point] for point in coordinates]
    if len(ring) >= 3 and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return ring


def _polygon(surface: list[Any]) -> list[list[list[float]]] | None:
    """A GeoJSON polygon (exterior ring first, then holes) from an IR surface, or ``None``."""
    rings = [_ring(ring) for ring in surface if ring]
    if not rings or len(rings[0]) < 4:
        return None
    return rings


def _place_geometry(city_object: CityObject) -> dict[str, Any] | None:
    """A 3D ``MultiPolygon`` of every surface across the object's geometries, or ``None``.

    Walks each geometry to surface level (so ``MultiSurface`` and ``Solid`` alike contribute their
    polygons) and collects them into one ``MultiPolygon``. This is the deliberate flattening noted
    in the module docstring: the per-geometry/LOD split and the semantics are not represented.
    """
    polygons: list[list[list[list[float]]]] = []
    for geometry in city_object.geometries:
        for surface, _value in iter_surfaces(
            geometry.boundaries, geometry.semantic_values
        ):
            polygon = _polygon(surface)
            if polygon is not None:
                polygons.append(polygon)
    if not polygons:
        return None
    return {"type": "MultiPolygon", "coordinates": polygons}


def _feature(city_object: CityObject) -> dict[str, Any]:
    """Encode one city object as a JSON-FG feature (3D geometry in ``place``, attrs in ``properties``)."""
    place = _place_geometry(city_object)
    properties = dict(city_object.attributes)
    if city_object.parent is not None:
        properties["parent_id"] = city_object.parent
    return {
        "type": "Feature",
        "id": city_object.id,
        "featureType": city_object.type,
        "geometry": None,
        "place": place,
        "properties": properties,
    }


def write_jsonfg(model: CityModel, path: Any = None) -> dict[str, Any] | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to a JSON-FG ``FeatureCollection``.

    Args:
        model: The city model to serialize.
        path: A filesystem path to write the document to, or ``None`` to return the document dict.

    Returns:
        The JSON-FG document as a ``dict`` when ``path`` is ``None``; otherwise ``None``.

    Note:
        One-way projection: each object becomes one feature with a 3D ``MultiPolygon`` ``place``,
        losing the per-geometry/LOD structure and semantics. There is no JSON-FG reader, so this is
        a terminal conversion, not a round-trip.

    Examples:
        - Each city object becomes one JSON-FG feature with a 3D ``place`` geometry:
            ```python
            >>> from pyramids_citymodels.readers.cityjson import parse_cityjson
            >>> doc = {"type": "CityJSON", "version": "2.0",
            ...     "CityObjects": {"b": {"type": "Building", "geometry": [
            ...         {"type": "MultiSurface", "lod": "2", "boundaries": [[[0, 1, 2]]]}]}},
            ...     "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]}
            >>> out = write_jsonfg(parse_cityjson(doc))
            >>> out["type"], out["features"][0]["featureType"]
            ('FeatureCollection', 'Building')
            >>> out["features"][0]["place"]["type"]
            'MultiPolygon'

            ```
        - Without a CRS, no ``coordRefSys`` is emitted:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject
            >>> doc = write_jsonfg(CityModel(objects={"b": CityObject(id="b", type="Building")}))
            >>> "coordRefSys" in doc
            False

            ```
    """
    document: dict[str, Any] = {
        "type": "FeatureCollection",
        "conformsTo": list(_CONFORMS_TO),
        "features": [_feature(city_object) for city_object in model.objects.values()],
    }
    coord_ref_sys = _coord_ref_sys(model)
    if coord_ref_sys is not None:
        document["coordRefSys"] = coord_ref_sys

    if path is None:
        return document
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    return None
