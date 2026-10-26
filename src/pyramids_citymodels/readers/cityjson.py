"""Parse CityJSON into the shared IR (issue #6).

CityJSON stores every vertex once in a flat ``vertices`` array and encodes geometry
``boundaries`` as integer indices into it; coordinates are usually quantized integers that a
``transform`` (per-axis ``scale`` + ``translate``) recovers to reals. Semantics are a parallel
structure: a ``surfaces`` list of semantic objects plus a ``values`` index array that maps each
boundary to one of them.

:func:`parse_cityjson` undoes all of that into the IR: it applies the transform, **dereferences**
boundary indices to coordinate tuples (so downstream code never sees indices), carries the
semantic ``surfaces``/``values`` through, and rebuilds the parent/child object hierarchy. It uses
only the standard-library ``json`` module — no new dependency, matching what cjio, the QGIS
Loader and plateaukit all do.

Versions 1.0, 1.1 and 2.0 share this layout; the version string is carried through on the model.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import json
import os
from typing import Any

from pyramids_citymodels.ir import (
    CityModel,
    CityObject,
    Geometry,
    SemanticSurface,
    crs_from_identifier,
)

__all__ = ["looks_like_cityjson", "parse_cityjson"]


def _load(source: Any) -> Any:
    """Return the decoded JSON from a path, an open file, a string or an already-decoded dict."""
    if isinstance(source, dict):
        return source
    if hasattr(source, "read"):
        return json.load(source)
    if isinstance(source, (str, os.PathLike)):
        with open(source, encoding="utf-8") as handle:
            return json.load(handle)
    raise TypeError(f"cannot read CityJSON from {type(source).__name__}")


def looks_like_cityjson(source: Any) -> bool:
    """Return ``True`` when ``source`` is a CityJSON document (``"type": "CityJSON"``).

    This is the content sniff the driver's ``can_open`` uses to tell CityJSON from GeoJSON for a
    bare ``.json`` file: a GeoJSON ``FeatureCollection`` has ``"type": "FeatureCollection"``, a
    CityJSON file has ``"type": "CityJSON"``. Reading is cheap and never raises — a file that is
    not JSON, or not a JSON object, simply returns ``False``.
    """
    try:
        document = _load(source)
    except (OSError, ValueError, TypeError):
        return False
    return isinstance(document, dict) and document.get("type") == "CityJSON"


def _dereference(element: Any, vertices: list[tuple[float, ...]]) -> Any:
    """Replace every vertex index in a boundaries structure with its coordinate tuple."""
    if isinstance(element, int):
        return vertices[element]
    return [_dereference(child, vertices) for child in element]


def _transform_vertices(
    raw_vertices: list[list[float]], transform: dict[str, Any] | None
) -> list[tuple[float, ...]]:
    """Apply the CityJSON ``transform`` (scale + translate) to the quantized vertex array."""
    if not transform:
        return [
            tuple(float(component) for component in vertex) for vertex in raw_vertices
        ]
    scale = transform.get("scale", [1.0, 1.0, 1.0])
    translate = transform.get("translate", [0.0, 0.0, 0.0])
    return [
        tuple(
            vertex[axis] * scale[axis] + translate[axis] for axis in range(len(vertex))
        )
        for vertex in raw_vertices
    ]


def _parse_semantics(
    semantics: dict[str, Any] | None,
) -> tuple[list[SemanticSurface], Any]:
    """Split a geometry's ``semantics`` into (surface objects, parallel value index)."""
    if not semantics:
        return [], None
    surfaces = []
    for surface in semantics.get("surfaces", []):
        attributes = {
            key: value
            for key, value in surface.items()
            if key not in ("type", "parent", "children")
        }
        surfaces.append(
            SemanticSurface(
                type=surface.get("type", "Unknown"),
                attributes=attributes,
                parent=surface.get("parent"),
                children=list(surface.get("children", [])),
            )
        )
    return surfaces, semantics.get("values")


def parse_cityjson(source: Any) -> CityModel:
    """Parse a CityJSON document into a :class:`~pyramids_citymodels.ir.CityModel`.

    Args:
        source: A path to a ``.city.json``/``.json`` file, an open text file, a JSON string, or
            an already-decoded CityJSON ``dict``.

    Returns:
        The city model with every boundary dereferenced to transformed coordinate tuples.

    Raises:
        ValueError: If the document is not a CityJSON object.
    """
    document = _load(source)
    if not (isinstance(document, dict) and document.get("type") == "CityJSON"):
        raise ValueError("not a CityJSON document (missing 'type': 'CityJSON')")

    vertices = _transform_vertices(
        document.get("vertices", []), document.get("transform")
    )
    metadata = document.get("metadata", {}) or {}
    crs = crs_from_identifier(metadata.get("referenceSystem"))

    model = CityModel(
        crs=crs,
        metadata=metadata,
        version=str(document.get("version"))
        if document.get("version") is not None
        else None,
        source_format="CityJSON",
    )

    for object_id, raw in document.get("CityObjects", {}).items():
        geometries = []
        for raw_geometry in raw.get("geometry", []):
            surfaces, values = _parse_semantics(raw_geometry.get("semantics"))
            lod = raw_geometry.get("lod")
            geometries.append(
                Geometry(
                    type=raw_geometry.get("type", "Unknown"),
                    lod=None if lod is None else str(lod),
                    boundaries=_dereference(
                        raw_geometry.get("boundaries", []), vertices
                    ),
                    semantics=surfaces,
                    semantic_values=values,
                )
            )
        model.objects[object_id] = CityObject(
            id=object_id,
            type=raw.get("type", "Unknown"),
            attributes=dict(raw.get("attributes", {}) or {}),
            geometries=geometries,
            parents=list(raw.get("parents", []) or []),
            children=list(raw.get("children", []) or []),
        )

    return model
