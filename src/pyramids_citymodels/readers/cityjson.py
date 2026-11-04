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
from pyramids_citymodels.options import ObjectFilter

__all__ = ["looks_like_cityjson", "parse_cityjson"]


def _extent_2d(extent: Any) -> "tuple[float, float, float, float] | None":
    """Normalize a CityJSON ``geographicalExtent`` to 2D ``(xmin, ymin, xmax, ymax)``, or ``None``.

    CityJSON declares an extent as ``[minx, miny, minz, maxx, maxy, maxz]`` (6 values); a 4-value
    ``[minx, miny, maxx, maxy]`` is accepted too. Anything shorter, or non-numeric, is treated as
    "no declared extent" so the bbox prefilter keeps the object rather than guessing.
    """
    if not isinstance(extent, (list, tuple)) or len(extent) < 4:
        return None
    try:
        values = [float(component) for component in extent]
    except (TypeError, ValueError):
        return None
    if len(values) >= 6:
        return (values[0], values[1], values[3], values[4])
    return (values[0], values[1], values[2], values[3])


def _load(source: Any) -> Any:
    """Return the decoded JSON from a path, an open file, a JSON string or a decoded dict.

    A ``str`` whose first non-space character is ``{`` is treated as JSON content; any other
    ``str`` (and any :class:`os.PathLike`) is treated as a filesystem path. This mirrors the
    content-or-path handling of the CityGML reader.
    """
    if isinstance(source, dict):
        return source
    if hasattr(source, "read"):
        return json.load(source)
    if isinstance(source, str) and source.lstrip().startswith("{"):
        return json.loads(source)
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

    Examples:
        - A CityJSON document is recognized, a GeoJSON one sharing the suffix is not:
            ```python
            >>> doc = '{"type": "CityJSON", "version": "2.0", "CityObjects": {}, "vertices": []}'
            >>> looks_like_cityjson(doc)
            True
            >>> looks_like_cityjson('{"type": "FeatureCollection", "features": []}')
            False

            ```
        - Content that is not JSON at all is declined without raising:
            ```python
            >>> looks_like_cityjson("not json at all")
            False

            ```
    """
    try:
        document = _load(source)
    except (OSError, ValueError, TypeError):
        return False
    return isinstance(document, dict) and document.get("type") == "CityJSON"


def _dereference(element: Any, vertices: list[tuple[float, ...]]) -> Any:
    """Replace every vertex index in a boundaries structure with its coordinate tuple.

    Raises:
        ValueError: If a boundary references a vertex index outside ``vertices`` (negative or
            past the end), rather than letting a bare ``IndexError`` or a silently-wrapped
            negative index escape (INF3).
    """
    if isinstance(element, int):
        if not 0 <= element < len(vertices):
            raise ValueError(
                f"CityJSON boundary references vertex index {element} out of range "
                f"(0..{len(vertices) - 1})"
            )
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


def _select_cityjson_ids(
    raw_objects: dict[str, Any], selector: "ObjectFilter | None"
) -> set[str]:
    """The object ids to read, applied to object *headers* before any geometry is dereferenced.

    With no selector every id is kept. Otherwise an object survives when it matches by
    id/type/attribute and is not proven outside the bbox by its declared ``geographicalExtent`` (one
    without an extent is kept and refined downstream); the survivors are then thinned by any random
    sample. This is the parse-time win: a city-scale file is subset without decoding what it
    discards (feature #5, bbox #32).
    """
    if selector is None:
        return set(raw_objects)
    candidates = [
        object_id
        for object_id, raw in raw_objects.items()
        if selector.matches(
            object_id, raw.get("type", "Unknown"), raw.get("attributes") or {}
        )
        and not selector.rejects_extent(_extent_2d(raw.get("geographicalExtent")))
    ]
    return selector.sample_ids(candidates)


def _build_geometry(
    raw_geometry: dict[str, Any], vertices: list[tuple[float, ...]]
) -> Geometry:
    """Build one IR :class:`Geometry` from a CityJSON geometry, dereferencing its boundaries."""
    surfaces, values = _parse_semantics(raw_geometry.get("semantics"))
    lod = raw_geometry.get("lod")
    appearance: dict[str, Any] = {
        key: raw_geometry[key] for key in ("material", "texture") if key in raw_geometry
    }
    return Geometry(
        type=raw_geometry.get("type", "Unknown"),
        lod=None if lod is None else str(lod),
        boundaries=_dereference(raw_geometry.get("boundaries", []), vertices),
        semantics=surfaces,
        semantic_values=values,
        appearance=appearance or None,
    )


def parse_cityjson(source: Any, selector: "ObjectFilter | None" = None) -> CityModel:
    """Parse a CityJSON document into a :class:`~pyramids_citymodels.ir.CityModel`.

    Args:
        source: A path to a ``.city.json``/``.json`` file, an open text file, a JSON string, or
            an already-decoded CityJSON ``dict``.
        selector: An optional :class:`~pyramids_citymodels.options.ObjectFilter`. When given, a
            city object is read only if it matches (by id/type/attributes, and — for an object that
            declares a ``geographicalExtent`` — a bbox prefilter); non-matching objects are skipped
            **before their geometry is dereferenced**, so a large file can be subset without paying
            to decode the objects it discards. ``None`` reads every object.

    Returns:
        The city model with every (kept) boundary dereferenced to transformed coordinate tuples.

    Raises:
        ValueError: If the document is not a CityJSON object.

    Examples:
        - Parse an in-memory CityJSON document, then read its object and a transformed coordinate
          (the ``transform`` turns the quantized index ``1`` into the real point ``(1.0, 0.0, 0.0)``):
            ```python
            >>> doc = {
            ...     "type": "CityJSON", "version": "2.0",
            ...     "transform": {"scale": [0.001, 0.001, 0.001], "translate": [0.0, 0.0, 0.0]},
            ...     "CityObjects": {"b": {"type": "Building", "geometry": [
            ...         {"type": "MultiSurface", "lod": "2", "boundaries": [[[0, 1, 2]]],
            ...          "semantics": {"surfaces": [{"type": "RoofSurface"}], "values": [0]}}]}},
            ...     "vertices": [[0, 0, 0], [1000, 0, 0], [1000, 1000, 0]]}
            >>> model = parse_cityjson(doc)
            >>> model.objects["b"].geometries[0].semantics[0].type
            'RoofSurface'
            >>> model.objects["b"].geometries[0].boundaries[0][0][1]
            (1.0, 0.0, 0.0)

            ```
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
        # Carry the document-level appearance pool and any schema Extensions through untouched, so
        # materials/textures and third-party extensions are never silently dropped (#13).
        appearances=dict(document.get("appearance", {}) or {}),
        extensions=dict(document.get("extensions", {}) or {}),
    )

    raw_objects = document.get("CityObjects", {}) or {}
    kept_ids = _select_cityjson_ids(raw_objects, selector)
    for object_id, raw in raw_objects.items():
        if object_id not in kept_ids:
            continue
        model.objects[object_id] = CityObject(
            id=object_id,
            type=raw.get("type", "Unknown"),
            attributes=dict(raw.get("attributes", {}) or {}),
            geometries=[
                _build_geometry(raw_geometry, vertices)
                for raw_geometry in raw.get("geometry", [])
            ],
            parents=list(raw.get("parents", []) or []),
            children=list(raw.get("children", []) or []),
        )

    return model
