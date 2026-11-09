"""Serialize the IR to CityJSON 2.0 and CityJSON Text Sequences (issues #24, #25, #27).

This is the mirror of :mod:`pyramids_citymodels.readers.cityjson`, and the inverse of what it
does: the reader dereferences the flat ``vertices`` pool and applies the ``transform`` so the IR
carries real coordinate tuples; the writer **rebuilds** that pool and re-indexes every boundary
back to integer indices (the round-trip-fidelity work of issue #24 — done purely at write time, so
the IR stays format-neutral and carries no CityJSON-specific vertex pool). Semantics
(``surfaces``/``values``), per-geometry appearance references, the document appearance pool,
extensions, metadata, the CRS and the version are all re-emitted from the IR, so
``parse_cityjson(write_cityjson(model))`` reproduces ``model`` — appearance included. The appearance
pool (``materials``, ``textures`` **and** the ``vertices-texture`` UV pool, all members of the
``appearance`` object) is carried through verbatim by the reader and re-emitted here, and a
geometry's ``material`` / ``texture`` value maps are re-emitted beside it, so both material and
texture references stay resolvable across a round-trip.

:func:`write_cityjson` is the public orchestrator (a small set of private ``_encode_*`` helpers
build each part); :meth:`pyramids_citymodels.ir.CityModel.to_cityjson` is the entry point on the
type. :func:`write_cityjsonseq` is the thin CityJSON Text Sequence (``.city.jsonl``) add-on: a
header line followed by one ``CityJSONFeature`` per top-level object, each with its own local
vertex pool. Both use only the standard-library ``json`` module — no new dependency.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import json
from typing import Any

from pyramids_citymodels.ir import CityModel, CityObject, Geometry, is_coordinate

__all__ = ["write_cityjson", "write_cityjsonseq"]

#: The CityJSON version this writer emits. The reader accepts 1.0/1.1/2.0 alike; 2.0 is current.
_DEFAULT_VERSION = "2.0"


class _VertexPool:
    """Accumulate coordinate tuples into a deduplicated, ordered vertex pool.

    The inverse of the reader's dereferencing step: every boundary coordinate is interned to a
    stable integer index (identical coordinates collapse to one vertex, exactly as a hand-written
    CityJSON file shares them), and :meth:`reindex` rewrites a dereferenced boundaries structure
    back to those indices. Without a ``quantizer`` coordinates are kept verbatim (no quantization), so
    a round-trip through the pool is exact; **with** one (the ``compress`` path) each coordinate is
    snapped to an integer grid before interning, so near-coincident points merge and the pool holds
    small integers — the reader restores them through the document ``transform``.
    """

    def __init__(self, quantizer: "Any | None" = None) -> None:
        self._index: dict[tuple[float, ...], int] = {}
        self.vertices: list[list[float]] = []
        self._quantizer = quantizer

    def intern(self, coordinate: Any) -> int:
        """Return the index of ``coordinate`` in the pool, appending it on first sight."""
        if self._quantizer is None:
            key: tuple[float, ...] = tuple(float(c) for c in coordinate)
        else:
            key = self._quantizer(coordinate)
        existing = self._index.get(key)
        if existing is not None:
            return existing
        index = len(self.vertices)
        self._index[key] = index
        self.vertices.append(list(key))
        return index

    def reindex(self, element: Any) -> Any:
        """Replace every coordinate in a boundaries structure with its pool index.

        A *coordinate* is a flat sequence of numbers (the leaf the reader produced by dereferencing
        a vertex index); anything else is a nesting level and is recursed into. This is the exact
        inverse of :func:`pyramids_citymodels.readers.cityjson._dereference`.
        """
        if is_coordinate(element):
            return self.intern(element)
        return [self.reindex(child) for child in element]


def _encode_semantics(geometry: Geometry) -> dict[str, Any] | None:
    """Rebuild a geometry's CityJSON ``semantics`` block from its IR surfaces and values.

    Returns ``None`` when the geometry carries no semantics, so the ``"semantics"`` key is omitted
    rather than emitted empty — matching what the reader treats as "no semantics".
    """
    if not geometry.semantics and geometry.semantic_values is None:
        return None
    surfaces: list[dict[str, Any]] = []
    for surface in geometry.semantics:
        encoded: dict[str, Any] = {"type": surface.type}
        encoded.update(surface.attributes)
        if surface.parent is not None:
            encoded["parent"] = surface.parent
        if surface.children:
            encoded["children"] = list(surface.children)
        surfaces.append(encoded)
    return {"surfaces": surfaces, "values": geometry.semantic_values}


def _encode_geometry(geometry: Geometry, pool: _VertexPool) -> dict[str, Any]:
    """Encode one IR :class:`~pyramids_citymodels.ir.Geometry` as a CityJSON geometry object."""
    encoded: dict[str, Any] = {
        "type": geometry.type,
        "boundaries": pool.reindex(geometry.boundaries),
    }
    if geometry.lod is not None:
        encoded["lod"] = geometry.lod
    semantics = _encode_semantics(geometry)
    if semantics is not None:
        encoded["semantics"] = semantics
    if geometry.appearance:
        # The per-geometry material/texture maps were carried through untouched on read; re-emit
        # them verbatim so a renderer can still resolve them against the document appearance pool.
        for key in ("material", "texture"):
            if key in geometry.appearance:
                encoded[key] = geometry.appearance[key]
    return encoded


def _encode_city_object(city_object: CityObject, pool: _VertexPool) -> dict[str, Any]:
    """Encode one IR :class:`~pyramids_citymodels.ir.CityObject` as a CityJSON city object."""
    encoded: dict[str, Any] = {"type": city_object.type}
    if city_object.attributes:
        encoded["attributes"] = dict(city_object.attributes)
    if city_object.geometries:
        encoded["geometry"] = [
            _encode_geometry(geometry, pool) for geometry in city_object.geometries
        ]
    if city_object.parents:
        encoded["parents"] = list(city_object.parents)
    if city_object.children:
        encoded["children"] = list(city_object.children)
    return encoded


def _reference_system(model: CityModel) -> str | None:
    """The OGC CRS URL for the model's CRS, or ``None``.

    The reader recognizes a URN, an OGC URL and a plain ``EPSG:code`` alike; the OGC URL is the
    spelling CityJSON 2.0's ``metadata.referenceSystem`` uses, so emit that for an EPSG CRS. A
    non-EPSG identifier is passed through verbatim rather than forced into a malformed EPSG URL.
    """
    if not model.crs:
        return None
    if not model.crs.upper().startswith("EPSG:"):
        return model.crs
    code = model.crs.split(":")[-1]
    return f"https://www.opengis.net/def/crs/EPSG/0/{code}"


def _metadata(model: CityModel) -> dict[str, Any]:
    """The document ``metadata`` block: the model's metadata, with ``referenceSystem`` from the CRS.

    The model's own ``metadata`` is re-emitted verbatim (so a CityJSON→CityJSON round-trip is
    exact), and ``referenceSystem`` is filled in from :attr:`CityModel.crs` only when the metadata
    does not already carry one — which is how a CityGML-sourced model (CRS on the model, empty
    metadata) still writes its CRS out.
    """
    metadata = dict(model.metadata)
    if "referenceSystem" not in metadata:
        reference_system = _reference_system(model)
        if reference_system is not None:
            metadata["referenceSystem"] = reference_system
    return metadata


def _iter_coordinates(element: Any) -> "Any":
    """Yield every coordinate (as a float list) anywhere in a boundaries structure."""
    if is_coordinate(element):
        yield [float(component) for component in element]
    elif isinstance(element, list):
        for child in element:
            yield from _iter_coordinates(child)


def _compress_transform(model: CityModel, digits: int) -> dict[str, Any] | None:
    """A CityJSON ``transform`` (``scale`` + ``translate``) that quantizes coordinates to ``digits``.

    ``translate`` is the per-axis minimum (keeps the integer grid small and non-negative) and ``scale``
    is ``10 ** -digits`` on every axis. Returns ``None`` for a model with no coordinates.
    """
    minima: list[float] | None = None
    for city_object in model.objects.values():
        for geometry in city_object.geometries:
            for coordinate in _iter_coordinates(geometry.boundaries):
                if minima is None:
                    minima = list(coordinate)
                else:
                    for axis in range(min(len(minima), len(coordinate))):
                        minima[axis] = min(minima[axis], coordinate[axis])
    if minima is None:
        return None
    step = 10.0**-digits
    return {"scale": [step] * len(minima), "translate": minima}


def _quantizer(transform: dict[str, Any]) -> "Any":
    """A coordinate → integer-tuple function for a ``transform`` (the inverse of the reader's apply)."""
    translate = transform["translate"]
    scale = transform["scale"]

    def quantize(coordinate: Any) -> tuple[int, ...]:
        return tuple(
            round((float(component) - translate[axis]) / scale[axis])
            for axis, component in enumerate(coordinate)
        )

    return quantize


def _build_document(model: CityModel, compress: int | None = None) -> dict[str, Any]:
    """Build the full CityJSON document dict from the IR (no file I/O).

    When ``compress`` is an integer, coordinates are quantized to that many decimal digits and a
    ``transform`` is written, so the ``vertices`` are compact integers (the reader restores them).
    """
    transform = _compress_transform(model, compress) if compress is not None else None
    pool = _VertexPool(_quantizer(transform) if transform is not None else None)
    city_objects = {
        object_id: _encode_city_object(city_object, pool)
        for object_id, city_object in model.objects.items()
    }
    document: dict[str, Any] = {
        "type": "CityJSON",
        "version": model.version or _DEFAULT_VERSION,
        "CityObjects": city_objects,
        "vertices": pool.vertices,
    }
    if transform is not None:
        document["transform"] = transform
    metadata = _metadata(model)
    if metadata:
        document["metadata"] = metadata
    if model.appearances:
        document["appearance"] = dict(model.appearances)
    if model.extensions:
        document["extensions"] = dict(model.extensions)
    return document


def _write_or_return(document: dict[str, Any], path: Any) -> dict[str, Any] | None:
    """Return ``document`` when ``path`` is ``None``, else write it as JSON and return ``None``."""
    if path is None:
        return document
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
    return None


def write_cityjson(
    model: CityModel, path: Any = None, compress: int | None = None
) -> dict[str, Any] | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to CityJSON 2.0.

    Rebuilds the flat ``vertices`` pool and re-indexes every boundary from the IR's coordinate
    tuples (the reader's dereferencing, inverted), then re-emits semantics, appearances,
    extensions, metadata, CRS and version. The result round-trips:
    ``parse_cityjson(write_cityjson(model))`` reproduces ``model`` for a conformant source — the one
    exception is a versionless model (``version is None``), which is normalized to ``"2.0"`` on
    output (a CityJSON document must carry a version).

    With ``compress`` set, coordinates are quantized to that many decimal digits and written as
    compact **integer** ``vertices`` plus a ``transform`` the reader multiplies back — the standard
    CityJSON size reduction (lossy to the chosen precision; near-coincident points also merge).

    Args:
        model: The city model to serialize.
        path: A filesystem path to write the document to, or ``None`` to return the document dict.
        compress: Decimal digits of coordinate precision to keep; ``None`` (default) writes exact
            full-precision floats with no ``transform``.

    Returns:
        The CityJSON document as a ``dict`` when ``path`` is ``None``; otherwise ``None`` (the
        document is written to ``path``).

    Examples:
        - Round-trip a parsed model back out and read a re-indexed coordinate:
            ```python
            >>> from pyramids_citymodels.readers.cityjson import parse_cityjson
            >>> doc = {
            ...     "type": "CityJSON", "version": "2.0",
            ...     "CityObjects": {"b": {"type": "Building", "geometry": [
            ...         {"type": "MultiSurface", "lod": "2", "boundaries": [[[0, 1, 2]]],
            ...          "semantics": {"surfaces": [{"type": "RoofSurface"}], "values": [0]}}]}},
            ...     "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]}
            >>> out = write_cityjson(parse_cityjson(doc))
            >>> out["type"], out["CityObjects"]["b"]["geometry"][0]["semantics"]["surfaces"][0]["type"]
            ('CityJSON', 'RoofSurface')
            >>> parse_cityjson(out) == parse_cityjson(doc)
            True

            ```
        - The vertex pool is rebuilt from the IR and boundaries re-indexed into it:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building", geometries=[
            ...     Geometry(type="MultiSurface", lod="2",
            ...              boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]])])})
            >>> doc = write_cityjson(model)
            >>> len(doc["vertices"])
            3
            >>> doc["CityObjects"]["b"]["geometry"][0]["boundaries"]
            [[[0, 1, 2]]]

            ```
        - `compress` quantizes to integer vertices and writes a `transform` the reader restores:
            ```python
            >>> from pyramids_citymodels.ir import CityModel, CityObject, Geometry
            >>> surface = [[(10.5, 20.25, 1.0), (11.5, 20.25, 1.0), (11.5, 21.25, 1.0)]]
            >>> model = CityModel(objects={"b": CityObject(id="b", type="Building",
            ...     geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])])})
            >>> doc = write_cityjson(model, compress=2)
            >>> doc["vertices"][0], doc["vertices"][1]
            ([0, 0, 0], [100, 0, 0])
            >>> doc["transform"]["scale"]
            [0.01, 0.01, 0.01]

            ```
    """
    return _write_or_return(_build_document(model, compress), path)


def _feature_document(root_id: str, members: dict[str, CityObject]) -> dict[str, Any]:
    """Build one ``CityJSONFeature`` object (a root object plus its descendants) with its own pool."""
    pool = _VertexPool()
    city_objects = {
        object_id: _encode_city_object(city_object, pool)
        for object_id, city_object in members.items()
    }
    return {
        "type": "CityJSONFeature",
        "id": root_id,
        "CityObjects": city_objects,
        "vertices": pool.vertices,
    }


def _descendants(root_id: str, model: CityModel) -> dict[str, CityObject]:
    """The root object and every object reachable through its ``children`` links, by id."""
    members: dict[str, CityObject] = {}
    stack = [root_id]
    while stack:
        current = stack.pop()
        city_object = model.objects.get(current)
        if city_object is None or current in members:
            continue
        members[current] = city_object
        stack.extend(city_object.children)
    return members


def write_cityjsonseq(model: CityModel, path: Any = None) -> str | None:
    """Serialize a :class:`~pyramids_citymodels.ir.CityModel` to a CityJSON Text Sequence.

    A thin add-on over the CityJSON encoding: the first line is a CityJSON header (metadata,
    version, CRS; empty ``CityObjects``/``vertices``), and each following line is one
    ``CityJSONFeature`` — a top-level object (one with no parent) bundled with its descendants and
    its own local vertex pool. This is the streamable ``.city.jsonl`` form (one object per line).

    Args:
        model: The city model to serialize.
        path: A path to write the sequence to, or ``None`` to return it as a single string.

    Returns:
        The newline-delimited sequence as a ``str`` when ``path`` is ``None``; otherwise ``None``.

    Examples:
        - A one-building model yields a header line plus one feature line:
            ```python
            >>> from pyramids_citymodels.readers.cityjson import parse_cityjson
            >>> doc = {"type": "CityJSON", "version": "2.0",
            ...     "CityObjects": {"b": {"type": "Building", "geometry": [
            ...         {"type": "MultiSurface", "lod": "2", "boundaries": [[[0, 1, 2]]]}]}},
            ...     "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]}
            >>> text = write_cityjsonseq(parse_cityjson(doc))
            >>> lines = text.splitlines()
            >>> import json
            >>> json.loads(lines[0])["type"], json.loads(lines[1])["type"]
            ('CityJSON', 'CityJSONFeature')

            ```
        - A part is bundled into its root object's feature line, not given its own:
            ```python
            >>> import json
            >>> from pyramids_citymodels.ir import CityModel, CityObject
            >>> model = CityModel(objects={
            ...     "b": CityObject(id="b", type="Building", children=["p"]),
            ...     "p": CityObject(id="p", type="BuildingPart", parents=["b"])})
            >>> feature = json.loads(write_cityjsonseq(model).splitlines()[1])
            >>> sorted(feature["CityObjects"])
            ['b', 'p']

            ```
    """
    header: dict[str, Any] = {
        "type": "CityJSON",
        "version": model.version or _DEFAULT_VERSION,
        "CityObjects": {},
        "vertices": [],
    }
    metadata = _metadata(model)
    if metadata:
        header["metadata"] = metadata
    if model.appearances:
        # Carry the document appearance pool on the header line (CityJSONSeq permits it there), so
        # the per-geometry material/texture indices each CityJSONFeature still emits can resolve —
        # dropping it would leave those indices dangling.
        header["appearance"] = dict(model.appearances)
    if model.extensions:
        header["extensions"] = dict(model.extensions)

    lines = [json.dumps(header, ensure_ascii=False)]
    for object_id, city_object in model.objects.items():
        if city_object.parents:
            continue  # a non-root object is emitted inside its root's feature
        feature = _feature_document(object_id, _descendants(object_id, model))
        lines.append(json.dumps(feature, ensure_ascii=False))
    text = "\n".join(lines) + "\n"

    if path is None:
        return text
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return None
