"""Typed errors raised by the city-model drivers.

Kept in one module so a caller can catch the package's own failures precisely
(``except pyramids_citymodels.errors.UnsupportedGeometryError``) rather than a bare ``ValueError``.
Each subclasses a built-in so existing ``except ValueError`` handlers keep working.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

__all__ = ["CityModelError", "UnsupportedGeometryError"]


class CityModelError(ValueError):
    """Base class for errors raised while reading a city model.

    Catch this to handle any failure this package raises with one ``except``; because it subclasses
    the built-in ``ValueError``, code that already catches ``ValueError`` keeps working unchanged.

    Examples:
        - The common base catches a more specific reader error and exposes its message:
            ```python
            >>> try:
            ...     raise UnsupportedGeometryError("city object 'x' has geometry type 'MultiPoint'")
            ... except CityModelError as error:
            ...     print(str(error))
            city object 'x' has geometry type 'MultiPoint'

            ```
    """


class UnsupportedGeometryError(CityModelError):
    """A geometry whose type cannot be represented as areal features.

    Raised when the flattener meets a geometry type it does not map to polygons — a point or line
    aggregate (``MultiPoint``, ``MultiLineString``) or a template reference (``GeometryInstance``).
    Rejecting these explicitly, with the offending object id and type, is deliberate: silently
    dropping them would hand back an empty row set with no indication why (report §5, risk #8).

    Examples:
        - Flattening a line/point geometry raises it, and the message names the offending type:
            ```python
            >>> from pyramids_citymodels.flatten import to_feature_collection
            >>> from pyramids_citymodels.readers.cityjson import parse_cityjson
            >>> doc = {
            ...     "type": "CityJSON", "version": "2.0",
            ...     "CityObjects": {"p": {"type": "Building", "geometry": [
            ...         {"type": "MultiLineString", "lod": "2", "boundaries": [[0, 1]]}]}},
            ...     "vertices": [[0, 0, 0], [1, 0, 0]]}
            >>> try:
            ...     to_feature_collection(parse_cityjson(doc))
            ... except UnsupportedGeometryError as error:
            ...     print("MultiLineString" in str(error))
            True

            ```
        - It is a ``ValueError``, so a broad handler still catches it:
            ```python
            >>> try:
            ...     raise UnsupportedGeometryError("MultiPoint is not areal")
            ... except ValueError as error:
            ...     print(str(error))
            MultiPoint is not areal

            ```
    """
