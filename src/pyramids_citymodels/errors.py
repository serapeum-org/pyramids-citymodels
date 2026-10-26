"""Typed errors raised by the city-model drivers.

Kept in one module so a caller can catch the package's own failures precisely
(``except pyramids_citymodels.errors.UnsupportedGeometryError``) rather than a bare ``ValueError``.
Each subclasses a built-in so existing ``except ValueError`` handlers keep working.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

__all__ = ["CityModelError", "UnsupportedGeometryError"]


class CityModelError(ValueError):
    """Base class for errors raised while reading a city model."""


class UnsupportedGeometryError(CityModelError):
    """A geometry whose type cannot be represented as areal features.

    Raised when the flattener meets a geometry type it does not map to polygons — a point or line
    aggregate (``MultiPoint``, ``MultiLineString``) or a template reference (``GeometryInstance``).
    Rejecting these explicitly, with the offending object id and type, is deliberate: silently
    dropping them would hand back an empty row set with no indication why (report §5, risk #8).
    """
