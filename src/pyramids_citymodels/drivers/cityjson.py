"""The CityJSON vector driver (issue #11).

Registered into the ``pyramids.vector_drivers`` entry-point group, this driver is what
``FeatureCollection.read_file`` dispatches a CityJSON read into. It follows the class-attribute
contract pyramids' driver registry reads without constructing the driver: the claims
(``name``/``long_name``/``suffixes``/…) are class attributes and :meth:`can_open` is a
classmethod, so discovery never imports the parser or shapely/geopandas — those load only inside
:meth:`read`, when a read is actually dispatched here.

The ``.json`` suffix is shared with GeoJSON, so a suffix alone cannot be trusted: :meth:`can_open`
**content-sniffs** ``"type": "CityJSON"`` to claim a file, which cleanly separates CityJSON from a
GeoJSON ``FeatureCollection`` (decision #3 in the ecosystem report).
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any


class CityJsonDriver:
    """Read CityJSON (``.city.json`` / ``.json`` / ``.cityjson``) as a ``FeatureCollection``."""

    name = "cityjson"
    long_name = "CityJSON (3D city model)"
    api_version = 1
    suffixes = (".city.json", ".cityjson", ".json")
    schemes: tuple[str, ...] = ()
    capabilities = frozenset({"read"})
    #: Parsing uses only the standard-library ``json`` module, so there is no optional backend.
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return ``True`` only when ``source`` is a CityJSON document.

        Content-sniffs ``"type": "CityJSON"`` rather than trusting the ``.json`` suffix, so a
        GeoJSON file sharing that suffix is declined. Never raises.
        """
        from pyramids_citymodels.readers.cityjson import looks_like_cityjson

        return looks_like_cityjson(source)

    def read(self, source: Any, **kwargs: Any) -> Any:
        """Parse ``source`` and flatten it into a :class:`pyramids.feature.FeatureCollection`.

        Keyword Args:
            strategy: ``"footprint"`` or ``"semantic_surfaces"`` (default).
            lod: Level of detail to select; ``None`` takes the highest present.
            geometry_dimension: ``"xyz"`` (default) keeps Z, ``"xy"`` drops it.
            child_inherits_parent_attributes: Whether child rows carry parent attributes.
        """
        from pyramids_citymodels.flatten import to_feature_collection
        from pyramids_citymodels.readers.cityjson import parse_cityjson

        model = parse_cityjson(source)
        return to_feature_collection(model, **kwargs)
