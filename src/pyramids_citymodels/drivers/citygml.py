"""The CityGML vector driver (issue #11).

Registered into the ``pyramids.vector_drivers`` entry-point group, this driver is what
``FeatureCollection.read_file`` dispatches a CityGML read into. Like the CityJSON driver it
follows the class-attribute contract — claims are class attributes, :meth:`can_open` is a
classmethod — so discovery reads the class without importing the parser or shapely/geopandas;
those load only inside :meth:`read`.

``.gml`` / ``.xml`` are shared with plenty of non-CityGML data, so :meth:`can_open` sniffs for a
CityGML namespace or a ``CityModel`` element rather than trusting the suffix.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any


class CityGmlDriver:
    """Read CityGML LOD2 (``.gml`` / ``.xml`` / ``.citygml``) as a ``FeatureCollection``."""

    name = "citygml"
    long_name = "CityGML (3D city model, LOD2 semantic surfaces)"
    api_version = 1
    suffixes = (".gml", ".citygml", ".xml")
    schemes: tuple[str, ...] = ()
    capabilities = frozenset({"read"})
    #: Parsing uses only the standard-library ``xml.etree.ElementTree``, so there is no backend.
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return ``True`` only when ``source`` looks like CityGML (namespace / ``CityModel``)."""
        from pyramids_citymodels.readers.citygml import looks_like_citygml

        return looks_like_citygml(source)

    def read(self, source: Any, **kwargs: Any) -> Any:
        """Parse ``source``'s LOD2 buildings into a :class:`pyramids.feature.FeatureCollection`.

        Keyword Args:
            strategy: ``"footprint"`` or ``"semantic_surfaces"`` (default).
            lod: Level of detail to select; ``None`` takes the highest present.
            geometry_dimension: ``"xyz"`` (default) keeps Z, ``"xy"`` drops it.
            child_inherits_parent_attributes: Whether child rows carry parent attributes.
        """
        from pyramids_citymodels.flatten import to_feature_collection
        from pyramids_citymodels.readers.citygml import parse_citygml

        model = parse_citygml(source)
        return to_feature_collection(model, **kwargs)
