"""The 3D Tiles vector driver (issue #16).

Registered into the ``pyramids.vector_drivers`` entry-point group alongside the CityJSON and
CityGML drivers, this is what ``FeatureCollection.read_file`` dispatches a 3D Tiles read into. It
follows the same class-attribute contract — claims are class attributes, :meth:`can_open` is a
classmethod — so discovery reads the class without importing the parser or numpy/shapely; those
load only inside :meth:`read`.

A ``tileset.json`` shares the ``.json`` suffix with CityJSON and GeoJSON, so :meth:`can_open`
**content-sniffs** the tileset structure (an ``asset`` object plus a ``root``/``geometricError``)
and the ``b3dm`` / glTF binary magic rather than trusting the suffix. The driver surfaces building
**footprints and batch-table attributes**; see :mod:`pyramids_citymodels.readers.threedtiles` for
the deliberate scope (footprints not meshes, no ECEF→geographic projection, uncompressed glTF only,
``pnts`` routed to the pointclouds package).
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from typing import Any


class ThreeDTilesDriver:
    """Read Cesium 3D Tiles (``tileset.json`` / ``.b3dm``) footprints as a ``FeatureCollection``."""

    name = "3dtiles"
    long_name = "3D Tiles (Cesium — building footprints + batch-table attributes)"
    api_version = 1
    suffixes = (".json", ".b3dm", ".glb")
    schemes: tuple[str, ...] = ()
    capabilities = frozenset({"read"})
    #: The tileset/b3dm/glb containers decode with the standard library and numpy (already a
    #: dependency), so there is no optional backend. A compressed-mesh backend (Draco/meshopt) is a
    #: documented follow-up that would attach its own ``extra`` here (#12).
    extra: str | None = None

    @classmethod
    def can_open(cls, source: Any) -> bool:
        """Return ``True`` only when ``source`` is a 3D Tiles tileset or a ``b3dm``/glTF tile.

        Content-sniffs the tileset structure (and the binary magic) rather than trusting the
        ``.json`` suffix it shares with CityJSON/GeoJSON, so a CityJSON file is declined. Never
        raises.

        Examples:
            - A tileset is claimed; a CityJSON document sharing the ``.json`` suffix is declined:
                ```python
                >>> ThreeDTilesDriver.can_open(
                ...     '{"asset": {"version": "1.0"}, "geometricError": 0, "root": {}}'
                ... )
                True
                >>> ThreeDTilesDriver.can_open(
                ...     '{"type": "CityJSON", "CityObjects": {}, "vertices": []}'
                ... )
                False

                ```
        """
        from pyramids_citymodels.readers.threedtiles import looks_like_3dtiles

        return looks_like_3dtiles(source)

    def read(self, source: Any, **kwargs: Any) -> Any:
        """Parse ``source`` and flatten its footprints into a ``FeatureCollection``.

        Keyword Args:
            strategy: ``"footprint"`` or ``"semantic_surfaces"`` (default). 3D Tiles carries no
                semantic surfaces, so each feature's footprint becomes one row either way.
            lod: Level of detail to select; ``None`` takes the highest present (3D Tiles has none).
            geometry_dimension: ``"xyz"`` (default) keeps Z, ``"xy"`` drops it.
            child_inherits_parent_attributes: Whether child rows carry parent attributes.

        Examples:
            - A ``tileset.json`` or ``.b3dm`` path yields one footprint row per batched feature;
              an in-memory tileset (no base directory to resolve content against) reads as an
              empty frame carrying only the geometry column:
                ```python
                >>> from pyramids_citymodels.drivers.threedtiles import ThreeDTilesDriver
                >>> frame = ThreeDTilesDriver().read(
                ...     '{"asset": {"version": "1.0"}, "geometricError": 0, "root": {}}'
                ... )
                >>> len(frame)
                0
                >>> list(frame.columns)
                ['geometry']

                ```
        """
        from pyramids_citymodels.flatten import to_feature_collection
        from pyramids_citymodels.readers.threedtiles import parse_3dtiles

        model = parse_3dtiles(source)
        return to_feature_collection(model, **kwargs)
