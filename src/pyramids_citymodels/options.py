"""Read options: how the 3D, hierarchical city model is projected onto flat features.

A GeoDataFrame is 2D-centric, flat and one-geometry-per-row; a city model is 3D, hierarchical
and graph-linked. There is no single right projection, so the projection is exposed as options
rather than hardcoded — the lesson the QGIS CityJSON Loader and 3DCityDB both teach (issues #9
and #10):

- :class:`LoadStrategy` — *what a row is*: one row per city object (its footprint) or one row
  per semantic surface (wall/roof/ground as individual features).
- :class:`GeometryDimension` — *how Z is carried*: dropped for a 2D footprint, or kept for 2.5D
  surfaces. A ``footprint`` load is always 2D regardless, since a footprint is a plan projection.
- ``lod`` — which level of detail to select when a city object carries several.
- ``child_inherits_parent_attributes`` — whether a ``BuildingPart`` row also carries its
  ``Building``'s attributes.

:class:`ReadOptions` bundles these, and :meth:`ReadOptions.coerce` accepts either the enums or
their plain-string values so a driver can pass user kwargs straight through.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "DEFAULT_LOD",
    "GeometryDimension",
    "LoadStrategy",
    "ReadOptions",
]

#: The level of detail selected when the caller does not ask for one and an object offers
#: several. LOD2 is the semantic-building level both issues (#1383/#1384) target.
DEFAULT_LOD = "2"


class LoadStrategy(StrEnum):
    """What a row in the output represents."""

    #: One row per city object, geometry = its 2D footprint (its ground surfaces).
    FOOTPRINT = "footprint"
    #: One row per semantic surface — wall/roof/ground each become an individual feature.
    SEMANTIC_SURFACES = "semantic_surfaces"


class GeometryDimension(StrEnum):
    """How Z is carried in the output geometry."""

    #: Drop Z — 2D geometry.
    XY = "xy"
    #: Keep Z — 2.5D geometry (coordinates retain their height).
    XYZ = "xyz"


@dataclass(frozen=True)
class ReadOptions:
    """The resolved set of read options the flattener consumes.

    Attributes:
        strategy: The :class:`LoadStrategy`.
        lod: The level of detail to select, or ``None`` to take the highest each object offers.
        geometry_dimension: The :class:`GeometryDimension` (ignored for a footprint load, which
            is always 2D).
        child_inherits_parent_attributes: When ``True``, a child object's rows also carry its
            parent's attributes (the parent's values do not overwrite the child's own).
    """

    strategy: LoadStrategy = LoadStrategy.SEMANTIC_SURFACES
    lod: str | None = None
    geometry_dimension: GeometryDimension = GeometryDimension.XYZ
    child_inherits_parent_attributes: bool = False

    @classmethod
    def coerce(
        cls,
        strategy: "str | LoadStrategy" = LoadStrategy.SEMANTIC_SURFACES,
        lod: Any = None,
        geometry_dimension: "str | GeometryDimension" = GeometryDimension.XYZ,
        child_inherits_parent_attributes: bool = False,
    ) -> "ReadOptions":
        """Build a :class:`ReadOptions`, accepting enums or their plain-string values.

        Raises:
            ValueError: If ``strategy`` or ``geometry_dimension`` is not a recognized value.

        Examples:
            >>> ReadOptions.coerce(strategy="footprint").strategy is LoadStrategy.FOOTPRINT
            True
            >>> ReadOptions.coerce(geometry_dimension="xy").geometry_dimension
            <GeometryDimension.XY: 'xy'>
            >>> ReadOptions.coerce(lod=2).lod
            '2'
        """
        try:
            resolved_strategy = LoadStrategy(strategy)
        except ValueError as error:
            valid = ", ".join(repr(member.value) for member in LoadStrategy)
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of {valid}"
            ) from error
        try:
            resolved_dimension = GeometryDimension(geometry_dimension)
        except ValueError as error:
            valid = ", ".join(repr(member.value) for member in GeometryDimension)
            raise ValueError(
                f"unknown geometry_dimension {geometry_dimension!r}; expected one of {valid}"
            ) from error
        return cls(
            strategy=resolved_strategy,
            lod=None if lod is None else str(lod),
            geometry_dimension=resolved_dimension,
            child_inherits_parent_attributes=bool(child_inherits_parent_attributes),
        )
