"""Shared model builders for the Draco 3D-Tiles write tests (glTF and b3dm content).

The glTF-content and b3dm Draco test modules build the same small :class:`CityModel` — a diagonal row
of buildings, each a stack of square surfaces — so they share one definition here instead of cloning it.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry


def square(
    x: float, y: float, z: float = 0.0
) -> list[list[tuple[float, float, float]]]:
    """A closed 4-unit square surface at ``(x, y)`` on the ``z`` plane."""
    return [[(x, y, z), (x + 4, y, z), (x + 4, y + 4, z), (x, y + 4, z)]]


def model(count: int = 2, surfaces: int = 1) -> CityModel:
    """``count`` buildings on a diagonal, each a stack of ``surfaces`` squares with distinctive attrs."""
    return CityModel(
        objects={
            f"o{i}": CityObject(
                id=f"o{i}",
                type="Building" if i % 2 == 0 else "Bridge",
                attributes={"height": 3.5 + i, "floors": i},
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            square(i * 50.0, i * 50.0, 3.0 * s) for s in range(surfaces)
                        ],
                    )
                ],
            )
            for i in range(count)
        }
    )
