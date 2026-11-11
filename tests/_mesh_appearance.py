"""Shared model builders for the mesh-appearance writer tests (OBJ / COLLADA / glTF).

These factories build the small textured/materialled :class:`CityModel` s the mesh-export tests reuse,
so the OBJ and COLLADA test modules share one definition instead of cloning it.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry


def triangle(x: float = 0.0) -> list[list[tuple[float, float, float]]]:
    """A single triangular surface offset along x."""
    return [[(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 1.0, 0.0)]]


def material_model(
    diffuse: list[float] | None = None, image: str | None = None
) -> CityModel:
    """A one-surface model referencing material 0 (optionally textured by texture 0)."""
    appearance: dict = {"material": {"": {"values": [0]}}}
    appearances: dict = {
        "materials": [{"name": "m", **({"diffuseColor": diffuse} if diffuse else {})}]
    }
    if image is not None:
        appearance["texture"] = {"": {"values": [[[0, 0, 1, 2]]]}}
        appearances["textures"] = [{"type": "PNG", "image": image}]
        appearances["vertices-texture"] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    geometry = Geometry(
        type="MultiSurface", lod="2", boundaries=[triangle()], appearance=appearance
    )
    return CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
        appearances=appearances,
    )


def two_material_model() -> CityModel:
    """A two-surface model: surface 0 red, surface 1 blue (distinct materials, no texture)."""
    geometry = Geometry(
        type="MultiSurface",
        lod="2",
        boundaries=[triangle(0.0), triangle(5.0)],
        appearance={"material": {"": {"values": [0, 1]}}},
    )
    return CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
        appearances={
            "materials": [
                {"name": "red", "diffuseColor": [1.0, 0.0, 0.0]},
                {"name": "blue", "diffuseColor": [0.0, 0.0, 1.0]},
            ]
        },
    )
