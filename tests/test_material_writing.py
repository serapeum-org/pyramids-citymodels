"""Tests for CityGML X3DMaterial writing (issue #43).

The CityGML writer emits `app:Appearance` / `app:X3DMaterial` for material-painted surfaces, targeting
each surface's `gml:Polygon` by a `gml:id` it assigns, so a material on the IR (read from CityJSON or
CityGML) survives `to_citygml()` and round-trips CityJSON → CityGML → CityJSON.
"""

import warnings

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.citygml import parse_citygml


def _material_model(
    values: object = None, *, geom_type: str = "MultiSurface"
) -> CityModel:
    """A one-building model whose surface(s) are painted red under theme ``t`` (index 0)."""
    if geom_type == "Solid":
        boundaries: object = [[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]]]
        default_values: object = [[0]]
    else:
        boundaries = [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]]
        default_values = [0]
    return CityModel(
        appearances={"materials": [{"name": "red", "diffuseColor": [1.0, 0.0, 0.0]}]},
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type=geom_type,
                        lod="2",
                        boundaries=boundaries,
                        appearance={
                            "material": {"t": {"values": values or default_values}}
                        },
                    )
                ],
            )
        },
    )


def test_writer_emits_x3dmaterial_targeting_a_polygon() -> None:
    xml = _material_model().to_citygml()
    assert "X3DMaterial" in xml
    assert "diffuseColor" in xml
    assert 'gml:id="poly_' in xml  # the polygon got an id
    assert "#poly_" in xml  # and the material targets it


def test_material_colour_round_trips_cityjson_to_citygml_to_cityjson() -> None:
    model = _material_model()
    back = parse_citygml(model.to_citygml())
    assert back.appearances["materials"][0]["diffuseColor"] == [1.0, 0.0, 0.0]
    values = back.objects["b"].geometries[0].appearance["material"]["t"]["values"]
    assert values == [0]  # the surface is still painted by material 0


def test_unpainted_surface_emits_no_appearance() -> None:
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                    )
                ],
            )
        }
    )
    xml = model.to_citygml()
    assert "X3DMaterial" not in xml
    assert "Appearance" not in xml
    assert parse_citygml(xml).objects["b"].geometries[0].appearance is None


def test_only_painted_surface_is_targeted_in_a_multi_surface() -> None:
    # Two surfaces, only the first painted (index 0, then null): exactly one target.
    model = _material_model(values=[0, None])
    model.objects["b"].geometries[0].boundaries = [
        [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]],
        [[(0.0, 0.0, 9.0), (1.0, 0.0, 9.0), (1.0, 1.0, 9.0)]],
    ]
    back = parse_citygml(model.to_citygml())
    values = back.objects["b"].geometries[0].appearance["material"]["t"]["values"]
    assert values == [0, None]


def test_material_on_a_solid_round_trips() -> None:
    model = _material_model(geom_type="Solid")
    back = parse_citygml(model.to_citygml()).objects["b"].geometries[0]
    assert back.type == "Solid"
    assert back.appearance["material"]["t"]["values"] == [[0]]  # shell-nested


def test_multi_theme_materials_each_get_an_appearance() -> None:
    model = CityModel(
        appearances={
            "materials": [
                {"name": "red", "diffuseColor": [1.0, 0.0, 0.0]},
                {"name": "blue", "diffuseColor": [0.0, 0.0, 1.0]},
            ]
        },
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                        appearance={
                            "material": {
                                "day": {"values": [0]},
                                "night": {"values": [1]},
                            }
                        },
                    )
                ],
            )
        },
    )
    back = parse_citygml(model.to_citygml()).objects["b"].geometries[0]
    assert back.appearance["material"]["day"]["values"] == [0]
    assert back.appearance["material"]["night"]["values"] == [1]


def test_scalars_and_issmooth_round_trip() -> None:
    model = _material_model()
    model.appearances["materials"][0].update(
        {"shininess": 0.5, "transparency": 0.25, "isSmooth": True}
    )
    material = parse_citygml(model.to_citygml()).appearances["materials"][0]
    assert material["shininess"] == 0.5
    assert material["transparency"] == 0.25
    assert material["isSmooth"] is True


def test_writing_a_material_emits_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _material_model().to_citygml()


def test_material_without_a_name_still_emits_and_round_trips() -> None:
    # A material lacking a "name" gets no gml:id on its X3DMaterial, but still paints and round-trips.
    model = _material_model()
    model.appearances["materials"][0] = {"diffuseColor": [0.0, 1.0, 0.0]}
    xml = model.to_citygml()
    assert "X3DMaterial" in xml
    back = parse_citygml(xml)
    assert back.appearances["materials"][0]["diffuseColor"] == [0.0, 1.0, 0.0]
    assert back.objects["b"].geometries[0].appearance["material"]["t"]["values"] == [0]
