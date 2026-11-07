"""Tests for CityGML thematic-namespace writing (issue #49).

The writer emits each non-Building object in its correct CityGML 2.0 thematic namespace
(`frn:CityFurniture`, `brid:Bridge`, …) instead of an invalid `bldg:` element, and nests a part under
its module's `consistsOf…Part` wrapper, so these objects round-trip CityGML → IR → CityGML
conformantly (no `bldg:` misuse warning).
"""

import warnings

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry


def _obj(object_id: str, object_type: str, **kw: object) -> CityObject:
    surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    return CityObject(
        id=object_id,
        type=object_type,
        geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[surface])],
        **kw,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "object_type, namespace, element",
    [
        ("CityFurniture", "citygml/cityfurniture/2.0", "frn:CityFurniture"),
        ("Bridge", "citygml/bridge/2.0", "brid:Bridge"),
        ("Tunnel", "citygml/tunnel/2.0", "tun:Tunnel"),
        (
            "SolitaryVegetationObject",
            "citygml/vegetation/2.0",
            "veg:SolitaryVegetationObject",
        ),
        ("WaterBody", "citygml/waterbody/2.0", "wtr:WaterBody"),
        ("LandUse", "citygml/landuse/2.0", "luse:LandUse"),
        ("ReliefFeature", "citygml/relief/2.0", "dem:ReliefFeature"),
        ("Road", "citygml/transportation/2.0", "tran:Road"),
        ("CityObjectGroup", "citygml/cityobjectgroup/2.0", "grp:CityObjectGroup"),
    ],
)
def test_object_type_emitted_in_its_thematic_namespace(
    object_type: str, namespace: str, element: str
) -> None:
    xml = CityModel(objects={"x": _obj("x", object_type)}).to_citygml()
    assert namespace in xml
    assert f"<{element} " in xml
    assert f"<bldg:{object_type}" not in xml  # not the invalid bldg: fallback


def test_part_nests_under_its_module_consists_of_wrapper() -> None:
    model = CityModel(
        objects={
            "br": _obj("br", "Bridge", children=["bp"]),
            "bp": _obj("bp", "BridgePart", parents=["br"]),
        }
    )
    xml = model.to_citygml()
    assert "<brid:consistsOfBridgePart>" in xml
    assert "<brid:BridgePart " in xml


def test_building_still_uses_bldg_namespace() -> None:
    xml = CityModel(objects={"b": _obj("b", "Building")}).to_citygml()
    assert "<bldg:Building " in xml


def test_known_city_objects_emit_no_unreadable_warning() -> None:
    from pyramids_citymodels.readers.citygml import parse_citygml

    model = CityModel(
        objects={
            "f": _obj("f", "CityFurniture"),
            "w": _obj("w", "WaterBody"),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error"
        )  # any "no known thematic namespace" warning would raise
        xml = model.to_citygml()
    # and they round-trip back to their own types
    back = parse_citygml(xml)
    assert {o.type for o in back.objects.values()} == {"CityFurniture", "WaterBody"}


def test_unknown_type_falls_back_to_bldg_with_warning() -> None:
    model = CityModel(objects={"x": _obj("x", "Nonsense")})
    with pytest.warns(UserWarning, match="no known thematic namespace"):
        xml = model.to_citygml()
    assert "<bldg:Nonsense" in xml
