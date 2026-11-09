"""Tests for CityGML thematic-namespace writing (issue #49).

The writer emits each non-Building object in its correct CityGML 2.0 thematic namespace
(`frn:CityFurniture`, `brid:Bridge`, …) instead of an invalid `bldg:` element, and nests a part under
its module's `consistsOf…Part` wrapper, so these objects round-trip CityGML → IR → CityGML
conformantly (no `bldg:` misuse warning).
"""

import warnings

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry, SemanticSurface


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


def test_bridge_boundary_surface_and_geometry_are_in_the_bridge_module() -> None:
    # M1: a bridge's boundary surface, its boundedBy wrapper and its lod2MultiSurface geometry are all
    # emitted in the bridge module (brid:), not hardcoded bldg:.
    model = CityModel(
        objects={
            "br": CityObject(
                id="br",
                type="Bridge",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]]
                        ],
                        semantics=[SemanticSurface(type="WallSurface")],
                        semantic_values=[0],
                    )
                ],
            )
        }
    )
    xml = model.to_citygml()
    assert "<brid:boundedBy>" in xml
    assert "<brid:WallSurface" in xml
    assert "<brid:lod2MultiSurface>" in xml
    assert "bldg:" not in xml  # nothing leaks into the building namespace


def test_mixed_geometry_emits_plain_before_bounded_by() -> None:
    # R2-M1: in CityGML 2.0 the plain lod{n}MultiSurface must precede boundedBy in the content
    # sequence; a geometry mixing a semantic and a plain surface must emit them in that order.
    roof = [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
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
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]],
                            roof,
                        ],
                        semantics=[SemanticSurface(type="RoofSurface")],
                        semantic_values=[None, 0],
                    )
                ],
            )
        }
    )
    xml = model.to_citygml()
    # (use the bldg: prefix: a bare "boundedBy>" also matches the root gml:boundedBy envelope)
    assert xml.index("<bldg:lod2MultiSurface>") < xml.index("<bldg:boundedBy>")


def test_two_same_lod_geometries_aggregate_to_one_property() -> None:
    # R2-L1: two LOD-2 geometries on one object share a single lod2MultiSurface (maxOccurs=1).
    roof = [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
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
                    ),
                    Geometry(type="MultiSurface", lod="2", boundaries=[roof]),
                ],
            )
        }
    )
    assert (
        model.to_citygml().count("<bldg:lod2MultiSurface>") == 1
    )  # one aggregated property


def test_demoted_semantic_surface_warns() -> None:
    # R2-N1: a typed surface a module cannot carry as a boundary surface is written as plain geometry
    # (its semantic type dropped) and that lossy demotion is warned, not silent.
    model = CityModel(
        objects={
            "l": CityObject(
                id="l",
                type="LandUse",  # no boundedBy slot
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                        semantics=[SemanticSurface(type="RoofSurface")],
                        semantic_values=[0],
                    )
                ],
            )
        }
    )
    with pytest.warns(UserWarning, match="semantic boundary surface"):
        model.to_citygml()


def test_plain_surfaces_aggregate_into_one_geometry_property() -> None:
    # The non-semantic surfaces of a geometry share ONE lodNMultiSurface property (it is maxOccurs=1);
    # emitting one property per surface would be schema-invalid.
    two = [
        [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]],
        [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]],
    ]
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[Geometry(type="MultiSurface", lod="2", boundaries=two)],
            )
        }
    )
    assert (
        model.to_citygml().count("lod2MultiSurface>") == 2
    )  # one open + one close tag


def test_solid_on_a_module_without_a_solid_slot_falls_back_to_plain_geometry() -> None:
    # CityFurniture has no lodNSolid slot, so a Solid geometry is written as its plain lodNGeometry
    # (not a bldg:lodNSolid) — the _append_solid-returns-False path.
    wall = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]]
    model = CityModel(
        objects={
            "f": CityObject(
                id="f",
                type="CityFurniture",
                geometries=[Geometry(type="Solid", lod="2", boundaries=[[wall]])],
            )
        }
    )
    xml = model.to_citygml()
    assert "<frn:lod2Geometry>" in xml
    assert "Solid>" not in xml  # no gml:Solid — the solid was flattened to surfaces


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


def test_known_representable_city_objects_emit_no_warning() -> None:
    from pyramids_citymodels.readers.citygml import parse_citygml

    model = CityModel(
        objects={
            "f": _obj("f", "CityFurniture"),
            "l": _obj("l", "LandUse"),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any writer warning would raise
        xml = model.to_citygml()
    # and they round-trip back to their own types
    back = parse_citygml(xml)
    assert {o.type for o in back.objects.values()} == {"CityFurniture", "LandUse"}


@pytest.mark.parametrize(
    "object_type", ["WaterBody", "ReliefFeature", "CityObjectGroup"]
)
def test_module_without_a_surface_slot_warns_but_still_namespaces_the_feature(
    object_type: str,
) -> None:
    # Relief, groups and a water *surface* have no slot in their module for the IR's generic surface
    # geometry, so the geometry falls back to a bldg: property with a warning — but the feature element
    # itself is still emitted in the object's own thematic namespace.
    model = CityModel(objects={"x": _obj("x", object_type)})
    with pytest.warns(UserWarning, match="no geometry slot"):
        xml = model.to_citygml()
    prefix = {"WaterBody": "wtr", "ReliefFeature": "dem", "CityObjectGroup": "grp"}[
        object_type
    ]
    assert f"<{prefix}:{object_type} " in xml  # feature namespaced correctly
    assert f"<bldg:{object_type}" not in xml


def test_unknown_type_falls_back_to_bldg_with_warning() -> None:
    model = CityModel(objects={"x": _obj("x", "Nonsense")})
    with pytest.warns(UserWarning, match="no known thematic namespace"):
        xml = model.to_citygml()
    assert "<bldg:Nonsense" in xml
