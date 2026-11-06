"""Tests for CityGML reader breadth — city objects beyond LOD2 buildings (reader-breadth feature).

The reader recognizes the common top-level city objects (bridges, tunnels, furniture, vegetation,
water, land use, relief, transportation, …) and reads geometry both from boundary surfaces (semantic)
and directly from ``lodX`` containers (furniture/vegetation/etc.), inferring the LOD from the
container.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.drivers.citygml import CityGmlDriver
from pyramids_citymodels.readers.citygml import parse_citygml

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:frn="http://www.opengis.net/citygml/cityfurniture/2.0" '
    'xmlns:veg="http://www.opengis.net/citygml/vegetation/2.0" '
    'xmlns:brid="http://www.opengis.net/citygml/bridge/2.0" '
    'xmlns:tun="http://www.opengis.net/citygml/tunnel/2.0" '
    'xmlns:wtr="http://www.opengis.net/citygml/waterbody/2.0" '
    'xmlns:gen="http://www.opengis.net/citygml/generics/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)
_RING = '<gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension="3">{}</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon>'
_TRI = _RING.format("0 0 0 1 0 0 1 1 0 0 0 0")


def _direct(element: str, container: str, gml_id: str) -> str:
    """A city object ``element`` carrying one triangle directly in a ``lodX`` ``container``."""
    return (
        f"<core:CityModel {_NS}><core:cityObjectMember>"
        f'<{element} gml:id="{gml_id}">'
        f"<{container}><gml:MultiSurface><gml:surfaceMember>{_TRI}"
        f"</gml:surfaceMember></gml:MultiSurface></{container}>"
        f"</{element}></core:cityObjectMember></core:CityModel>"
    )


def _boundary(element: str, surface: str, gml_id: str) -> str:
    """A city object ``element`` carrying one ``surface`` boundary surface (a triangle)."""
    return (
        f"<core:CityModel {_NS}><core:cityObjectMember>"
        f'<{element} gml:id="{gml_id}">'
        f"<bldg:boundedBy><bldg:{surface}><bldg:lod2MultiSurface><gml:MultiSurface>"
        f"<gml:surfaceMember>{_TRI}</gml:surfaceMember>"
        f"</gml:MultiSurface></bldg:lod2MultiSurface></bldg:{surface}></bldg:boundedBy>"
        f"</{element}></core:cityObjectMember></core:CityModel>"
    )


@pytest.mark.parametrize(
    "element,container,gml_id,expected_lod",
    [
        ("frn:CityFurniture", "frn:lod1Geometry", "F1", "1"),
        ("veg:SolitaryVegetationObject", "veg:lod2Geometry", "V1", "2"),
        ("veg:PlantCover", "veg:lod1MultiSurface", "P1", "1"),
        ("wtr:WaterBody", "wtr:lod1MultiSurface", "W1", "1"),
        ("gen:GenericCityObject", "gen:lod2Geometry", "G1", "2"),
    ],
)
def test_direct_geometry_object_types_are_read_with_inferred_lod(
    element: str, container: str, gml_id: str, expected_lod: str
) -> None:
    model = parse_citygml(_direct(element, container, gml_id))
    assert set(model.objects) == {gml_id}
    geometry = model.objects[gml_id].geometries[0]
    assert len(geometry.boundaries) == 1  # the one triangle
    assert geometry.lod == expected_lod  # read from the lodX container
    assert geometry.semantics == []  # direct geometry has no semantic surface
    assert geometry.semantic_values is None


def test_direct_geometry_object_type_is_the_feature_type() -> None:
    model = parse_citygml(_direct("frn:CityFurniture", "frn:lod1Geometry", "F1"))
    assert model.objects["F1"].type == "CityFurniture"


@pytest.mark.parametrize(
    "element,gml_id",
    [("brid:Bridge", "BR1"), ("tun:Tunnel", "T1")],
)
def test_bridge_and_tunnel_boundary_surfaces_are_read_with_semantics(
    element: str, gml_id: str
) -> None:
    model = parse_citygml(_boundary(element, "RoofSurface", gml_id))
    geometry = model.objects[gml_id].geometries[0]
    assert geometry.lod == "2"  # the semantic-surface level
    assert [surface.type for surface in geometry.semantics] == ["RoofSurface"]


def test_water_surface_is_tagged_as_a_semantic_boundary_surface() -> None:
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember>"
        '<wtr:WaterBody gml:id="W1"><wtr:boundedBy><wtr:WaterSurface>'
        f"<wtr:lod2Surface><gml:MultiSurface><gml:surfaceMember>{_TRI}"
        "</gml:surfaceMember></gml:MultiSurface></wtr:lod2Surface>"
        "</wtr:WaterSurface></wtr:boundedBy></wtr:WaterBody>"
        "</core:cityObjectMember></core:CityModel>"
    )
    geometry = parse_citygml(xml).objects["W1"].geometries[0]
    assert [surface.type for surface in geometry.semantics] == ["WaterSurface"]


def test_nested_part_of_a_non_building_links_to_its_parent() -> None:
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember>"
        '<brid:Bridge gml:id="BR1"><brid:consistsOfBridgePart>'
        '<brid:BridgePart gml:id="BP1">'
        f"<brid:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>{_TRI}"
        "</gml:surfaceMember></gml:MultiSurface></brid:lod2MultiSurface>"
        "</brid:BridgePart></brid:consistsOfBridgePart></brid:Bridge>"
        "</core:cityObjectMember></core:CityModel>"
    )
    model = parse_citygml(xml)
    assert set(model.objects) == {"BR1", "BP1"}
    assert model.objects["BP1"].parents == ["BR1"]
    assert "BP1" in model.objects["BR1"].children


def test_boundary_surface_lod_follows_its_container() -> None:
    # A boundary surface under a lod3MultiSurface is LOD 3, not hardcoded LOD 2 (review R2-L3).
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod3MultiSurface><gml:MultiSurface>"
        f"<gml:surfaceMember>{_TRI}</gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod3MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    geometries = parse_citygml(xml).objects["B1"].geometries
    assert len(geometries) == 1
    assert geometries[0].lod == "3"
    assert [surface.type for surface in geometries[0].semantics] == ["RoofSurface"]


def test_building_behaviour_is_unchanged() -> None:
    # The existing LOD2 semantic-building path must be untouched by the breadth generalization.
    model = parse_citygml(_boundary("bldg:Building", "RoofSurface", "B1"))
    geometry = model.objects["B1"].geometries[0]
    assert geometry.lod == "2"
    assert [surface.type for surface in geometry.semantics] == ["RoofSurface"]


def test_breadth_objects_flatten_to_a_feature_collection() -> None:
    # A furniture object reaches a FeatureCollection through the driver (direct geometry included).
    frame = CityGmlDriver().read(_direct("frn:CityFurniture", "frn:lod1Geometry", "F1"))
    assert len(frame) >= 1
    assert "F1" in set(frame["city_object_id"])


def test_direct_geometry_without_a_lod_container_has_lod_none() -> None:
    # A polygon held straight on the feature (no lodX wrapper) still reads; its LOD is unknown.
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember>"
        f'<frn:CityFurniture gml:id="F1">{_TRI}</frn:CityFurniture>'
        "</core:cityObjectMember></core:CityModel>"
    )
    geometry = parse_citygml(xml).objects["F1"].geometries[0]
    assert geometry.lod is None
    assert len(geometry.boundaries) == 1


def test_polygon_outside_any_feature_is_ignored() -> None:
    # A stray polygon not inside a recognized city object has no nearest feature and is dropped.
    xml = (
        f"<core:CityModel {_NS}>"
        f"<core:cityObjectMember><frn:CityFurniture gml:id='F1'>"
        f"<frn:lod1Geometry><gml:MultiSurface><gml:surfaceMember>{_TRI}"
        "</gml:surfaceMember></gml:MultiSurface></frn:lod1Geometry></frn:CityFurniture>"
        f"</core:cityObjectMember>{_TRI}</core:CityModel>"
    ).replace("'", '"')
    model = parse_citygml(xml)
    assert set(model.objects) == {"F1"}
    assert (
        len(model.objects["F1"].geometries[0].boundaries) == 1
    )  # the stray triangle not added


def test_direct_degenerate_polygon_is_skipped() -> None:
    # A direct polygon whose ring has fewer than three points is not a surface and is dropped.
    two_point = _RING.format("0 0 0 1 0 0")
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember>"
        f'<frn:CityFurniture gml:id="F1"><frn:lod1Geometry><gml:MultiSurface>'
        f"<gml:surfaceMember>{two_point}</gml:surfaceMember>"
        "</gml:MultiSurface></frn:lod1Geometry></frn:CityFurniture>"
        "</core:cityObjectMember></core:CityModel>"
    )
    assert (
        parse_citygml(xml).objects["F1"].geometries == []
    )  # no valid surface -> no geometry


def _building_lod2_roof_and_lod1_block() -> str:
    """A building with a LOD2 RoofSurface boundary surface AND a directly-held lod1Solid block."""
    roof = _RING.format("0 0 9 2 0 9 2 2 9 0 0 9")
    block = _RING.format("0 0 0 2 0 0 2 2 0 0 0 0")
    return (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
        f"<gml:surfaceMember>{roof}</gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "<bldg:lod1Solid><gml:Solid><gml:exterior><gml:CompositeSurface>"
        f"<gml:surfaceMember>{block}</gml:surfaceMember>"
        "</gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod1Solid>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')


def test_direct_lod1_geometry_is_separate_from_lod2_boundary_surfaces() -> None:
    # A building with LOD2 boundary surfaces AND a directly-held lod1Solid must yield TWO geometries
    # at their own LODs, not one merged/relabelled LOD2 geometry (review H1).
    geometries = (
        parse_citygml(_building_lod2_roof_and_lod1_block()).objects["B1"].geometries
    )
    by_lod = {geometry.lod: geometry for geometry in geometries}
    assert sorted(by_lod) == ["1", "2"]
    assert [s.type for s in by_lod["2"].semantics] == [
        "RoofSurface"
    ]  # LOD2 keeps its semantics
    assert by_lod["1"].semantics == []  # the LOD1 block carries none
    assert len(by_lod["1"].boundaries) == 1
    assert len(by_lod["2"].boundaries) == 1


def test_identical_surface_at_two_lods_survives_in_both() -> None:
    # A surface present at lod1 AND lod2 with identical coordinates must appear in BOTH geometries;
    # the coordinate de-dup is per-LOD (dual-encoding within one LOD), not cross-LOD (review R2-M2).
    same = _RING.format("0 0 0 2 0 0 2 2 0 0 0 0")
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod1MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        f"{same}</gml:surfaceMember></gml:MultiSurface></bldg:lod1MultiSurface>"
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        f"{same}</gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    by_lod = {g.lod: g for g in parse_citygml(xml).objects["B1"].geometries}
    assert sorted(by_lod) == ["1", "2"]
    assert len(by_lod["1"].boundaries) == 1
    assert len(by_lod["2"].boundaries) == 1


def test_dual_encoded_surface_is_not_double_counted() -> None:
    # The same polygon coordinates under a boundary surface AND inline in a lod2 solid must be
    # collected once, not twice (review H1 — dedupe by coordinate, not object identity).
    same = _RING.format("0 0 9 2 0 9 2 2 9 0 0 9")
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
        f"<gml:surfaceMember>{same}</gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "<bldg:lod2Solid><gml:Solid><gml:exterior><gml:CompositeSurface>"
        f"<gml:surfaceMember>{same}</gml:surfaceMember>"
        "</gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod2Solid>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    total = sum(len(g.boundaries) for g in parse_citygml(xml).objects["B1"].geometries)
    assert total == 1  # the dual-encoded roof is kept once


def test_boundary_surface_made_of_triangles_is_read_with_semantics() -> None:
    # A boundary surface whose LOD2 geometry is a gml:TriangulatedSurface/gml:Triangle (valid CityGML)
    # must be read AND keep its semantic type — not dropped (review R2-M1: pass 1 only matched Polygon).
    tri = (
        "<gml:Triangle><gml:exterior><gml:LinearRing>"
        '<gml:posList srsDimension="3">0 0 9 2 0 9 2 2 9 0 0 9</gml:posList>'
        "</gml:LinearRing></gml:exterior></gml:Triangle>"
    )
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface>"
        f"<gml:TriangulatedSurface><gml:trianglePatches>{tri}</gml:trianglePatches>"
        "</gml:TriangulatedSurface></bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    geometries = parse_citygml(xml).objects["B1"].geometries
    assert len(geometries) == 1
    assert len(geometries[0].boundaries) == 1
    assert [surface.type for surface in geometries[0].semantics] == ["RoofSurface"]


def test_two_direct_surfaces_sharing_an_exterior_but_different_holes_both_kept() -> (
    None
):
    # De-dup keys on the full surface (all rings), so two surfaces with the same exterior ring but
    # different interior rings (holes) are not collapsed as duplicates (review R2-L1).
    def poly(hole: str) -> str:
        return (
            "<gml:Polygon><gml:exterior><gml:LinearRing>"
            '<gml:posList srsDimension="3">0 0 0 10 0 0 10 10 0 0 0 0</gml:posList>'
            "</gml:LinearRing></gml:exterior><gml:interior><gml:LinearRing>"
            f'<gml:posList srsDimension="3">{hole}</gml:posList>'
            "</gml:LinearRing></gml:interior></gml:Polygon>"
        )

    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod1MultiSurface><gml:MultiSurface>"
        f"<gml:surfaceMember>{poly('1 1 0 2 1 0 2 2 0 1 1 0')}</gml:surfaceMember>"
        f"<gml:surfaceMember>{poly('5 5 0 6 5 0 6 6 0 5 5 0')}</gml:surfaceMember>"
        "</gml:MultiSurface></bldg:lod1MultiSurface>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    geometries = parse_citygml(xml).objects["B1"].geometries
    assert len(geometries) == 1
    assert len(geometries[0].boundaries) == 2  # both surfaces kept (holes differ)


def test_tin_relief_triangles_are_read() -> None:
    # A TINRelief whose geometry is a gml:TriangulatedSurface of gml:Triangle patches must be read,
    # not silently dropped (review L1 — the type was advertised but only gml:Polygon was collected).
    tri = (
        "<gml:Triangle><gml:exterior><gml:LinearRing>"
        '<gml:posList srsDimension="3">0 0 0 1 0 0 1 1 0 0 0 0</gml:posList>'
        "</gml:LinearRing></gml:exterior></gml:Triangle>"
    )
    xml = (
        f"<core:CityModel {_NS} xmlns:dem=\"http://www.opengis.net/citygml/relief/2.0\">"
        '<core:cityObjectMember><dem:TINRelief gml:id="T1"><dem:tin>'
        f"<gml:TriangulatedSurface><gml:trianglePatches>{tri}</gml:trianglePatches>"
        "</gml:TriangulatedSurface></dem:tin></dem:TINRelief></core:cityObjectMember>"
        "</core:CityModel>"
    )
    geometries = parse_citygml(xml).objects["T1"].geometries
    assert len(geometries) == 1
    assert len(geometries[0].boundaries) == 1  # the one triangle, read as a surface


def test_mixed_types_in_one_document_all_read() -> None:
    xml = (
        f"<core:CityModel {_NS}>"
        '<core:cityObjectMember><frn:CityFurniture gml:id="F1">'
        f"<frn:lod1Geometry><gml:MultiSurface><gml:surfaceMember>{_TRI}"
        "</gml:surfaceMember></gml:MultiSurface></frn:lod1Geometry>"
        "</frn:CityFurniture></core:cityObjectMember>"
        '<core:cityObjectMember><bldg:Building gml:id="B1">'
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
        f"<gml:surfaceMember>{_TRI}</gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    )
    model = parse_citygml(xml)
    assert {model.objects["F1"].type, model.objects["B1"].type} == {
        "CityFurniture",
        "Building",
    }
