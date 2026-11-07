"""Tests for geometry-type preservation (§3 gap 2).

The CityGML reader labels a feature's directly-held geometry `Solid` (not `MultiSurface`) when it
comes from a `lodXSolid` container, and the CityGML writer emits a non-semantic `Solid` as a
`gml:Solid`, so a `Solid` survives a CityJSON ⇄ CityGML round-trip.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.citygml import parse_citygml

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)
_RING = (
    "<gml:Polygon><gml:exterior><gml:LinearRing>"
    '<gml:posList srsDimension="3">{}</gml:posList>'
    "</gml:LinearRing></gml:exterior></gml:Polygon>"
)


def _solid_building(*rings: str) -> str:
    members = "".join(f"<gml:surfaceMember>{r}</gml:surfaceMember>" for r in rings)
    return (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        f"<bldg:lod2Solid><gml:Solid><gml:exterior><gml:CompositeSurface>{members}"
        "</gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod2Solid>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')


def test_reader_labels_a_lodx_solid_as_solid() -> None:
    xml = _solid_building(
        _RING.format("0 0 0 1 0 0 1 1 0 0 0 0"),
        _RING.format("0 0 0 1 0 0 1 0 1 0 0 0"),
    )
    geometry = parse_citygml(xml).objects["B1"].geometries[0]
    assert geometry.type == "Solid"
    assert len(geometry.boundaries) == 1  # one shell
    assert len(geometry.boundaries[0]) == 2  # two surfaces in the shell


def test_reader_still_labels_a_multisurface_as_multisurface() -> None:
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        f"{_RING.format('0 0 0 1 0 0 1 1 0 0 0 0')}</gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    assert parse_citygml(xml).objects["B1"].geometries[0].type == "MultiSurface"


def test_boundary_surface_geometry_stays_multisurface() -> None:
    # A semantic boundary-surface geometry is a MultiSurface regardless of a lodXSolid elsewhere.
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
        f"<gml:surfaceMember>{_RING.format('0 0 9 1 0 9 1 1 9 0 0 9')}</gml:surfaceMember>"
        "</gml:MultiSurface></bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    geometry = parse_citygml(xml).objects["B1"].geometries[0]
    assert geometry.type == "MultiSurface"
    assert [s.type for s in geometry.semantics] == ["RoofSurface"]


def _cityjson_solid() -> CityModel:
    shell = [
        [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]],
        [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)]],
    ]
    return CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[Geometry(type="Solid", lod="2", boundaries=[shell])],
            )
        }
    )


def test_writer_emits_a_plain_solid_as_gml_solid() -> None:
    xml = _cityjson_solid().to_citygml()
    assert "lod2Solid" in xml
    assert "<gml:Solid" in xml


def test_cityjson_solid_round_trips_through_citygml() -> None:
    back = parse_citygml(_cityjson_solid().to_citygml()).objects["b"].geometries[0]
    assert back.type == "Solid"
    assert len(back.boundaries) == 1  # still one shell
    assert len(back.boundaries[0]) == 2  # both surfaces survive


def test_cityjson_solid_round_trips_through_cityjson_unchanged() -> None:
    # The CityJSON path already preserved the type; guard it stays a fixed point.
    model = _cityjson_solid()
    out = model.to_cityjson()
    assert out["CityObjects"]["b"]["geometry"][0]["type"] == "Solid"


def test_material_on_a_solid_is_nested_one_shell_deeper() -> None:
    # A lodXSolid polygon painted by an X3DMaterial: the geometry is a Solid, so its material map
    # values are nested one shell deeper (parallel to the Solid's [[surfaces]] boundaries).
    xml = (
        f"<core:CityModel {_NS} "
        'xmlns:app="http://www.opengis.net/citygml/appearance/2.0">'
        "<core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2Solid><gml:Solid><gml:exterior><gml:CompositeSurface><gml:surfaceMember>"
        "<gml:Polygon gml:id='sp'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 1 0 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:CompositeSurface>"
        "</gml:exterior></gml:Solid></bldg:lod2Solid></bldg:Building></core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance><app:theme>wall</app:theme>"
        "<app:surfaceDataMember><app:X3DMaterial><app:diffuseColor>1 0 0</app:diffuseColor>"
        "<app:target>#sp</app:target></app:X3DMaterial></app:surfaceDataMember>"
        "</app:Appearance></app:appearanceMember></core:CityModel>"
    ).replace("'", '"')
    geometry = parse_citygml(xml).objects["B1"].geometries[0]
    assert geometry.type == "Solid"
    # values nest one level deeper than a MultiSurface's per-surface [idx] list.
    assert geometry.appearance["material"]["wall"]["values"] == [[0]]


def test_citygml_to_cityjson_rings_are_open() -> None:
    # CityJSON requires OPEN rings (no repeated closing vertex). A CityGML LinearRing is closed, so
    # the reader must open it; otherwise external CityJSON validators flag the duplicate vertex (L1).
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        "<gml:Polygon><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 1 0 0 1 0 0 0 0</gml:posList>"  # closed: 5 positions
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    model = parse_citygml(xml)
    ring = model.objects["B1"].geometries[0].boundaries[0][0]
    assert len(ring) == 4  # 4 distinct vertices, closing vertex dropped
    assert ring[0] != ring[-1]  # open
    # The CityJSON vertex indices for the ring also carry no repeated closing index.
    cityjson = model.to_cityjson()
    indices = cityjson["CityObjects"]["B1"]["geometry"][0]["boundaries"][0][0]
    assert len(indices) == 4
    assert indices[0] != indices[-1]


def test_multisolid_is_not_emitted_as_a_single_solid() -> None:
    # A MultiSolid must not be collapsed into one gml:Solid (which would mislabel it Solid on read).
    # It is written as its surfaces instead: type is not preserved, but it is never mislabelled.
    solid_a = [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]]]
    solid_b = [[[(5.0, 5.0, 0.0), (6.0, 5.0, 0.0), (6.0, 6.0, 0.0), (5.0, 5.0, 0.0)]]]
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(type="MultiSolid", lod="2", boundaries=[solid_a, solid_b])
                ],
            )
        }
    )
    xml = model.to_citygml()
    assert "<gml:Solid" not in xml  # not collapsed into a single gml:Solid
    back = parse_citygml(xml).objects["b"].geometries[0]
    assert back.type == "MultiSurface"  # honestly typed, not a bogus Solid
    # both solids' surfaces survive as surfaces of the read-back MultiSurface
    assert len(back.boundaries) == 2
