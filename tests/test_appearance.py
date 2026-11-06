"""Tests for appearance fidelity (feature #16).

Two sides: CityJSON materials/textures (incl. the ``vertices-texture`` UV pool) round-trip through
parse → write → parse unchanged, and the CityGML reader reads ``app:X3DMaterial`` into the IR's
appearance pool + per-geometry material maps so CityGML → CityJSON carries materials.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.readers.citygml import parse_citygml
from pyramids_citymodels.readers.cityjson import parse_cityjson
from pyramids_citymodels.writers.cityjson import write_cityjson

# --- CityJSON appearance round-trip -------------------------------------------------------------

_CITYJSON_WITH_APPEARANCE = {
    "type": "CityJSON",
    "version": "2.0",
    "CityObjects": {
        "b": {
            "type": "Building",
            "geometry": [
                {
                    "type": "MultiSurface",
                    "lod": "2",
                    "boundaries": [[[0, 1, 2]]],
                    "material": {"rgb": {"values": [0]}},
                    "texture": {"rgb": {"values": [[[0, 1, 2]]]}},
                }
            ],
        }
    },
    "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
    "appearance": {
        "materials": [{"name": "red", "diffuseColor": [1.0, 0.0, 0.0]}],
        "textures": [{"type": "PNG", "image": "wall.png"}],
        "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    },
}


def test_cityjson_materials_round_trip() -> None:
    out = write_cityjson(parse_cityjson(_CITYJSON_WITH_APPEARANCE))
    assert out["appearance"]["materials"] == [
        {"name": "red", "diffuseColor": [1.0, 0.0, 0.0]}
    ]
    assert out["CityObjects"]["b"]["geometry"][0]["material"] == {
        "rgb": {"values": [0]}
    }


def test_cityjson_textures_and_uv_pool_round_trip() -> None:
    # The vertices-texture UV pool and the per-geometry texture values both survive unchanged.
    out = write_cityjson(parse_cityjson(_CITYJSON_WITH_APPEARANCE))
    assert out["appearance"]["textures"] == [{"type": "PNG", "image": "wall.png"}]
    assert out["appearance"]["vertices-texture"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert out["CityObjects"]["b"]["geometry"][0]["texture"] == {
        "rgb": {"values": [[[0, 1, 2]]]}
    }


def test_cityjson_appearance_is_a_fixed_point() -> None:
    # parse(write(parse(doc))) == parse(doc): appearance included, the round-trip is exact.
    once = parse_cityjson(_CITYJSON_WITH_APPEARANCE)
    assert parse_cityjson(write_cityjson(once)) == once


# --- CityGML X3DMaterial reading ----------------------------------------------------------------

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:app="http://www.opengis.net/citygml/appearance/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)


def _citygml_with_material(diffuse: str = "1.0 0.0 0.0", theme: str = "rgb") -> str:
    """A one-building CityGML doc whose roof polygon ``poly1`` is painted by an X3DMaterial."""
    return (
        f"<core:CityModel {_NS}>"
        '<core:cityObjectMember><bldg:Building gml:id="B1">'
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
        '<gml:surfaceMember><gml:Polygon gml:id="poly1"><gml:exterior><gml:LinearRing>'
        '<gml:posList srsDimension="3">0 0 9 2 0 9 2 2 9 0 0 9</gml:posList>'
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>"
        "</gml:MultiSurface></bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "</bldg:Building></core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance>"
        f"<app:theme>{theme}</app:theme>"
        '<app:surfaceDataMember><app:X3DMaterial gml:id="mat">'
        f"<app:diffuseColor>{diffuse}</app:diffuseColor>"
        "<app:shininess>0.2</app:shininess>"
        "<app:target>#poly1</app:target>"
        "</app:X3DMaterial></app:surfaceDataMember>"
        "</app:Appearance></app:appearanceMember></core:CityModel>"
    )


def test_citygml_x3d_material_becomes_a_document_material() -> None:
    model = parse_citygml(_citygml_with_material())
    assert model.appearances["materials"] == [
        {"diffuseColor": [1.0, 0.0, 0.0], "shininess": 0.2, "name": "mat"}
    ]


def test_citygml_material_target_maps_to_the_painted_surface() -> None:
    model = parse_citygml(_citygml_with_material(theme="rgbTexture"))
    appearance = model.objects["B1"].geometries[0].appearance
    assert appearance == {"material": {"rgbTexture": {"values": [0]}}}


def test_citygml_material_carries_through_to_cityjson() -> None:
    out = parse_citygml(_citygml_with_material()).to_cityjson()
    assert out["appearance"]["materials"][0]["diffuseColor"] == [1.0, 0.0, 0.0]
    assert out["CityObjects"]["B1"]["geometry"][0]["material"] == {
        "rgb": {"values": [0]}
    }


def test_citygml_without_appearance_has_empty_appearances() -> None:
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
        "<gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>"
        '<gml:posList srsDimension="3">0 0 9 2 0 9 2 2 9 0 0 9</gml:posList>'
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>"
        "</gml:MultiSurface></bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    model = parse_citygml(xml)
    assert model.appearances == {}
    assert model.objects["B1"].geometries[0].appearance is None


def test_citygml_material_non_numeric_color_is_skipped_not_fatal() -> None:
    # A malformed diffuseColor is dropped (not stored, not raised); the material still registers.
    model = parse_citygml(_citygml_with_material(diffuse="red green blue"))
    material = model.appearances["materials"][0]
    assert "diffuseColor" not in material  # unparseable colour omitted
    assert material["shininess"] == 0.2  # the rest of the material is intact


def test_citygml_material_reads_is_smooth() -> None:
    xml = _citygml_with_material().replace(
        "<app:shininess>0.2</app:shininess>", "<app:isSmooth>true</app:isSmooth>"
    )
    assert parse_citygml(xml).appearances["materials"][0]["isSmooth"] is True


def test_citygml_material_without_a_gml_id_has_no_name() -> None:
    xml = _citygml_with_material().replace(
        '<app:X3DMaterial gml:id="mat">', "<app:X3DMaterial>"
    )
    assert "name" not in parse_citygml(xml).appearances["materials"][0]


def test_citygml_material_targeting_no_surface_paints_nothing() -> None:
    # The material registers in the pool, but no feature gets a material map (nothing matched).
    xml = _citygml_with_material().replace(
        "<app:target>#poly1</app:target>", "<app:target>#ghost</app:target>"
    )
    model = parse_citygml(xml)
    assert len(model.appearances["materials"]) == 1
    assert model.objects["B1"].geometries[0].appearance is None


def test_citygml_unpainted_surface_gets_a_null_material_value() -> None:
    # Two surfaces, only one targeted: the painted one is index 0, the other is null.
    xml = (
        f"<core:CityModel {_NS}>"
        '<core:cityObjectMember><bldg:Building gml:id="B1"><bldg:boundedBy>'
        "<bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        '<gml:Polygon gml:id="poly1"><gml:exterior><gml:LinearRing>'
        '<gml:posList srsDimension="3">0 0 9 2 0 9 2 2 9 0 0 9</gml:posList>'
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
        "<bldg:boundedBy><bldg:WallSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
        '<gml:surfaceMember><gml:Polygon gml:id="poly2"><gml:exterior><gml:LinearRing>'
        '<gml:posList srsDimension="3">0 0 0 2 0 0 2 0 9 0 0 0</gml:posList>'
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:WallSurface></bldg:boundedBy></bldg:Building>"
        "</core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance><app:theme>rgb</app:theme>"
        '<app:surfaceDataMember><app:X3DMaterial gml:id="mat">'
        "<app:diffuseColor>0.1 0.2 0.3</app:diffuseColor><app:target>#poly1</app:target>"
        "</app:X3DMaterial></app:surfaceDataMember></app:Appearance></app:appearanceMember>"
        "</core:CityModel>"
    )
    values = (
        parse_citygml(xml)
        .objects["B1"]
        .geometries[0]
        .appearance["material"]["rgb"]["values"]
    )
    assert values == [0, None]  # roof painted (index 0), wall unpainted (null)
