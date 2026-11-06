"""Tests for CityGML ``app:ParameterizedTexture`` reading (completes feature #16).

The CityGML reader reads `ParameterizedTexture` into a CityJSON-shaped appearance: a `textures[]`
pool, a shared `vertices-texture` UV pool, and a per-geometry `texture` map whose values parallel
the geometry's boundaries down to ring level (`[textureIndex, uv₀, …]` per ring). So a textured
CityGML building carries its texture into CityJSON, not just its material colours.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.readers.citygml import parse_citygml

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:app="http://www.opengis.net/citygml/appearance/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)


def _textured(image: str, texcoords: str, *, theme: str = "rgbTexture") -> str:
    """A one-wall building whose polygon ``poly1`` is painted by one ParameterizedTexture."""
    return (
        f"<core:CityModel {_NS}>"
        "<core:cityObjectMember><bldg:Building gml:id='B1'><bldg:boundedBy><bldg:WallSurface>"
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        "<gml:Polygon gml:id='poly1'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 0 1 0 0 1 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon>"
        "</gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>"
        "</bldg:WallSurface></bldg:boundedBy></bldg:Building></core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance>"
        f"<app:theme>{theme}</app:theme>"
        f"<app:surfaceDataMember><app:ParameterizedTexture><app:imageURI>{image}</app:imageURI>"
        "<app:target uri='#poly1'><app:TexCoordList>"
        f"<app:textureCoordinates ring='#poly1_0'>{texcoords}</app:textureCoordinates>"
        "</app:TexCoordList></app:target></app:ParameterizedTexture></app:surfaceDataMember>"
        "</app:Appearance></app:appearanceMember></core:CityModel>"
    ).replace("'", '"')


def test_texture_pool_and_image_type() -> None:
    model = parse_citygml(_textured("wall.jpg", "0 0 1 0 1 1 0 1 0 0"))
    assert model.appearances["textures"] == [{"type": "JPG", "image": "wall.jpg"}]


def test_vertices_texture_pool_holds_the_uv_pairs() -> None:
    model = parse_citygml(_textured("wall.png", "0 0 1 0 1 1 0 1 0 0"))
    assert model.appearances["vertices-texture"] == [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]


def test_image_suffix_maps_to_cityjson_texture_type() -> None:
    assert (
        parse_citygml(_textured("t.png", "0 0 1 0 1 1 0 1 0 0")).appearances[
            "textures"
        ][0]["type"]
        == "PNG"
    )
    assert (
        parse_citygml(_textured("t.unknown", "0 0 1 0 1 1 0 1 0 0")).appearances[
            "textures"
        ][0]["type"]
        == "PNG"
    )


def test_per_geometry_texture_values_parallel_the_ring() -> None:
    # 5-vertex closed ring → ring value is [textureIndex, uv0, uv1, uv2, uv3, uv4].
    geometry = (
        parse_citygml(_textured("wall.png", "0 0 1 0 1 1 0 1 0 0"))
        .objects["B1"]
        .geometries[0]
    )
    values = geometry.appearance["texture"]["rgbTexture"]["values"]
    assert values == [[[0, 0, 1, 2, 3, 4]]]  # one surface, one ring


def test_texture_rides_into_cityjson() -> None:
    model = parse_citygml(_textured("wall.png", "0 0 1 0 1 1 0 1 0 0"))
    cityjson = model.to_cityjson()
    assert cityjson["appearance"]["textures"] == [{"type": "PNG", "image": "wall.png"}]
    assert len(cityjson["appearance"]["vertices-texture"]) == 5
    assert "texture" in cityjson["CityObjects"]["B1"]["geometry"][0]


def test_untextured_building_has_no_texture_appearance() -> None:
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        "<gml:Polygon><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 1 0 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    model = parse_citygml(xml)
    assert "textures" not in (model.appearances or {})
    assert model.objects["B1"].geometries[0].appearance is None


def test_uv_count_mismatch_falls_back_to_untextured_ring_with_warning() -> None:
    # 3 UV pairs for a 5-vertex ring is irreconcilable (not the open-ring count 4) → the ring is left
    # untextured ([None]) and a warning is emitted rather than dropping silently.
    with pytest.warns(UserWarning, match="could not be matched"):
        geometry = (
            parse_citygml(_textured("wall.png", "0 0 1 0 1 1"))
            .objects["B1"]
            .geometries[0]
        )
    assert geometry.appearance["texture"]["rgbTexture"]["values"] == [[[None]]]


def test_open_length_uv_list_is_closed_to_match_the_ring() -> None:
    # 4 UV pairs (one per distinct vertex) for a stored closed 5-vertex ring: the texture survives,
    # with the UV list closed by repeating the first index (0,1,2,3 -> 0,1,2,3,0).
    geometry = (
        parse_citygml(_textured("wall.png", "0 0 1 0 1 1 0 1"))
        .objects["B1"]
        .geometries[0]
    )
    values = geometry.appearance["texture"]["rgbTexture"]["values"]
    assert values == [[[0, 0, 1, 2, 3, 0]]]  # [textureIndex, uv0..uv3, uv0 repeated]


def test_parameterized_texture_without_image_is_skipped() -> None:
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        "<gml:Polygon gml:id='poly1'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 1 0 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:Building></core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance><app:theme>t</app:theme>"
        "<app:surfaceDataMember><app:ParameterizedTexture>"  # no app:imageURI
        "<app:target uri='#poly1'><app:TexCoordList>"
        "<app:textureCoordinates ring='#r'>0 0 1 0 1 1 0 0</app:textureCoordinates>"
        "</app:TexCoordList></app:target></app:ParameterizedTexture>"
        "</app:surfaceDataMember></app:Appearance></app:appearanceMember></core:CityModel>"
    ).replace("'", '"')
    model = parse_citygml(xml)
    assert "textures" not in (model.appearances or {})
    assert model.objects["B1"].geometries[0].appearance is None


def test_malformed_texture_coordinates_are_skipped() -> None:
    # Non-numeric texcoords → that ring is dropped; with no resolvable UVs the surface is unpainted,
    # so the texture is still pooled but the geometry carries no appearance.
    model = parse_citygml(_textured("wall.png", "nope nope nope"))
    assert model.appearances["textures"] == [{"type": "PNG", "image": "wall.png"}]
    assert model.objects["B1"].geometries[0].appearance is None


def test_target_without_uri_is_ignored() -> None:
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>"
        "<gml:Polygon gml:id='poly1'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 1 0 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:Building></core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance><app:theme>t</app:theme>"
        "<app:surfaceDataMember><app:ParameterizedTexture><app:imageURI>w.png</app:imageURI>"
        "<app:target><app:TexCoordList>"  # target with no uri attribute
        "<app:textureCoordinates ring='#r'>0 0 1 0 1 1 0 0</app:textureCoordinates>"
        "</app:TexCoordList></app:target></app:ParameterizedTexture>"
        "</app:surfaceDataMember></app:Appearance></app:appearanceMember></core:CityModel>"
    ).replace("'", '"')
    model = parse_citygml(xml)
    # The texture object is still pooled, but no geometry is painted (no resolvable target).
    assert model.appearances["textures"] == [{"type": "PNG", "image": "w.png"}]
    assert model.objects["B1"].geometries[0].appearance is None


def test_partially_textured_multi_surface_fills_unpainted_surface() -> None:
    # Two surfaces, only the first textured; the second is filled with an untextured ring.
    xml = (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2MultiSurface><gml:MultiSurface>"
        "<gml:surfaceMember><gml:Polygon gml:id='poly1'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 1 0 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>"
        "<gml:surfaceMember><gml:Polygon gml:id='poly2'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 9 1 0 9 1 1 9 0 0 9</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>"
        "</gml:MultiSurface></bldg:lod2MultiSurface></bldg:Building></core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance><app:theme>t</app:theme>"
        "<app:surfaceDataMember><app:ParameterizedTexture><app:imageURI>w.png</app:imageURI>"
        "<app:target uri='#poly1'><app:TexCoordList>"
        "<app:textureCoordinates ring='#r'>0 0 1 0 1 1 0 0</app:textureCoordinates>"
        "</app:TexCoordList></app:target></app:ParameterizedTexture>"
        "</app:surfaceDataMember></app:Appearance></app:appearanceMember></core:CityModel>"
    ).replace("'", '"')
    values = (
        parse_citygml(xml)
        .objects["B1"]
        .geometries[0]
        .appearance["texture"]["t"]["values"]
    )
    assert len(values) == 2  # one value per surface
    assert values[0][0][0] == 0  # first surface painted (texture index 0)
    assert values[1] == [[None]]  # second surface filled, untextured


def test_texture_on_a_solid_is_nested_one_shell_deeper() -> None:
    # A textured lodXSolid exercises _nest_appearance_for_solid on the texture branch: the values are
    # wrapped one shell deeper than the MultiSurface form ([[ [ [tex, uv…] ] ]] — shell/surface/ring).
    xml = (
        f"<core:CityModel {_NS}>"
        "<core:cityObjectMember><bldg:Building gml:id='B1'>"
        "<bldg:lod2Solid><gml:Solid><gml:exterior><gml:CompositeSurface><gml:surfaceMember>"
        "<gml:Polygon gml:id='poly1'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 0 1 0 0 1 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:CompositeSurface>"
        "</gml:exterior></gml:Solid></bldg:lod2Solid></bldg:Building></core:cityObjectMember>"
        "<app:appearanceMember><app:Appearance><app:theme>t</app:theme>"
        "<app:surfaceDataMember><app:ParameterizedTexture><app:imageURI>w.png</app:imageURI>"
        "<app:target uri='#poly1'><app:TexCoordList>"
        "<app:textureCoordinates ring='#r'>0 0 1 0 1 1 0 1 0 0</app:textureCoordinates>"
        "</app:TexCoordList></app:target></app:ParameterizedTexture>"
        "</app:surfaceDataMember></app:Appearance></app:appearanceMember></core:CityModel>"
    ).replace("'", '"')
    geometry = parse_citygml(xml).objects["B1"].geometries[0]
    assert geometry.type == "Solid"
    values = geometry.appearance["texture"]["t"]["values"]
    # one shell, one surface, one ring of [textureIndex, uv0..uv4]
    assert values == [[[[0, 0, 1, 2, 3, 4]]]]
