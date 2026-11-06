"""Tests for CityGML ``app:ParameterizedTexture`` reading (completes feature #16).

The CityGML reader reads `ParameterizedTexture` into a CityJSON-shaped appearance: a `textures[]`
pool, a shared `vertices-texture` UV pool, and a per-geometry `texture` map whose values parallel
the geometry's boundaries down to ring level (`[textureIndex, uv₀, …]` per ring). So a textured
CityGML building carries its texture into CityJSON, not just its material colours.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

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


def test_uv_count_mismatch_falls_back_to_untextured_ring() -> None:
    # Only 3 UV pairs for a 5-vertex ring → ring is untextured ([None]), not a malformed value.
    geometry = (
        parse_citygml(_textured("wall.png", "0 0 1 0 1 1")).objects["B1"].geometries[0]
    )
    values = geometry.appearance["texture"]["rgbTexture"]["values"]
    assert values == [[[None]]]
