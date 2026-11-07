"""Tests for CityGML ParameterizedTexture writing (issue #44).

The CityGML writer emits `app:ParameterizedTexture` (image + per-polygon `app:target` TexCoordLists)
for textured surfaces, reusing the `gml:id`/`_UniqueIds` plumbing from the material writer (#43) and
assigning each painted `gml:LinearRing` an id. A texture on the IR (read from CityJSON or CityGML)
therefore survives `to_citygml()` and round-trips CityJSON → CityGML → CityJSON.
"""

import re
import warnings

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.citygml import parse_citygml


def _texture_model(
    values: object = None, *, geom_type: str = "MultiSurface", image: str = "wall.png"
) -> CityModel:
    """A one-surface building textured by image index 0 (UV indices 0,1,2 of the pool)."""
    if geom_type == "Solid":
        boundaries: object = [[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]]]
        default: object = [[[[0, 0, 1, 2]]]]
    else:
        boundaries = [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]]
        default = [[[0, 0, 1, 2]]]
    return CityModel(
        appearances={
            "textures": [{"type": "PNG", "image": image}],
            "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        },
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type=geom_type,
                        lod="2",
                        boundaries=boundaries,
                        appearance={"texture": {"t": {"values": values or default}}},
                    )
                ],
            )
        },
    )


def test_writer_emits_parameterized_texture_with_image_and_targets() -> None:
    xml = _texture_model().to_citygml()
    assert "<app:ParameterizedTexture" in xml
    assert "<app:imageURI>wall.png</app:imageURI>" in xml
    assert "<app:textureCoordinates" in xml
    assert 'uri="#poly' in xml  # the target references the polygon
    assert 'ring="#ring' in xml  # the texture coordinates reference the ring


def test_texture_image_round_trips_cityjson_to_citygml_to_cityjson() -> None:
    back = parse_citygml(_texture_model().to_citygml())
    assert back.appearances["textures"] == [{"type": "PNG", "image": "wall.png"}]


def test_texture_values_round_trip_on_a_multisurface() -> None:
    geometry = parse_citygml(_texture_model().to_citygml()).objects["b"].geometries[0]
    assert geometry.appearance["texture"]["t"]["values"] == [[[0, 0, 1, 2]]]


def test_texture_coordinates_are_closed_to_the_ring() -> None:
    # The 3-vertex (open) ring's UVs (0,1,2) are closed on write: four coordinate pairs.
    xml = _texture_model().to_citygml()
    coords = re.search(
        r"<app:textureCoordinates[^>]*>([^<]*)</app:textureCoordinates>", xml
    )
    assert coords is not None
    pairs = coords.group(1).split()
    assert len(pairs) == 8  # 4 (u, v) pairs = closed 4-position ring


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
    assert "ParameterizedTexture" not in xml
    assert "Appearance" not in xml


def test_texture_on_a_solid_round_trips_shell_nested() -> None:
    back = (
        parse_citygml(_texture_model(geom_type="Solid").to_citygml())
        .objects["b"]
        .geometries[0]
    )
    assert back.type == "Solid"
    assert back.appearance["texture"]["t"]["values"] == [
        [[[0, 0, 1, 2]]]
    ]  # one shell deeper


def test_material_and_texture_share_one_appearance_per_theme() -> None:
    model = _texture_model()
    model.appearances["materials"] = [{"name": "red", "diffuseColor": [1.0, 0.0, 0.0]}]
    model.objects["b"].geometries[0].appearance["material"] = {"t": {"values": [0]}}
    xml = model.to_citygml()
    assert xml.count("<app:Appearance>") == 1  # one Appearance for theme "t"
    assert "<app:X3DMaterial" in xml
    assert "<app:ParameterizedTexture" in xml
    back = parse_citygml(xml).objects["b"].geometries[0]
    assert back.appearance["material"]["t"]["values"] == [0]
    assert back.appearance["texture"]["t"]["values"] == [[[0, 0, 1, 2]]]


def test_out_of_range_texture_index_warns_and_leaves_ring_untextured() -> None:
    model = _texture_model(values=[[[5, 0, 1, 2]]])  # texture pool has only index 0
    with pytest.warns(UserWarning, match="texture index 5 is out of range"):
        xml = model.to_citygml()
    assert "ParameterizedTexture" not in xml


def test_out_of_range_uv_index_warns_and_leaves_ring_untextured() -> None:
    model = _texture_model(values=[[[0, 0, 1, 9]]])  # UV pool has only 0,1,2
    with pytest.warns(UserWarning, match="UV index"):
        xml = model.to_citygml()
    assert "ParameterizedTexture" not in xml


def test_non_integer_texture_index_warns_not_crash() -> None:
    model = _texture_model(values=[[["0", 0, 1, 2]]])
    with pytest.warns(UserWarning, match="non-integer texture index"):
        xml = model.to_citygml()
    assert "ParameterizedTexture" not in xml


def test_all_gml_ids_unique_with_textures() -> None:
    xml = _texture_model().to_citygml()
    ids = re.findall(r'gml:id="([^"]*)"', xml)
    assert len(ids) == len(set(ids)), f"duplicate gml:id in {ids}"


def test_writing_a_texture_emits_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _texture_model().to_citygml()


def test_ring_value_without_uvs_leaves_surface_untextured() -> None:
    # A ring value carrying a texture index but no UV coordinates paints nothing (no crash).
    model = _texture_model(values=[[[0]]])
    xml = model.to_citygml()
    assert "ParameterizedTexture" not in xml


def test_extra_ring_values_beyond_the_polygon_rings_are_ignored() -> None:
    # The surface has one ring but the texture value lists two — the extra entry is ignored, and the
    # real ring still paints.
    model = _texture_model(values=[[[0, 0, 1, 2], [0, 0, 1, 2]]])
    xml = model.to_citygml()
    assert xml.count("<app:textureCoordinates") == 1  # only the one real ring
    assert "ParameterizedTexture" in xml
