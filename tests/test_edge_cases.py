"""Edge-case coverage for the parsers, flattener and IR.

Exercises the branches the minimal happy-path fixtures do not: BuildingPart hierarchy and
nearest-feature attribution, ``gml:pos`` rings and interior rings, anonymous ids, string/bytes
sources, the no-ground footprint fallback, an empty model, and the readers' error paths.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

import geopandas as gpd
import pytest

from pyramids_citymodels.flatten import to_feature_collection
from pyramids_citymodels.ir import CityModel, Geometry, SemanticSurface, iter_surfaces
from pyramids_citymodels.readers.citygml import looks_like_citygml, parse_citygml
from pyramids_citymodels.readers.cityjson import parse_cityjson

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)

GML_WITH_PART = f"""<?xml version="1.0"?>
<core:CityModel {_NS}>
  <core:cityObjectMember>
    <bldg:Building gml:id="BA">
      <bldg:consistsOfBuildingPart>
        <bldg:BuildingPart gml:id="BA_p1">
          <bldg:boundedBy>
            <bldg:RoofSurface>
              <bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior><gml:LinearRing>
                    <gml:pos srsDimension="3">0 0 2</gml:pos>
                    <gml:pos srsDimension="3">2 0 2</gml:pos>
                    <gml:pos srsDimension="3">2 2 2</gml:pos>
                    <gml:pos srsDimension="3">0 0 2</gml:pos>
                  </gml:LinearRing></gml:exterior>
                  <gml:interior><gml:LinearRing>
                    <gml:posList srsDimension="3">0.5 0.5 2 1 0.5 2 1 1 2 0.5 0.5 2</gml:posList>
                  </gml:LinearRing></gml:interior>
                </gml:Polygon>
              </gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>
            </bldg:RoofSurface>
          </bldg:boundedBy>
        </bldg:BuildingPart>
      </bldg:consistsOfBuildingPart>
    </bldg:Building>
  </core:cityObjectMember>
</core:CityModel>"""

GML_ANONYMOUS = f"""<core:CityModel {_NS}>
  <core:cityObjectMember>
    <bldg:Building>
      <bldg:boundedBy><bldg:WallSurface>
        <bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>
          <gml:Polygon><gml:exterior><gml:LinearRing>
            <gml:posList srsDimension="3">0 0 0 1 0 0 1 0 1 0 0 0</gml:posList>
          </gml:LinearRing></gml:exterior></gml:Polygon>
        </gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>
      </bldg:WallSurface></bldg:boundedBy>
    </bldg:Building>
  </core:cityObjectMember>
</core:CityModel>"""


def test_citygml_building_part_becomes_linked_object() -> None:
    model = parse_citygml(GML_WITH_PART)
    assert set(model.objects) == {"BA", "BA_p1"}
    assert model.objects["BA"].children == ["BA_p1"]
    assert model.objects["BA_p1"].parent == "BA"
    # The roof surface is inside the part, so it is attributed to the part, not the building.
    assert model.objects["BA"].geometries == []
    part_geometry = model.objects["BA_p1"].geometries[0]
    assert part_geometry.semantics[0].type == "RoofSurface"


def test_citygml_reads_gml_pos_and_interior_rings() -> None:
    model = parse_citygml(GML_WITH_PART)
    surface = model.objects["BA_p1"].geometries[0].boundaries[0]
    exterior, interior = surface[0], surface[1]
    assert exterior[0] == (0.0, 0.0, 2.0)
    assert len(exterior) == 4
    assert len(interior) == 4  # interior ring parsed from posList


def test_citygml_anonymous_building_gets_generated_id_and_bytes_source() -> None:
    model = parse_citygml(GML_ANONYMOUS.encode("utf-8"))
    assert "Building_0" in model.objects
    assert model.objects["Building_0"].geometries[0].semantics[0].type == "WallSurface"


def test_looks_like_citygml_from_string_namespace() -> None:
    assert looks_like_citygml(GML_ANONYMOUS) is True


def test_cityjson_parse_from_open_file_handle() -> None:
    with open(CITYJSON, encoding="utf-8") as handle:
        model = parse_cityjson(handle)
    assert "bldg1" in model.objects


def test_cityjson_without_transform_uses_raw_vertices() -> None:
    document = {
        "type": "CityJSON",
        "version": "1.1",
        "CityObjects": {
            "b": {
                "type": "Building",
                "geometry": [
                    {
                        "type": "MultiSurface",
                        "lod": "1",
                        "boundaries": [[[0, 1, 2]]],
                    }
                ],
            }
        },
        "vertices": [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
    }
    model = parse_cityjson(document)
    ring = model.objects["b"].geometries[0].boundaries[0][0]
    assert ring[1] == (5.0, 0.0, 0.0)


def test_cityjson_rejects_non_cityjson_and_bad_type() -> None:
    with pytest.raises(ValueError, match="not a CityJSON document"):
        parse_cityjson({"type": "FeatureCollection"})
    with pytest.raises(TypeError):
        parse_cityjson(12345)


def test_footprint_falls_back_to_all_surfaces_when_no_ground() -> None:
    document = {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": {
            "roof_only": {
                "type": "Building",
                "geometry": [
                    {
                        "type": "MultiSurface",
                        "lod": "2",
                        "boundaries": [[[0, 1, 2]]],
                        "semantics": {
                            "surfaces": [{"type": "RoofSurface"}],
                            "values": [0],
                        },
                    }
                ],
            }
        },
        "vertices": [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [1.0, 1.0, 5.0]],
    }
    result = to_feature_collection(parse_cityjson(document), strategy="footprint")
    assert len(result) == 1
    assert not result.geometry.iloc[0].has_z
    assert result.geometry.iloc[0].area == pytest.approx(0.5)


def test_empty_model_flattens_to_empty_feature_collection() -> None:
    result = to_feature_collection(CityModel(crs="EPSG:4326"))
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0


def test_geometry_semantic_for_handles_bad_indices() -> None:
    geometry = Geometry(
        type="MultiSurface",
        lod="2",
        boundaries=[],
        semantics=[SemanticSurface(type="RoofSurface")],
    )
    assert geometry.semantic_for(None) is None
    assert geometry.semantic_for(5) is None
    assert geometry.semantic_for("x") is None
    assert geometry.semantic_for(0).type == "RoofSurface"


def test_iter_surfaces_on_non_list_yields_nothing() -> None:
    assert list(iter_surfaces(None, None)) == []
    assert list(iter_surfaces(42, None)) == []
