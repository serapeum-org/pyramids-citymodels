"""Tests for flattening the IR into a FeatureCollection (issues #8, #9, #10)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

import geopandas as gpd

from pyramids_citymodels.flatten import to_feature_collection
from pyramids_citymodels.readers.cityjson import parse_cityjson

DATA = Path(__file__).parent / "data"
MODEL = parse_cityjson(DATA / "minimal.city.json")


def test_result_is_a_feature_collection_in_the_models_crs() -> None:
    result = to_feature_collection(MODEL)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs is not None
    assert result.crs.to_epsg() == 7415


def test_semantic_surfaces_yields_one_row_per_surface() -> None:
    result = to_feature_collection(MODEL, strategy="semantic_surfaces")
    # bldg1 has 3 surfaces, bldg1_part has 1 -> 4 rows.
    assert len(result) == 4
    assert set(result["surface_type"]) == {
        "GroundSurface",
        "WallSurface",
        "RoofSurface",
    }
    assert "parent_id" in result.columns
    assert "city_object_id" in result.columns
    part_rows = result[result["city_object_id"] == "bldg1_part"]
    assert list(part_rows["parent_id"]) == ["bldg1"]


def test_geometry_dimension_controls_z() -> None:
    kept = to_feature_collection(
        MODEL, strategy="semantic_surfaces", geometry_dimension="xyz"
    )
    assert kept.geometry.iloc[0].has_z
    dropped = to_feature_collection(
        MODEL, strategy="semantic_surfaces", geometry_dimension="xy"
    )
    assert not dropped.geometry.iloc[0].has_z


def test_footprint_yields_one_row_per_object_in_2d() -> None:
    result = to_feature_collection(MODEL, strategy="footprint")
    assert len(result) == 2
    assert set(result["city_object_id"]) == {"bldg1", "bldg1_part"}
    assert not result.geometry.iloc[0].has_z
    # bldg1's footprint is its unit-square ground surface.
    bldg1 = result[result["city_object_id"] == "bldg1"].geometry.iloc[0]
    assert bldg1.area == 1.0


def test_lod_selection_filters_geometries() -> None:
    matching = to_feature_collection(MODEL, lod="2")
    assert len(matching) == 4
    missing = to_feature_collection(MODEL, lod="3")
    assert len(missing) == 0
    assert isinstance(missing, gpd.GeoDataFrame)


def test_child_inherits_parent_attributes() -> None:
    without = to_feature_collection(MODEL, child_inherits_parent_attributes=False)
    part_without = without[without["city_object_id"] == "bldg1_part"]
    assert part_without["function"].isna().all()

    inherited = to_feature_collection(MODEL, child_inherits_parent_attributes=True)
    part_with = inherited[inherited["city_object_id"] == "bldg1_part"]
    assert (part_with["function"] == "residential").all()


def test_surface_geometry_is_2_5d_with_non_areal_vertical_walls() -> None:
    """Pin the documented 2.5D contract (MF1): horizontal surfaces are 2D-valid with positive
    area; a vertical wall is deliberately non-areal in the XY plane but keeps its Z."""
    fc = to_feature_collection(MODEL, strategy="semantic_surfaces", geometry_dimension="xyz")
    bldg1 = fc[fc["city_object_id"] == "bldg1"]
    by_type = dict(zip(bldg1["surface_type"], bldg1.geometry))

    assert by_type["RoofSurface"].is_valid and by_type["RoofSurface"].area > 0
    assert by_type["GroundSurface"].is_valid and by_type["GroundSurface"].area > 0

    wall = by_type["WallSurface"]
    assert wall.has_z, "the wall keeps its Z (2.5D)"
    assert not wall.is_valid, "a vertical wall is non-areal in the XY plane by design"
    assert wall.area == 0.0
