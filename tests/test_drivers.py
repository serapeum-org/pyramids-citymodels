"""Tests for the VectorDriver classes and their registration (issue #11)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import tomllib
from pathlib import Path

import geopandas as gpd

from pyramids_citymodels.drivers.citygml import CityGmlDriver
from pyramids_citymodels.drivers.cityjson import CityJsonDriver

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"
CITYGML = DATA / "minimal_building.gml"
PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def test_drivers_declare_the_class_attribute_contract() -> None:
    for driver in (CityJsonDriver, CityGmlDriver):
        assert isinstance(driver.name, str) and driver.name
        assert isinstance(driver.long_name, str) and driver.long_name
        assert driver.api_version == 1
        assert isinstance(driver.suffixes, tuple) and driver.suffixes
        assert "read" in driver.capabilities
        # Claims are readable without constructing the driver (discovery never builds it).
        assert "read" in type(driver).__dict__.get("capabilities", driver.capabilities)


def test_cityjson_can_open_sniffs_content_not_suffix() -> None:
    assert CityJsonDriver.can_open(CITYJSON) is True
    # A GeoJSON file would share the .json suffix but must be declined.
    assert (
        CityJsonDriver.can_open('{"type": "FeatureCollection", "features": []}')
        is False
    )
    assert CityGmlDriver.can_open(CITYJSON) is False


def test_citygml_can_open_sniffs_namespace() -> None:
    assert CityGmlDriver.can_open(CITYGML) is True
    assert CityGmlDriver.can_open("<root/>") is False
    assert CityJsonDriver.can_open(CITYGML) is False


def test_cityjson_driver_read_returns_feature_collection() -> None:
    result = CityJsonDriver().read(CITYJSON, strategy="semantic_surfaces")
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 4


def test_citygml_driver_read_returns_feature_collection() -> None:
    result = CityGmlDriver().read(CITYGML, strategy="semantic_surfaces")
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 3
    assert set(result["surface_type"]) == {
        "GroundSurface",
        "RoofSurface",
        "WallSurface",
    }


def test_entry_points_registered_under_vector_drivers_group() -> None:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    entry_points = config["project"]["entry-points"]["pyramids.vector_drivers"]
    assert (
        entry_points["cityjson"]
        == "pyramids_citymodels.drivers.cityjson:CityJsonDriver"
    )
    assert (
        entry_points["citygml"] == "pyramids_citymodels.drivers.citygml:CityGmlDriver"
    )
