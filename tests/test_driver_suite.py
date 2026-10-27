"""End-to-end driver suite exercising the #14 DoD through the public driver API.

Each test maps to one DoD bullet (vertices/boundaries round-trip, semantic-surface extraction,
LOD selection, strategy options, can_open sniffing), driven through CityJsonDriver / CityGmlDriver
against the committed hand-built fixtures — deterministic and offline.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

import geopandas as gpd

from pyramids_citymodels.drivers.citygml import CityGmlDriver
from pyramids_citymodels.drivers.cityjson import CityJsonDriver

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"
CITYGML = DATA / "minimal_building.gml"


class TestCityJsonDriverSuite:
    """The CityJSON driver against the committed fixture."""

    def test_vertices_and_boundaries_round_trip(self) -> None:
        """A transformed, dereferenced coordinate reaches the output geometry."""
        frame = CityJsonDriver().read(CITYJSON, strategy="semantic_surfaces")
        ground = frame[frame["surface_type"] == "GroundSurface"].geometry.iloc[0]
        assert (1.0, 1.0, 0.0) in list(ground.exterior.coords)

    def test_semantic_surface_extraction(self) -> None:
        """Every boundary-surface type is extracted into the surface_type column."""
        frame = CityJsonDriver().read(CITYJSON, strategy="semantic_surfaces")
        assert set(frame["surface_type"]) == {
            "GroundSurface",
            "WallSurface",
            "RoofSurface",
        }

    def test_lod_selection(self) -> None:
        """An explicit lod selects matching geometry; a missing one yields an empty frame."""
        assert len(CityJsonDriver().read(CITYJSON, lod="2")) == 4
        assert len(CityJsonDriver().read(CITYJSON, lod="3")) == 0

    def test_strategy_options(self) -> None:
        """footprint yields one row per object; semantic_surfaces one row per surface."""
        footprint = CityJsonDriver().read(CITYJSON, strategy="footprint")
        surfaces = CityJsonDriver().read(CITYJSON, strategy="semantic_surfaces")
        assert len(footprint) == 2
        assert len(surfaces) == 4
        assert isinstance(footprint, gpd.GeoDataFrame)

    def test_can_open_sniffing(self) -> None:
        """can_open claims CityJSON and declines GeoJSON / CityGML."""
        assert CityJsonDriver.can_open(CITYJSON) is True
        assert CityJsonDriver.can_open('{"type": "FeatureCollection"}') is False
        assert CityJsonDriver.can_open(CITYGML) is False


class TestCityGmlDriverSuite:
    """The CityGML driver against the committed LOD2 fixture."""

    def test_semantic_surface_extraction_and_round_trip(self) -> None:
        """Boundary-surface semantics are reconstructed and coordinates round-trip."""
        frame = CityGmlDriver().read(CITYGML, strategy="semantic_surfaces")
        assert set(frame["surface_type"]) == {
            "GroundSurface",
            "RoofSurface",
            "WallSurface",
        }
        ground = frame[frame["surface_type"] == "GroundSurface"].geometry.iloc[0]
        assert (1.0, 1.0, 0.0) in list(ground.exterior.coords)

    def test_strategy_options(self) -> None:
        """footprint collapses the building to a single plan-view row."""
        assert len(CityGmlDriver().read(CITYGML, strategy="footprint")) == 1

    def test_can_open_sniffing(self) -> None:
        """can_open claims CityGML and declines plain XML / CityJSON."""
        assert CityGmlDriver.can_open(CITYGML) is True
        assert CityGmlDriver.can_open("<root/>") is False
        assert CityGmlDriver.can_open(CITYJSON) is False
