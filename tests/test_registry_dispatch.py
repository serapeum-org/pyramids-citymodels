"""End-to-end registry dispatch and conformance against the shipped core registry (issue #19).

The other driver tests construct the drivers directly (``CityJsonDriver().read(...)``). These
tests exercise the real seam: that ``pyramids-core``'s ``feature/_drivers.py`` registry discovers
our entry points, that ``FeatureCollection.read_file`` actually dispatches a city-model file into
the right driver (content-claimed and by explicit ``driver=``), that a GeoJSON look-alike still
falls through to OGR, and that every driver passes pyramids' own conformance kit.

The whole module is skipped on a ``pyramids-core`` older than 0.72.0 — the first release that ships
the ``pyramids.vector_drivers`` consumer. The floor in ``pyproject.toml`` is 0.72.0, so CI always
runs these; the skip only protects a developer who pins an older core locally. Capability is
detected by import, never by an environment variable (repo testing rule).
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

import pytest

from pyramids_citymodels.drivers.citygml import CityGmlDriver
from pyramids_citymodels.drivers.cityjson import CityJsonDriver
from pyramids_citymodels.drivers.threedtiles import ThreeDTilesDriver

_drivers = pytest.importorskip(
    "pyramids.feature._drivers",
    reason="pyramids-core < 0.72.0 has no pyramids.vector_drivers consumer",
)
driver_conformance = pytest.importorskip("pyramids.feature.driver_conformance")

from pyramids.feature import FeatureCollection  # noqa: E402  (after the capability gate)

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"
CITYGML = DATA / "minimal_building.gml"
TILESET = DATA / "tileset.json"


@pytest.fixture(autouse=True)
def _fresh_registry() -> None:
    """Drop the process-wide discovery cache so each test sees a clean entry-point walk."""
    _drivers.clear_driver_cache()
    yield
    _drivers.clear_driver_cache()


class TestDiscovery:
    """The installed entry points resolve to our driver classes through the real registry."""

    def test_our_three_drivers_are_discovered(self) -> None:
        discovered = _drivers.vector_drivers()
        assert {"cityjson", "citygml", "3dtiles"} <= set(discovered)

    def test_discovery_resolves_to_our_actual_classes(self) -> None:
        discovered = _drivers.vector_drivers()
        assert discovered["cityjson"] is CityJsonDriver
        assert discovered["citygml"] is CityGmlDriver
        assert discovered["3dtiles"] is ThreeDTilesDriver

    def test_no_driver_is_skipped_for_a_missing_extra(self) -> None:
        """All three declare ``extra = None``, so none is ever gated out at discovery (#12)."""
        for driver in (CityJsonDriver, CityGmlDriver, ThreeDTilesDriver):
            assert driver.extra is None


class TestConformance:
    """Every driver passes pyramids' own vector-driver conformance kit."""

    @pytest.mark.parametrize(
        "driver", [CityJsonDriver, CityGmlDriver, ThreeDTilesDriver]
    )
    def test_driver_conforms(self, driver: type) -> None:
        driver_conformance.assert_vector_driver_conforms(driver)


class TestContentDispatch:
    """``read_file`` routes a city-model file to the driver whose ``can_open`` claims it."""

    def test_cityjson_dispatches(self) -> None:
        frame = FeatureCollection.read_file(CITYJSON)
        assert len(frame) == len(CityJsonDriver().read(CITYJSON))
        assert len(frame) > 0

    def test_citygml_dispatches(self) -> None:
        frame = FeatureCollection.read_file(CITYGML)
        assert set(frame["surface_type"]) == {
            "GroundSurface",
            "RoofSurface",
            "WallSurface",
        }

    def test_3dtiles_dispatches(self) -> None:
        frame = FeatureCollection.read_file(TILESET)
        assert set(frame["city_object_id"]) == {"b0", "b1"}

    def test_select_driver_picks_by_content_for_a_shared_suffix(self) -> None:
        """``tileset.json`` and ``*.city.json`` share the ``.json`` suffix; content decides."""
        assert _drivers.select_driver(str(TILESET)).name == "3dtiles"
        assert _drivers.select_driver(str(CITYJSON)).name == "cityjson"

    def test_compound_city_json_suffix_routes_on_its_last_suffix(self) -> None:
        """A ``.city.json`` matches on its final ``.json`` suffix (the registry takes the last).

        The fixture is named ``minimal.city.json``; the core ``extract_tokens`` keys off the last
        suffix only (``.city.json`` → ``.json``), so the driver must declare ``.json`` — which it
        does — and content-sniff. This is the reconciliation #19 called for.
        """
        from pyramids.base._driver_discovery import extract_tokens

        assert ".json" in CityJsonDriver.suffixes
        assert extract_tokens("building.city.json") == {".json"}
        assert CITYJSON.name.endswith(".city.json")
        assert _drivers.select_driver(str(CITYJSON)).name == "cityjson"


class TestExplicitAndFallthrough:
    """An explicit ``driver=`` wins; a GeoJSON look-alike is left to OGR."""

    def test_explicit_driver_name(self) -> None:
        frame = FeatureCollection.read_file(CITYJSON, driver="cityjson")
        assert len(frame) > 0

    def test_geojson_falls_through_to_ogr(self, tmp_path: Path) -> None:
        geojson = tmp_path / "points.geojson"
        geojson.write_text(
            '{"type": "FeatureCollection", "features": [{"type": "Feature", '
            '"properties": {"n": 1}, "geometry": {"type": "Point", '
            '"coordinates": [4.0, 52.0]}}]}',
            encoding="utf-8",
        )
        # None of our drivers claim a GeoJSON, so read_file uses the built-in OGR reader.
        assert _drivers.select_driver(str(geojson)) is None
        frame = FeatureCollection.read_file(geojson)
        assert len(frame) == 1
