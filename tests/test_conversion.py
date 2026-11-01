"""Tests for the IR conversion entry points: CityModel.from_file / to_file (issues #23, #27)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import json
from pathlib import Path

import pytest

from pyramids_citymodels.errors import CityModelError
from pyramids_citymodels.ir import CityModel

DATA = Path(__file__).parent / "data"
CITYJSON = DATA / "minimal.city.json"
CITYGML = DATA / "minimal_building.gml"
TILESET = DATA / "tileset.json"


def test_from_file_dispatches_cityjson_by_content() -> None:
    model = CityModel.from_file(CITYJSON)
    assert model.source_format == "CityJSON"
    assert set(model.objects) == {"bldg1", "bldg1_part"}


def test_from_file_dispatches_citygml_by_suffix() -> None:
    model = CityModel.from_file(CITYGML)
    assert model.source_format == "CityGML"
    assert set(model.objects) == {"B1"}


def test_from_file_dispatches_3dtiles_by_content() -> None:
    model = CityModel.from_file(TILESET)
    assert model.source_format == "3DTiles"


def test_from_file_rejects_an_unrecognized_file(tmp_path: Path) -> None:
    unknown = tmp_path / "x.txt"
    unknown.write_text("not a city model", encoding="utf-8")
    with pytest.raises(CityModelError, match="no city-model reader recognizes"):
        CityModel.from_file(unknown)


def test_to_file_round_trips_cityjson_through_disk(tmp_path: Path) -> None:
    model = CityModel.from_file(CITYJSON)
    out = tmp_path / "out.city.json"
    assert model.to_file(out) is None
    assert CityModel.from_file(out) == model


def test_convert_citygml_to_cityjson_is_a_single_expression(tmp_path: Path) -> None:
    """The ADR-0005 headline: from_file(gml).to_file(cityjson), never via FeatureCollection."""
    out = tmp_path / "building.city.json"
    CityModel.from_file(CITYGML).to_file(out)
    converted = CityModel.from_file(out)
    assert converted.source_format == "CityJSON"
    assert set(converted.objects) == {"B1"}
    # Geometry survives the format change: the roof surface is still there.
    types = {s.type for s in converted.objects["B1"].geometries[0].semantics}
    assert "RoofSurface" in types


def test_convert_cityjson_to_citygml_preserves_hierarchy(tmp_path: Path) -> None:
    out = tmp_path / "building.gml"
    CityModel.from_file(CITYJSON).to_file(out)
    converted = CityModel.from_file(out)
    assert converted.objects["bldg1_part"].parent == "bldg1"


@pytest.mark.parametrize(
    ("suffix", "expected_type"),
    [
        (".city.json", "CityJSON"),
        (".cityjson", "CityJSON"),
        (".json", "CityJSON"),
        (".city.jsonl", "CityJSONFeature"),
        (".jsonl", "CityJSONFeature"),
        (".gml", "CityModel"),
        (".fg.json", "FeatureCollection"),
        (".jsonfg", "FeatureCollection"),
    ],
)
def test_to_file_selects_the_writer_by_suffix(
    tmp_path: Path, suffix: str, expected_type: str
) -> None:
    model = CityModel.from_file(CITYJSON)
    out = tmp_path / f"out{suffix}"
    model.to_file(out)
    text = out.read_text(encoding="utf-8")
    if suffix in (".city.jsonl", ".jsonl"):
        # Sequence: the type being checked is on the first feature line, not the header.
        assert json.loads(text.strip().splitlines()[1])["type"] == expected_type
    elif suffix == ".gml":
        assert "CityModel" in text
    else:
        assert json.loads(text)["type"] == expected_type


def test_to_file_rejects_an_unknown_suffix() -> None:
    with pytest.raises(CityModelError, match="cannot infer a city-model writer"):
        CityModel.from_file(CITYJSON).to_file("out.dwg")


def test_to_cityjson_returns_a_dict_without_a_path() -> None:
    document = CityModel.from_file(CITYJSON).to_cityjson()
    assert isinstance(document, dict) and document["type"] == "CityJSON"
