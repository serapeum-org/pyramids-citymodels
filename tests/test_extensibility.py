"""Tests for the extensibility hooks: pass-through, appearances, versioning, unsupported types (#13)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

import pytest

from pyramids_citymodels.errors import CityModelError, UnsupportedGeometryError
from pyramids_citymodels.flatten import to_feature_collection
from pyramids_citymodels.readers.citygml import parse_citygml
from pyramids_citymodels.readers.cityjson import parse_cityjson

DATA = Path(__file__).parent / "data"

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)


class TestCityJsonPassThrough:
    """CityJSON Extensions, unknown attributes, and appearances survive parsing."""

    def _doc(self) -> dict:
        return {
            "type": "CityJSON",
            "version": "2.0",
            "extensions": {
                "Noise": {"url": "http://example.com/noise.json", "version": "1.0"}
            },
            "appearance": {"materials": [{"name": "roofmat"}], "textures": []},
            "CityObjects": {
                "b": {
                    "type": "Building",
                    "attributes": {"+customField": "keep-me", "roofType": "gabled"},
                    "geometry": [
                        {
                            "type": "MultiSurface",
                            "lod": "2",
                            "boundaries": [[[0, 1, 2]]],
                            "material": {"visual": {"values": [0]}},
                        }
                    ],
                }
            },
            "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
        }

    def test_document_extensions_survive(self) -> None:
        model = parse_cityjson(self._doc())
        assert model.extensions["Noise"]["version"] == "1.0"

    def test_unknown_object_attribute_survives(self) -> None:
        model = parse_cityjson(self._doc())
        assert model.objects["b"].attributes["+customField"] == "keep-me"

    def test_document_and_geometry_appearance_captured(self) -> None:
        model = parse_cityjson(self._doc())
        assert model.appearances["materials"][0]["name"] == "roofmat"
        appearance = model.objects["b"].geometries[0].appearance
        assert appearance is not None and "material" in appearance

    def test_absent_appearance_is_none(self) -> None:
        doc = self._doc()
        del doc["CityObjects"]["b"]["geometry"][0]["material"]
        model = parse_cityjson(doc)
        assert model.objects["b"].geometries[0].appearance is None


class TestCityGmlPassThroughAndVersion:
    """CityGML unknown/ADE attributes survive, and the version is detected from the namespace."""

    def test_unknown_ade_attribute_survives(self) -> None:
        energy = 'xmlns:energy="http://www.sig3d.org/citygml/2.0/energy/1.0"'
        xml = f"""<core:CityModel {_NS} {energy}>
          <core:cityObjectMember><bldg:Building gml:id="B">
            <bldg:roofType>1000</bldg:roofType>
            <energy:buildingType>residential</energy:buildingType>
          </bldg:Building></core:cityObjectMember>
        </core:CityModel>"""
        attributes = parse_citygml(xml).objects["B"].attributes
        assert attributes["roofType"] == "1000"
        assert attributes["buildingType"] == "residential"

    def test_version_detected_from_fixture(self) -> None:
        model = parse_citygml(DATA / "minimal_building.gml")
        assert model.version == "2.0"

    @pytest.mark.parametrize("version", ["1.0", "3.0"])
    def test_version_detected_for_each_namespace(self, version: str) -> None:
        xml = (
            f'<core:CityModel xmlns:core="http://www.opengis.net/citygml/{version}">'
            "<core:cityObjectMember/></core:CityModel>"
        )
        assert parse_citygml(xml).version == version

    @pytest.mark.parametrize(
        "xml",
        [
            "<root><child/></root>",  # no namespace at all
            '<x:Thing xmlns:x="http://www.opengis.net/x"><x:a/></x:Thing>',  # non-citygml ns
            '<x:Thing xmlns:x="http://www.opengis.net/citygml"/>',  # citygml ns, no version tail
        ],
    )
    def test_version_is_none_without_a_versioned_citygml_namespace(
        self, xml: str
    ) -> None:
        assert parse_citygml(xml).version is None

    def test_empty_and_whitespace_leaf_attributes_are_skipped(self) -> None:
        xml = f"""<core:CityModel {_NS}>
          <core:cityObjectMember><bldg:Building gml:id="B">
            <bldg:roofType>1000</bldg:roofType>
            <bldg:empty/>
            <bldg:blank>   </bldg:blank>
          </bldg:Building></core:cityObjectMember>
        </core:CityModel>"""
        attributes = parse_citygml(xml).objects["B"].attributes
        assert attributes["roofType"] == "1000"
        assert "empty" not in attributes
        assert "blank" not in attributes


class TestUnsupportedGeometry:
    """A non-areal geometry type is rejected with a clear, catchable error."""

    def _doc(self, geometry_type: str) -> dict:
        return {
            "type": "CityJSON",
            "version": "2.0",
            "CityObjects": {
                "x": {
                    "type": "Building",
                    "geometry": [
                        {"type": geometry_type, "lod": "2", "boundaries": [[0, 1, 2]]}
                    ],
                }
            },
            "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
        }

    def test_multilinestring_raises(self) -> None:
        with pytest.raises(UnsupportedGeometryError, match="not areal"):
            to_feature_collection(parse_cityjson(self._doc("MultiLineString")))

    def test_error_is_a_value_error(self) -> None:
        assert issubclass(UnsupportedGeometryError, ValueError)
        assert issubclass(UnsupportedGeometryError, CityModelError)

    def test_error_names_the_object_and_type(self) -> None:
        with pytest.raises(UnsupportedGeometryError) as exc:
            to_feature_collection(parse_cityjson(self._doc("MultiPoint")))
        message = str(exc.value)
        assert "'x'" in message and "MultiPoint" in message
