"""Tests for CityGML attribute ``uom`` (unit of measure) round-tripping (issue #48).

A ``uom`` on a CityGML attribute (e.g. ``<bldg:measuredHeight uom="m">``) is read into a parallel
``CityObject.attribute_units`` side-map (keeping ``attributes`` a plain name→value dict) and
re-emitted by the writer, so the unit survives CityGML → IR → CityGML.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject
from pyramids_citymodels.readers.citygml import parse_citygml

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)


def _building(body: str) -> str:
    return (
        f"<core:CityModel {_NS}><core:cityObjectMember><bldg:Building gml:id='B'>"
        f"{body}</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')


def test_reader_captures_uom_into_attribute_units() -> None:
    model = parse_citygml(
        _building('<bldg:measuredHeight uom="m">12.5</bldg:measuredHeight>')
    )
    obj = model.objects["B"]
    assert obj.attributes["measuredHeight"] == 12.5  # plain value, unchanged
    assert obj.attribute_units == {"measuredHeight": "m"}  # unit in the side map


def test_writer_emits_uom_on_the_native_element() -> None:
    model = CityModel(
        objects={
            "B": CityObject(
                id="B",
                type="Building",
                attributes={"measuredHeight": 12.5},
                attribute_units={"measuredHeight": "m"},
            )
        }
    )
    assert 'uom="m"' in model.to_citygml()


def test_uom_round_trips_citygml_to_citygml() -> None:
    model = parse_citygml(
        _building('<bldg:measuredHeight uom="m">9.0</bldg:measuredHeight>')
    )
    back = parse_citygml(model.to_citygml()).objects["B"]
    assert back.attributes["measuredHeight"] == 9.0
    assert back.attribute_units == {"measuredHeight": "m"}


def test_attribute_without_uom_has_no_unit_entry() -> None:
    model = parse_citygml(_building("<bldg:measuredHeight>7</bldg:measuredHeight>"))
    obj = model.objects["B"]
    assert obj.attributes["measuredHeight"] == 7.0
    assert obj.attribute_units == {}


def test_object_without_units_writes_no_uom() -> None:
    model = CityModel(
        objects={
            "B": CityObject(id="B", type="Building", attributes={"measuredHeight": 3.0})
        }
    )
    assert "uom=" not in model.to_citygml()
