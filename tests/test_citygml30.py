"""Tests for CityGML 3.0 reading (§3 gap 4).

The reader is version-agnostic (it matches CityGML by local name), so it reads a 3.0 document —
`gml/3.2` ids, boundary surfaces under `bldg:boundary`, a `lod2Solid` — and records `version="3.0"`.
Emitting *valid* 3.0 is a separate effort (3.0 restructures the geometry model), so the CityGML
writer downgrades a 3.0 source to valid 2.0 and warns that the version does not round-trip.
"""

import warnings

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.citygml import parse_citygml

_NS30 = (
    'xmlns:core="http://www.opengis.net/citygml/3.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/3.0" '
    'xmlns:app="http://www.opengis.net/citygml/appearance/3.0" '
    'xmlns:gml="http://www.opengis.net/gml/3.2"'
)


def _building_30() -> str:
    # 3.0 nests a boundary surface under bldg:boundary (2.0 used bldg:boundedBy).
    return (
        f"<core:CityModel {_NS30}><core:cityObjectMember><bldg:Building gml:id='B30'>"
        "<bldg:boundary><bldg:WallSurface gml:id='w1'><bldg:lod2MultiSurface><gml:MultiSurface>"
        "<gml:surfaceMember><gml:Polygon gml:id='p1'><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 0 1 0 0 1 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface>"
        "</bldg:lod2MultiSurface></bldg:WallSurface></bldg:boundary>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')


def test_version_is_detected_as_30() -> None:
    assert parse_citygml(_building_30()).version == "3.0"


def test_building_and_semantic_surface_are_read_from_30() -> None:
    model = parse_citygml(_building_30())
    assert "B30" in model.objects
    geometry = model.objects["B30"].geometries[0]
    assert [s.type for s in geometry.semantics] == ["WallSurface"]
    assert len(geometry.boundaries) == 1  # the wall surface


def test_gml32_id_is_read() -> None:
    # The gml:id lives in the gml/3.2 namespace; the reader matches it by local name.
    assert "B30" in parse_citygml(_building_30()).objects


def test_30_lodx_solid_reads_as_solid() -> None:
    xml = (
        f"<core:CityModel {_NS30}><core:cityObjectMember><bldg:Building gml:id='B30'>"
        "<bldg:lod2Solid><gml:Solid><gml:exterior><gml:CompositeSurface><gml:surfaceMember>"
        "<gml:Polygon><gml:exterior><gml:LinearRing>"
        "<gml:posList srsDimension='3'>0 0 0 1 0 0 1 1 0 0 0 0</gml:posList>"
        "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember></gml:CompositeSurface>"
        "</gml:exterior></gml:Solid></bldg:lod2Solid>"
        "</bldg:Building></core:cityObjectMember></core:CityModel>"
    ).replace("'", '"')
    assert parse_citygml(xml).objects["B30"].geometries[0].type == "Solid"


def test_writer_downgrades_a_30_source_to_20_with_a_warning() -> None:
    model = CityModel(
        version="3.0",
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
        },
    )
    with pytest.warns(UserWarning, match="does not round-trip"):
        out = model.to_citygml()
    assert "citygml/2.0" in out  # emitted as valid 2.0
    assert "citygml/3.0" not in out
    # The downgraded document is still readable (as 2.0).
    assert parse_citygml(out).version == "2.0"


def test_writer_does_not_warn_for_a_20_source() -> None:
    model = CityModel(
        version="2.0",
        objects={"b": CityObject(id="b", type="Building")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any version warning would raise
        model.to_citygml()
