"""Tests for CityGML 3.0 reading and writing (§3 gap 4 / #50).

The reader is version-agnostic (it matches CityGML by local name), so it reads a 3.0 document —
`gml/3.2` ids, boundary surfaces under `bldg:boundary`, a `lod2Solid` — and records `version="3.0"`.
The writer now emits valid 3.0 for the same model (3.0 namespaces + `gml/3.2` + the `bldg:boundary`
wrapper), so the version round-trips 3.0 → 3.0.
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


def _solid_30_model() -> CityModel:
    from pyramids_citymodels.ir import SemanticSurface

    return CityModel(
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
                        semantics=[SemanticSurface(type="RoofSurface")],
                        semantic_values=[0],
                    )
                ],
            )
        },
    )


def test_writer_emits_valid_30_without_downgrade_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a downgrade warning would raise
        out = _solid_30_model().to_citygml()
    assert "citygml/3.0" in out  # emitted as 3.0, not downgraded
    assert "http://www.opengis.net/gml/3.2" in out  # 3.0 uses GML 3.2
    assert "citygml/2.0" not in out


def test_30_version_and_semantics_round_trip_30_to_30() -> None:
    back = parse_citygml(_solid_30_model().to_citygml())
    assert back.version == "3.0"  # version round-trips
    assert back.objects["b"].geometries[0].semantics[0].type == "RoofSurface"


def test_30_boundary_surface_uses_core_boundary_and_construction_module() -> None:
    # M2: CityGML 3.0 moved boundary surfaces to the Construction module and the relation to core:
    # <core:boundary> → <con:RoofSurface> → <core:lod2MultiSurface>. (Prefixes are auto-assigned for
    # 3.0, so assert on namespaces/local-names rather than literal prefixes.)
    xml = _solid_30_model().to_citygml()
    assert (
        "http://www.opengis.net/citygml/construction/3.0" in xml
    )  # con: module declared
    assert ":boundary>" in xml  # core:boundary wrapper
    assert ":RoofSurface>" in xml
    assert ":lod2MultiSurface>" in xml


def test_30_relief_without_a_slot_falls_back_and_warns() -> None:
    # R2-L2: the 3.0 path honours the per-type profile — a ReliefFeature (no surface slot) falls back
    # to a bldg: property and warns, just like in 2.0, rather than silently emitting a core: slot.
    model = CityModel(
        version="3.0",
        objects={
            "r": CityObject(
                id="r",
                type="ReliefFeature",
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
    with pytest.warns(UserWarning, match="no geometry slot"):
        model.to_citygml()


def test_unknown_version_downgrades_to_20_with_warning() -> None:
    model = CityModel(
        version="9.9",
        objects={"b": CityObject(id="b", type="Building")},
    )
    with pytest.warns(UserWarning, match="does not round-trip"):
        out = model.to_citygml()
    assert "citygml/2.0" in out
    assert parse_citygml(out).version == "2.0"


def test_writer_does_not_warn_for_a_20_source() -> None:
    model = CityModel(
        version="2.0",
        objects={"b": CityObject(id="b", type="Building")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any version warning would raise
        model.to_citygml()
