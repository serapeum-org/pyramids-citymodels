"""Tests for the CityGML writer (issue #26) and its round-trip fidelity contract."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path
from xml.etree import ElementTree

from pyramids_citymodels.ir import CityModel, CityObject, Geometry, SemanticSurface
from pyramids_citymodels.readers.citygml import parse_citygml
from pyramids_citymodels.writers.citygml import write_citygml

DATA = Path(__file__).parent / "data"
CITYGML = DATA / "minimal_building.gml"


def _local_names(xml: str) -> set[str]:
    root = ElementTree.fromstring(xml)
    return {element.tag.rsplit("}", 1)[-1] for element in root.iter()}


def test_round_trip_reproduces_the_model() -> None:
    """parse → write → parse reproduces the CityGML-sourced IR (geometry + semantics + hierarchy)."""
    model = parse_citygml(CITYGML)
    assert parse_citygml(write_citygml(model)) == model


def test_output_is_well_formed_citygml_with_expected_elements() -> None:
    names = _local_names(write_citygml(parse_citygml(CITYGML)))
    assert {
        "CityModel",
        "cityObjectMember",
        "Building",
        "boundedBy",
        "Polygon",
    } <= names
    assert {"RoofSurface", "WallSurface", "GroundSurface"} <= names


def test_object_id_and_type_are_written_as_gml_id() -> None:
    model = parse_citygml(CITYGML)
    reparsed = parse_citygml(write_citygml(model))
    assert set(reparsed.objects) == {"B1"}
    assert reparsed.objects["B1"].type == "Building"


def test_crs_round_trips_through_the_envelope() -> None:
    model = parse_citygml(CITYGML)
    assert parse_citygml(write_citygml(model)).crs == "EPSG:25832"


def test_building_part_hierarchy_nests_and_round_trips() -> None:
    model = CityModel(
        objects={
            "bldg": CityObject(id="bldg", type="Building", children=["part"]),
            "part": CityObject(
                id="part",
                type="BuildingPart",
                parents=["bldg"],
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
                        ],
                        semantics=[SemanticSurface(type="RoofSurface")],
                        semantic_values=[0],
                    )
                ],
            ),
        }
    )
    xml = write_citygml(model)
    assert "consistsOfBuildingPart" in _local_names(xml)
    reparsed = parse_citygml(xml)
    assert reparsed.objects["part"].parent == "bldg"
    assert reparsed.objects["bldg"].children == ["part"]


def test_attributes_keep_their_type_via_native_and_generic_elements() -> None:
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                attributes={
                    "measuredHeight": 10.0,  # native bldg: element
                    "storeysAboveGround": 3,  # native bldg: element (int)
                    "owner": "City",  # gen:stringAttribute
                    "roomCount": 7,  # gen:intAttribute
                    "area": 12.5,  # gen:doubleAttribute
                    "heritage": True,  # bool → gen:stringAttribute, not intAttribute
                },
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                        semantics=[SemanticSurface(type="GroundSurface")],
                        semantic_values=[0],
                    )
                ],
            )
        }
    )
    attributes = parse_citygml(write_citygml(model)).objects["b"].attributes
    assert attributes["measuredHeight"] == 10.0
    assert attributes["storeysAboveGround"] == 3
    assert attributes["owner"] == "City"
    assert attributes["roomCount"] == 7
    assert attributes["area"] == 12.5
    assert (
        attributes["heritage"] == "True"
    )  # bool kept as text, not coerced to a number


def test_interior_rings_are_written_as_holes() -> None:
    exterior = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
    hole = [(2.0, 2.0, 0.0), (4.0, 2.0, 0.0), (4.0, 4.0, 0.0), (2.0, 4.0, 0.0)]
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[[exterior, hole]],
                        semantics=[SemanticSurface(type="GroundSurface")],
                        semantic_values=[0],
                    )
                ],
            )
        }
    )
    xml = write_citygml(model)
    assert "interior" in _local_names(xml)
    surface = parse_citygml(xml).objects["b"].geometries[0].boundaries[0]
    assert len(surface) == 2  # exterior + one hole


def test_surface_without_semantics_is_written_as_plain_geometry() -> None:
    """A surface with no semantic type still yields valid CityGML (a bare lod MultiSurface)."""
    model = CityModel(
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
                        semantics=[],
                        semantic_values=None,
                    )
                ],
            )
        }
    )
    xml = write_citygml(model)
    names = _local_names(xml)
    # Written directly under the building, not wrapped in a bldg:<type>Surface boundary element.
    assert "lod2MultiSurface" in names
    assert not ({"RoofSurface", "WallSurface", "GroundSurface"} & names)


def test_model_without_crs_or_geometry_omits_the_envelope() -> None:
    model = CityModel(objects={"b": CityObject(id="b", type="Building")})
    assert "Envelope" not in _local_names(write_citygml(model))


def test_degenerate_surface_emits_no_polygon() -> None:
    """A boundary surface whose ring has too few points produces no bldg:boundedBy."""
    model = CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]
                        ],  # 2-point ring
                        semantics=[SemanticSurface(type="RoofSurface")],
                        semantic_values=[0],
                    )
                ],
            )
        }
    )
    names = _local_names(write_citygml(model))
    assert "Polygon" not in names and "RoofSurface" not in names


def test_non_building_part_child_is_still_emitted() -> None:
    """A child that is not a BuildingPart is attached (not nested) so its geometry is not lost."""
    model = CityModel(
        objects={
            "site": CityObject(id="site", type="Building", children=["annex"]),
            "annex": CityObject(
                id="annex",
                type="Building",
                parents=["site"],
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                        semantics=[SemanticSurface(type="GroundSurface")],
                        semantic_values=[0],
                    )
                ],
            ),
        }
    )
    names = _local_names(write_citygml(model))
    assert "consistsOfBuildingPart" not in names
    assert "GroundSurface" in names  # the annex's geometry is present


def test_write_to_path_returns_none(tmp_path: Path) -> None:
    out = tmp_path / "out.gml"
    assert write_citygml(parse_citygml(CITYGML), out) is None
    assert "CityModel" in _local_names(out.read_text(encoding="utf-8"))
