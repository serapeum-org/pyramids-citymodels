"""Regression tests for the Round-1 review findings on the city-model writers.

Each test encodes a defect the review found (H1/H2/H3/M2/M3/L1/L3/L4) and fails against the code as
it was before the corresponding fix.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from xml.etree import ElementTree

import pytest
from pyramids_citymodels.writers.citygml import write_citygml

from pyramids_citymodels.ir import CityModel, CityObject, Geometry, SemanticSurface
from pyramids_citymodels.readers.citygml import parse_citygml


def _solid_model() -> CityModel:
    """A one-building CityModel whose geometry is a CityJSON-style ``Solid`` (a shell of surfaces)."""
    roof = [[(0.0, 0.0, 3.0), (2.0, 0.0, 3.0), (2.0, 2.0, 3.0), (0.0, 2.0, 3.0)]]
    ground = [[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0)]]
    return CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="Solid",
                        lod="2",
                        boundaries=[[roof, ground]],  # one shell of two surfaces
                        semantics=[
                            SemanticSurface(type="RoofSurface"),
                            SemanticSurface(type="GroundSurface"),
                        ],
                        semantic_values=[[0, 1]],  # nested one level, per Solid
                    )
                ],
            )
        },
        crs="EPSG:25832",
    )


class TestH1SolidGeometry:
    """H1 — a Solid's surfaces must survive the CityGML writer, not be silently dropped."""

    def test_solid_surfaces_survive_to_citygml(self) -> None:
        """A Solid geometry writes both of its surfaces as semantic boundary surfaces."""
        reparsed = parse_citygml(write_citygml(_solid_model()))
        assert reparsed.objects["b"].geometries, "Solid geometry was dropped entirely"
        types = {s.type for s in reparsed.objects["b"].geometries[0].semantics}
        assert types == {"RoofSurface", "GroundSurface"}, f"surfaces lost: {types}"


class TestH2RingClosure:
    """H2 — emitted gml:LinearRings must be closed and carry at least four positions."""

    def test_rings_are_closed_and_have_four_positions(self) -> None:
        """A 3-point (triangle) CityJSON ring is emitted closed with >= 4 positions."""
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
                            semantics=[SemanticSurface(type="RoofSurface")],
                            semantic_values=[0],
                        )
                    ],
                )
            }
        )
        root = ElementTree.fromstring(write_citygml(model))
        pos_list = next(e for e in root.iter() if e.tag.endswith("posList"))
        dim = int(pos_list.get("srsDimension"))
        numbers = [float(t) for t in pos_list.text.split()]
        points = [tuple(numbers[i : i + dim]) for i in range(0, len(numbers), dim)]
        assert len(points) >= 4, f"ring has only {len(points)} positions (< 4)"
        assert points[0] == points[-1], (
            f"ring is not closed: {points[0]} != {points[-1]}"
        )


class TestH3VersionPreserved:
    """H3 — the source CityGML version must round-trip (for 1.0/2.0 the writer matches)."""

    def test_citygml_1_0_version_round_trips(self) -> None:
        """A CityGML 1.0 source is re-emitted so its version reads back as 1.0, not 2.0."""
        xml = (
            '<core:CityModel xmlns:core="http://www.opengis.net/citygml/1.0" '
            'xmlns:bldg="http://www.opengis.net/citygml/building/1.0" '
            'xmlns:gml="http://www.opengis.net/gml">'
            '<core:cityObjectMember><bldg:Building gml:id="B">'
            "<bldg:boundedBy><bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface>"
            "<gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing>"
            '<gml:posList srsDimension="3">0 0 1 1 0 1 1 1 1 0 0 1</gml:posList>'
            "</gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>"
            "</gml:MultiSurface></bldg:lod2MultiSurface></bldg:RoofSurface></bldg:boundedBy>"
            "</bldg:Building></core:cityObjectMember></core:CityModel>"
        )
        model = parse_citygml(xml)
        assert model.version == "1.0", f"source parsed as {model.version!r}"
        assert parse_citygml(write_citygml(model)).version == "1.0"


class TestM2NonBuildingWarning:
    """M2 — writing an object the CityGML reader cannot read back must warn, not drop silently."""

    def test_non_building_object_warns(self) -> None:
        """A top-level Bridge (unsupported by the semantic reader) triggers a warning."""
        model = CityModel(objects={"x": CityObject(id="x", type="Bridge")})
        with pytest.warns(UserWarning, match="Bridge"):
            write_citygml(model)
