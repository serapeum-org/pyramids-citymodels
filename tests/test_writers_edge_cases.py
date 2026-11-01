"""Edge-case and defensive-branch coverage for the city-model writers (issues #25, #26, #27)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from xml.etree import ElementTree

from pyramids_citymodels.writers.citygml import write_citygml
from pyramids_citymodels.writers.cityjson import write_cityjson, write_cityjsonseq

from pyramids_citymodels.ir import CityModel, CityObject, Geometry, SemanticSurface


def _local_names(xml: str) -> set[str]:
    return {
        element.tag.rsplit("}", 1)[-1] for element in ElementTree.fromstring(xml).iter()
    }


class TestCityJsonEdges:
    """Defensive branches of the CityJSON writer."""

    def test_semantic_surface_parent_is_emitted(self) -> None:
        """A semantic surface with a parent index writes a ``parent`` key.

        Test scenario:
            A child RoofSurface pointing at parent index 0 round-trips the ``parent`` field.
        """
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
                            semantics=[
                                SemanticSurface(type="Roof"),
                                SemanticSurface(type="RoofSurface", parent=0),
                            ],
                            semantic_values=[1],
                        )
                    ],
                )
            }
        )
        surfaces = write_cityjson(model)["CityObjects"]["b"]["geometry"][0][
            "semantics"
        ]["surfaces"]
        assert surfaces[1]["parent"] == 0, f"parent index not emitted: {surfaces[1]}"

    def test_object_without_geometry_omits_the_geometry_key(self) -> None:
        """A city object with no geometries writes no ``geometry`` key.

        Test scenario:
            An object carrying only attributes/links has no ``geometry`` member.
        """
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", attributes={"name": "x"})}
        )
        encoded = write_cityjson(model)["CityObjects"]["b"]
        assert "geometry" not in encoded, f"empty geometry should be omitted: {encoded}"

    def test_cityjsonseq_header_carries_extensions(self) -> None:
        """The sequence header line re-emits the model's extensions.

        Test scenario:
            A model with an ``extensions`` entry writes it into the header object.
        """
        import json

        model = CityModel(
            objects={"b": CityObject(id="b", type="Building")},
            extensions={"Noise": {"url": "http://example.com/noise.json"}},
        )
        header = json.loads(write_cityjsonseq(model).strip().splitlines()[0])
        assert header["extensions"] == {
            "Noise": {"url": "http://example.com/noise.json"}
        }

    def test_cityjsonseq_bundles_a_shared_child_once(self) -> None:
        """A child reachable twice in the descendant walk is included once.

        Test scenario:
            A parent listing the same child id twice yields one entry for it (no duplication,
            no infinite loop).
        """
        model = CityModel(
            objects={
                "root": CityObject(
                    id="root", type="Building", children=["part", "part"]
                ),
                "part": CityObject(id="part", type="BuildingPart", parents=["root"]),
            }
        )
        import json

        feature = json.loads(write_cityjsonseq(model).strip().splitlines()[1])
        assert set(feature["CityObjects"]) == {"root", "part"}, feature["CityObjects"]


class TestCityGmlEdges:
    """Defensive branches of the CityGML writer."""

    def test_none_lod_defaults_to_lod2(self) -> None:
        """A geometry with ``lod=None`` is written under a ``lod2`` wrapper.

        Test scenario:
            A semantic surface on a None-LOD geometry still produces a lod2MultiSurface.
        """
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(
                            type="MultiSurface",
                            lod=None,
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
        assert "lod2MultiSurface" in _local_names(write_citygml(model))

    def test_non_numeric_lod_defaults_to_lod2(self) -> None:
        """A non-numeric ``lod`` string falls back to the ``lod2`` wrapper.

        Test scenario:
            lod="garbage" cannot parse to an int and defaults to 2.
        """
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(
                            type="MultiSurface",
                            lod="garbage",
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
        assert "lod2MultiSurface" in _local_names(write_citygml(model))

    def test_mixed_dimension_coordinates_are_padded(self) -> None:
        """A ring mixing 3D and 2D coordinates pads the short ones to the ring dimension.

        Test scenario:
            A ring whose first point is 3D but a later point is 2D writes a posList whose token
            count is a clean multiple of 3 (the short point padded with 0.0).
        """
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
                                [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0)]]
                            ],
                            semantics=[SemanticSurface(type="GroundSurface")],
                            semantic_values=[0],
                        )
                    ],
                )
            }
        )
        root = ElementTree.fromstring(write_citygml(model))
        pos_list = next(e for e in root.iter() if e.tag.endswith("posList"))
        tokens = pos_list.text.split()
        assert len(tokens) % 3 == 0, f"posList not a multiple of srsDimension: {tokens}"

    def test_non_semantic_degenerate_surface_is_skipped(self) -> None:
        """A non-semantic surface that is also degenerate produces no geometry element.

        Test scenario:
            No semantic type and a 2-point ring → neither a boundedBy nor a bare MultiSurface.
        """
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(
                            type="MultiSurface",
                            lod="2",
                            boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]],
                            semantics=[],
                            semantic_values=None,
                        )
                    ],
                )
            }
        )
        assert "Polygon" not in _local_names(write_citygml(model))

    def test_missing_child_reference_is_skipped(self) -> None:
        """A ``children`` id with no matching object is skipped, not emitted.

        Test scenario:
            A building listing a child id absent from the model produces no nested feature and
            does not raise.
        """
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", children=["ghost"])}
        )
        names = _local_names(write_citygml(model))
        assert "consistsOfBuildingPart" not in names

    def test_crs_without_geometry_emits_srsname_without_corners(self) -> None:
        """A model with a CRS but no coordinates emits an envelope carrying only ``srsName``.

        Test scenario:
            crs set, no geometry → gml:Envelope with srsName and no lower/upperCorner.
        """
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building")}, crs="EPSG:28992"
        )
        root = ElementTree.fromstring(write_citygml(model))
        envelope = next(e for e in root.iter() if e.tag.endswith("Envelope"))
        assert envelope.get("srsName") == "EPSG:28992"
        assert not [c for c in envelope if c.tag.endswith("Corner")], (
            "no corners without coords"
        )
