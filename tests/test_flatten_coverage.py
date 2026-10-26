"""Branch-coverage tests for the flattener's private helpers and degenerate-input paths."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import geopandas as gpd

from pyramids_citymodels.flatten import (
    _polygon,
    _select_geometries,
    to_feature_collection,
)
from pyramids_citymodels.ir import CityModel, CityObject, Geometry, SemanticSurface

TRIANGLE = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]


def _single_object_model(city_object: CityObject) -> CityModel:
    """Wrap one city object in a model."""
    model = CityModel()
    model.objects[city_object.id] = city_object
    return model


class TestPolygon:
    """Tests for the private ``_polygon`` surface-to-shapely builder."""

    def test_empty_surface_returns_none(self) -> None:
        """A surface with no non-empty rings yields no polygon.

        Test scenario:
            ``[]`` and ``[[]]`` both have no usable ring, so ``_polygon`` returns ``None``.
        """
        assert _polygon([], keep_z=False) is None, "empty surface must give None"
        assert _polygon([[]], keep_z=False) is None, (
            "surface of empty ring must give None"
        )

    def test_degenerate_exterior_returns_none(self) -> None:
        """An exterior ring of fewer than three points is not a polygon.

        Test scenario:
            A two-point ring returns ``None`` rather than a broken polygon.
        """
        assert _polygon([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]], keep_z=True) is None

    def test_pads_two_dimensional_coordinate_when_keeping_z(self) -> None:
        """A 2D coordinate gains a zero Z when ``keep_z`` is set.

        Test scenario:
            With ``keep_z=True`` a ``(x, y)`` coordinate becomes ``(x, y, 0.0)``.
        """
        polygon = _polygon([[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]], keep_z=True)
        assert polygon is not None
        assert polygon.exterior.coords[0] == (0.0, 0.0, 0.0), "2D coord should pad Z=0"


class TestSelectGeometries:
    """Tests for the private ``_select_geometries`` LOD picker."""

    def test_no_geometries_returns_empty(self) -> None:
        """An object with no geometries selects nothing.

        Test scenario:
            ``_select_geometries([], None)`` returns ``[]``.
        """
        assert _select_geometries([], None) == []

    def test_non_numeric_lod_is_tolerated_in_auto_select(self) -> None:
        """A non-numeric LOD does not crash highest-LOD auto-selection.

        Test scenario:
            A geometry whose ``lod`` is ``"roof"`` is still returned when ``lod=None``; the
            non-numeric value sorts as the lowest rank rather than raising.
        """
        geometry = Geometry(type="MultiSurface", lod="roof", boundaries=[TRIANGLE])
        assert _select_geometries([geometry], None) == [geometry]


class TestToFeatureCollectionEdgeCases:
    """Tests for degenerate and structural paths through ``to_feature_collection``."""

    def test_object_without_geometry_is_skipped(self) -> None:
        """A city object carrying no geometry contributes no rows.

        Test scenario:
            A lone attribute-only object yields an empty FeatureCollection.
        """
        model = _single_object_model(CityObject(id="x", type="Building"))
        result = to_feature_collection(model)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_degenerate_surface_is_dropped_in_semantic_strategy(self) -> None:
        """A degenerate surface is skipped while valid surfaces in the same object survive.

        Test scenario:
            A geometry with one two-point surface and one valid triangle yields a single row.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]], TRIANGLE],
            semantics=[
                SemanticSurface(type="WallSurface"),
                SemanticSurface(type="RoofSurface"),
            ],
            semantic_values=[0, 1],
        )
        model = _single_object_model(
            CityObject(id="b", type="Building", geometries=[geometry])
        )
        result = to_feature_collection(model, strategy="semantic_surfaces")
        assert len(result) == 1, "only the valid surface should produce a row"
        assert result["surface_type"].iloc[0] == "RoofSurface"

    def test_footprint_of_only_degenerate_surfaces_yields_no_row(self) -> None:
        """An object whose every surface is degenerate produces no footprint row.

        Test scenario:
            A geometry with a single two-point surface flattens to an empty collection.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]],
        )
        model = _single_object_model(
            CityObject(id="b", type="Building", geometries=[geometry])
        )
        result = to_feature_collection(model, strategy="footprint")
        assert len(result) == 0

    def test_footprint_unions_disjoint_grounds_into_multipolygon(self) -> None:
        """Two disjoint ground surfaces union into a single MultiPolygon footprint row.

        Test scenario:
            Two far-apart ground squares produce one row whose geometry is a MultiPolygon.
        """
        near = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]]
        far = [
            [(10.0, 10.0, 0.0), (11.0, 10.0, 0.0), (11.0, 11.0, 0.0), (10.0, 11.0, 0.0)]
        ]
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[near, far],
            semantics=[
                SemanticSurface(type="GroundSurface"),
                SemanticSurface(type="GroundSurface"),
            ],
            semantic_values=[0, 1],
        )
        model = _single_object_model(
            CityObject(id="b", type="Building", geometries=[geometry])
        )
        result = to_feature_collection(model, strategy="footprint")
        assert len(result) == 1
        assert result.geometry.iloc[0].geom_type == "MultiPolygon"

    def test_footprint_excludes_non_areal_walls(self) -> None:
        """A vertical wall (valid 3D, zero 2D area) does not pollute a plan-view footprint (SF1).

        Test scenario:
            A wall-only object yields no footprint row, and a ground+wall object yields a
            footprint equal to the ground alone (area 1.0), not a degenerate union.
        """
        wall = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]]
        ground = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]]

        wall_only = _single_object_model(
            CityObject(
                id="w",
                type="Building",
                geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[wall])],
            )
        )
        assert len(to_feature_collection(wall_only, strategy="footprint")) == 0

        mixed = _single_object_model(
            CityObject(
                id="m",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[ground, wall],
                        semantics=[
                            SemanticSurface(type="GroundSurface"),
                            SemanticSurface(type="WallSurface"),
                        ],
                        semantic_values=[0, 1],
                    )
                ],
            )
        )
        footprint = to_feature_collection(mixed, strategy="footprint")
        assert len(footprint) == 1
        assert footprint.geometry.iloc[0].area == 1.0
