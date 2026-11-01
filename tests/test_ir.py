"""Tests for the shared intermediate representation (issue #5)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.errors import CityModelError
from pyramids_citymodels.ir import (
    CityModel,
    CityObject,
    Geometry,
    VertexPool,
    crs_from_identifier,
    iter_surfaces,
    normalize_lod,
)
from pyramids_citymodels.readers.cityjson import parse_cityjson


def test_crs_from_identifier_handles_urn_url_and_plain() -> None:
    assert crs_from_identifier("urn:ogc:def:crs:EPSG::7415") == "EPSG:7415"
    assert (
        crs_from_identifier("https://www.opengis.net/def/crs/EPSG/0/25832")
        == "EPSG:25832"
    )
    assert crs_from_identifier("EPSG:4326") == "EPSG:4326"
    assert crs_from_identifier(None) is None
    assert crs_from_identifier("not a crs") is None


def test_normalize_lod_makes_numeric_lods_comparable() -> None:
    assert normalize_lod("2") == normalize_lod(2) == "2.0"
    assert normalize_lod("2.2") == "2.2"
    assert normalize_lod(None) is None


def test_iter_surfaces_multisurface_pairs_each_surface_with_its_value() -> None:
    # A MultiSurface: boundaries is a list of surfaces, values a flat list.
    surface_a = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    surface_b = [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
    pairs = list(iter_surfaces([surface_a, surface_b], [0, 1]))
    assert [value for _surface, value in pairs] == [0, 1]
    assert pairs[0][0] is surface_a


def test_iter_surfaces_solid_descends_shells() -> None:
    # A Solid: boundaries is a list of shells (each a list of surfaces); values nests one level.
    surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    solid = [[surface, surface]]  # one shell, two surfaces
    pairs = list(iter_surfaces(solid, [[7, 9]]))
    assert [value for _surface, value in pairs] == [7, 9]


def test_city_object_parent_is_first_parent_or_none() -> None:
    assert CityObject(id="a", type="Building").parent is None
    assert CityObject(id="b", type="BuildingPart", parents=["a"]).parent == "a"


# --- round-trip rebuild of the CityJSON vertex pool (issue #24) -------------------------------


def _model_with(*boundaries: list) -> tuple[CityModel, list[Geometry]]:
    """A one-building model whose single object carries one geometry per ``boundaries`` entry."""
    geometries = [
        Geometry(type="MultiSurface", lod="2", boundaries=b) for b in boundaries
    ]
    model = CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=geometries)}
    )
    return model, geometries


def test_vertex_pool_reindexes_to_geometry_identical_to_source() -> None:
    # The issue's acceptance criterion: a pool rebuilt from the IR re-indexes back to the exact
    # same geometry (lossless, no transform).
    surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    model, (geometry,) = _model_with([surface])
    pool = model.vertex_pool()
    assert pool.transform is None
    indexed = geometry.index_into(pool)
    assert indexed == [[[0, 1, 2]]]
    assert pool.dereference(indexed) == geometry.boundaries


def test_vertex_pool_deduplicates_shared_vertices() -> None:
    # Two surfaces sharing an edge pool to four vertices, not six, and both still round-trip.
    surface_a = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    surface_b = [[(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]]
    model, (geometry,) = _model_with([surface_a, surface_b])
    pool = model.vertex_pool()
    indexed = geometry.index_into(pool)
    assert len(pool.vertices) == 4
    assert indexed == [[[0, 1, 2]], [[1, 2, 3]]]
    assert pool.dereference(indexed) == geometry.boundaries


def test_vertex_pool_shared_across_geometries() -> None:
    # One pool indexed by several geometries is the single document-wide vertices array a writer
    # emits; a coordinate common to two geometries is stored once.
    surface_a = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    surface_b = [[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0)]]
    model, (first, second) = _model_with([surface_a], [surface_b])
    pool = model.vertex_pool()
    assert first.index_into(pool) == [[[0, 1, 2]]]
    assert second.index_into(pool) == [[[0, 3, 4]]]  # reuses vertex 0, appends the rest
    assert len(pool.vertices) == 5


def test_vertex_pool_quantizes_with_a_transform_and_round_trips_within_tolerance() -> (
    None
):
    surface = [[(1234.567, 10.0, 0.0), (1234.568, 11.0, 5.0)]]
    model, (geometry,) = _model_with([surface])
    pool = model.vertex_pool(precision=3)
    assert pool.transform == {
        "scale": [0.001, 0.001, 0.001],
        "translate": [1234.567, 10.0, 0.0],
    }
    assert all(isinstance(component, int) for v in pool.vertices for component in v)
    recovered = pool.dereference(geometry.index_into(pool))
    for got_ring, want_ring in zip(recovered[0], surface):
        for got, want in zip(got_ring, want_ring):
            for got_c, want_c in zip(got, want):
                assert abs(got_c - want_c) <= 5e-4


def test_coordinates_walks_every_geometry_in_document_order() -> None:
    surface_a = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]
    surface_b = [[(2.0, 0.0, 0.0)]]
    model, _ = _model_with([surface_a], [surface_b])
    assert list(model.coordinates()) == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]


def test_empty_model_rebuilds_an_empty_pool() -> None:
    assert CityModel().vertex_pool().vertices == []
    # With no coordinates there is nothing to quantize, so even a precision request yields no
    # transform (rather than a degenerate, invalid {"scale": [], "translate": []}).
    assert CityModel().vertex_pool(precision=3).transform is None
    assert CityModel().vertex_pool(precision=3).vertices == []


def test_vertex_pool_rebuilt_from_a_parsed_cityjson_document_round_trips() -> None:
    # End to end: parse real CityJSON (vertices+transform dereferenced away on read), rebuild the
    # pool, and confirm every geometry re-indexes to coordinates identical to the parsed ones.
    document = {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": [0.001, 0.001, 0.001], "translate": [0.0, 0.0, 0.0]},
        "CityObjects": {
            "b": {
                "type": "Building",
                "geometry": [
                    {
                        "type": "MultiSurface",
                        "lod": "2",
                        "boundaries": [[[0, 1, 2]], [[1, 2, 0]]],
                    }
                ],
            }
        },
        "vertices": [[0, 0, 0], [1000, 0, 0], [1000, 1000, 0]],
    }
    model = parse_cityjson(document)
    pool = model.vertex_pool()
    for city_object in model.objects.values():
        for geometry in city_object.geometries:
            assert pool.dereference(geometry.index_into(pool)) == geometry.boundaries


def test_vertex_pool_preloaded_with_vertices_dedupes_against_them() -> None:
    pool = VertexPool(vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert pool.index_of((1.0, 0.0, 0.0)) == 1  # existing, not appended
    assert pool.index_of((2.0, 0.0, 0.0)) == 2  # new, appended
    assert len(pool.vertices) == 3


def test_geometry_coordinates_walks_boundaries_to_coordinate_level() -> None:
    # Geometry.coordinates yields each coordinate of a Solid (surfaces two levels down).
    surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]
    geometry = Geometry(type="Solid", lod="2", boundaries=[[surface]])
    assert list(geometry.coordinates()) == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]


def test_geometry_coordinates_skips_non_coordinate_non_list_leaves() -> None:
    # A stray leaf that is neither a coordinate nor a list (defensive guard in the boundary walk)
    # is skipped rather than raising, so a malformed boundaries structure still yields its real
    # coordinates.
    geometry = Geometry(
        type="MultiSurface",
        lod="2",
        boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]], "stray"],
    )
    assert list(geometry.coordinates()) == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]


def test_vertex_pool_quantize_and_dequantize_are_inverse_within_the_step() -> None:
    # quantize/dequantize directly: a real coordinate maps to an integer vertex and back.
    pool = VertexPool(transform={"scale": [0.01, 0.01], "translate": [100.0, 200.0]})
    assert pool.quantize((100.5, 200.25)) == (50, 25)
    assert pool.dequantize((50, 25)) == (100.5, 200.25)


def test_index_of_rejects_mixed_dimension_coordinates() -> None:
    # A 2D vertex pooled after a 3D one cannot share one CityJSON `vertices` array, so index_of
    # raises rather than silently building a non-uniform (non-serializable) pool.
    pool = VertexPool()
    pool.index_of((0.0, 0.0, 0.0))
    with pytest.raises(CityModelError, match="mixed-dimension"):
        pool.index_of((1.0, 0.0))


def test_vertex_pool_rejects_mixed_dimension_model() -> None:
    # Mixed-dimension geometries (reachable from CityGML's per-element srsDimension) are rejected
    # when rebuilding the pool, in both the lossless and the quantized paths.
    three_d = Geometry(type="MultiSurface", lod="2", boundaries=[[[(0.0, 0.0, 0.0)]]])
    two_d = Geometry(type="MultiSurface", lod="2", boundaries=[[[(1.0, 2.0)]]])
    model = CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[three_d, two_d])}
    )
    for precision in (None, 3):
        # Lossless catches it at index_into; quantized catches it earlier, in vertex_pool's scan.
        with pytest.raises(CityModelError, match="mixed-dimension"):
            pool = model.vertex_pool(precision=precision)
            for geometry in model.objects["b"].geometries:
                geometry.index_into(pool)


def test_for_coordinates_rejects_mixed_dimension_stream() -> None:
    # The quantizing scan catches a mixed-dimension stream early, before any indexing.
    with pytest.raises(CityModelError, match="mixed-dimension"):
        VertexPool.for_coordinates([(10.0, 20.0), (5.0, 8.0, 3.0)], precision=2)


def test_for_coordinates_rejects_negative_precision() -> None:
    # precision is decimal places; a negative value would quantize to tens/hundreds, which a caller
    # passing "decimal places" never means, so it is rejected rather than silently accepted.
    with pytest.raises(ValueError, match="precision"):
        VertexPool.for_coordinates([(1234.0, 5678.0)], precision=-1)


def test_index_into_requires_nested_boundaries() -> None:
    # A geometry's boundaries are always a nested list; a bare coordinate is out of contract and is
    # rejected, so index_into never returns a scalar int in violation of its list[Any] annotation.
    geometry = Geometry(type="MultiSurface", lod="2", boundaries=(1.0, 2.0, 3.0))
    with pytest.raises(AssertionError):
        geometry.index_into(VertexPool())
