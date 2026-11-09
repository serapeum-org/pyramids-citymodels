"""Tests for IR geometry processing & cleanup (issue #52).

`translate`, `quantize`, `clean_rings` and `triangulate` each return a new `CityModel` (inputs
untouched). The coordinate transforms preserve semantics/appearance; `triangulate` keeps per-triangle
semantics but drops holes and per-geometry appearance.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry, SemanticSurface


def _model(boundaries: object, **geom_kw: object) -> CityModel:
    return CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface", lod="2", boundaries=boundaries, **geom_kw
                    )  # type: ignore[arg-type]
                ],
            )
        }
    )


def _tri() -> object:
    return [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]]


def test_translate_shifts_every_coordinate_and_leaves_input_untouched() -> None:
    model = _model(_tri())
    out = model.translate((10.0, 20.0, 30.0))
    assert out.objects["b"].geometries[0].boundaries[0][0][0] == (10.0, 20.0, 30.0)
    assert model.objects["b"].geometries[0].boundaries[0][0][0] == (
        0.0,
        0.0,
        0.0,
    )  # input intact


def test_translate_pads_missing_axes_with_zero() -> None:
    out = _model([[[(1.0, 2.0)]]]).translate((5.0,))  # 2-D point, 1-axis offset
    assert out.objects["b"].geometries[0].boundaries[0][0][0] == (6.0, 2.0)


def test_quantize_rounds_coordinates() -> None:
    model = _model([[[(0.123456, 1.987654, 2.5), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]])
    rounded = model.quantize(2).objects["b"].geometries[0].boundaries[0][0][0]
    assert rounded == (0.12, 1.99, 2.5)


def test_clean_rings_removes_consecutive_duplicate_vertices() -> None:
    ring = [[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    cleaned = _model([ring]).clean_rings().objects["b"].geometries[0].boundaries[0][0]
    assert cleaned == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]


def test_triangulate_fans_a_quad_into_two_triangles() -> None:
    quad = [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]]]
    out = _model(quad).triangulate().objects["b"].geometries[0]
    assert out.type == "MultiSurface"
    assert len(out.boundaries) == 2  # two triangles
    assert all(len(surface[0]) == 3 for surface in out.boundaries)  # each a triangle


def test_triangulate_preserves_per_triangle_semantics() -> None:
    quad = [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]]]
    model = _model(
        quad,
        semantics=[SemanticSurface(type="RoofSurface")],
        semantic_values=[0],
    )
    out = model.triangulate().objects["b"].geometries[0]
    assert out.semantic_values == [0, 0]  # both triangles keep the roof semantic
    assert out.semantics[0].type == "RoofSurface"


def test_triangulate_drops_appearance() -> None:
    model = _model(_tri(), appearance={"material": {"t": {"values": [0]}}})
    assert model.triangulate().objects["b"].geometries[0].appearance is None


def test_triangulate_handles_a_closed_ring() -> None:
    # A closed quad (first == last) opens to 4 distinct vertices → two triangles.
    quad = [
        [
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
            ]
        ]
    ]
    out = _model(quad).triangulate().objects["b"].geometries[0]
    assert len(out.boundaries) == 2


def test_triangulate_skips_empty_and_subareal_surfaces() -> None:
    # An empty surface and a 2-vertex (sub-areal) ring contribute no triangles; the valid one does.
    boundaries = [[], [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]], *_tri()]
    out = _model(boundaries).triangulate().objects["b"].geometries[0]
    assert len(out.boundaries) == 1  # only the valid triangle survives


def test_transform_output_does_not_alias_input_semantics_or_appearance() -> None:
    # A transformed model shares no mutable structure with the input (like CityModel.merge), so a
    # later in-place edit of the output cannot leak back into the source.
    model = _model(
        _tri(),
        semantics=[SemanticSurface(type="RoofSurface")],
        semantic_values=[0],
        appearance={"material": {"t": {"values": [0]}}},
    )
    out = model.translate((1.0, 0.0, 0.0))
    out_geom = out.objects["b"].geometries[0]
    in_geom = model.objects["b"].geometries[0]
    assert out_geom.semantics is not in_geom.semantics
    assert out_geom.appearance is not in_geom.appearance
    out_geom.semantics[0].type = "WallSurface"  # mutate the copy
    assert in_geom.semantics[0].type == "RoofSurface"  # source intact


def test_coordinate_mappers_leave_a_non_coordinate_node_untouched() -> None:
    # Defensive: a stray non-list, non-coordinate node (e.g. None) in the boundaries is passed through
    # unchanged by the recursive mappers rather than crashing.
    boundaries = [[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], None]]
    translated = (
        _model(boundaries).translate((1.0, 0.0, 0.0)).objects["b"].geometries[0]
    )
    assert translated.boundaries[0][0][0] == (1.0, 0.0, 0.0)
    assert translated.boundaries[0][1] is None  # the stray node survives
    cleaned = _model(boundaries).clean_rings().objects["b"].geometries[0]
    assert cleaned.boundaries[0][1] is None
