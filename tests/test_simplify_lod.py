"""Tests for mesh simplification and per-tile 3D Tiles LOD (issue #73).

``CityModel.simplify(cell_size)`` is a stdlib-only vertex-clustering decimation: coordinates snap to a
grid so nearby vertices merge and collapsed surfaces drop out. ``to_3dtiles(..., simplify=ratio)`` uses
it to build a real level-of-detail tree in the explicit multi-tile mode — each interior tile carries a
simplified copy of its subtree and the tree refines ``REPLACE``, so a client (and this package's reader)
shows the coarse interior until a child replaces it. The reader skips a REPLACE tile's own content when
it has children, so an LOD tileset round-trips to the same feature set (no duplicate-at-each-level).
"""

import tempfile
import warnings
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import parse_3dtiles


def _square(x: float, y: float) -> list[list[tuple[float, float, float]]]:
    """A closed 5-unit square surface at ``(x, y)`` on the z=0 plane."""
    ring = [(x, y, 0.0), (x + 5, y, 0.0), (x + 5, y + 5, 0.0), (x, y + 5, 0.0)]
    return [ring + [ring[0]]]


def _grid_model(count: int) -> CityModel:
    """``count`` square buildings spread across a grid."""
    return CityModel(
        objects={
            f"o{i}": CityObject(
                id=f"o{i}",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[_square(i * 20.0, (i % 3) * 20.0)],
                    )
                ],
            )
            for i in range(count)
        }
    )


class TestSimplify:
    """Tests for the ``CityModel.simplify`` vertex-clustering transform."""

    def test_vertices_within_a_cell_merge(self):
        """Sub-cell-spaced vertices snap together, so a ring loses the clustered ones.

        Test scenario:
            A ring with three x-values 0 / 0.3 / 0.6 (all in the first unit cell) collapses them to one.
        """
        ring = [
            (0.0, 0.0, 0.0),
            (0.3, 0.0, 0.0),
            (0.6, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
        ]
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(type="MultiSurface", lod="2", boundaries=[[ring]])
                    ],
                )
            }
        )
        simplified = model.simplify(1.0).objects["b"].geometries[0].boundaries[0][0]
        assert simplified == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], (
            f"got {simplified}"
        )

    def test_error_is_bounded_by_half_a_cell(self):
        """Every coordinate moves at most ``cell_size / 2`` per axis.

        Test scenario:
            A fine ring simplified at cell 2.0 keeps every vertex within 1.0 of an original coordinate.
        """
        ring = [(0.0, 0.0, 0.0), (0.9, 0.3, 0.0), (2.1, 1.7, 0.0), (0.0, 2.0, 0.0)]
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(type="MultiSurface", lod="2", boundaries=[[ring]])
                    ],
                )
            }
        )
        simplified = model.simplify(2.0).objects["b"].geometries[0].boundaries[0][0]
        for point in simplified:
            for axis in range(3):
                closest = min(abs(point[axis] - original[axis]) for original in ring)
                assert closest <= 1.0, (
                    f"axis {axis} of {point} moved more than cell/2 (1.0): {closest}"
                )

    def test_is_idempotent_at_the_same_cell_size(self):
        """Re-simplifying at the same cell size changes nothing.

        Test scenario:
            ``simplify(1.0)`` applied twice equals applying it once.
        """
        model = _grid_model(3)
        once = model.simplify(1.0)
        twice = once.simplify(1.0)
        assert (
            once.objects["o0"].geometries[0].boundaries
            == twice.objects["o0"].geometries[0].boundaries
        )

    def test_collapsed_surface_is_dropped_not_left_degenerate(self):
        """A surface that clusters below 3 distinct vertices is dropped, not kept as a degenerate ring (M1).

        Test scenario:
            A thin triangle collapses to a line at cell 1.0; the simplified model has no surface, so a
            non-mesh writer (CityJSON) emits no degenerate polygon.
        """
        ring = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.0, 0.0)]
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(type="MultiSurface", lod="2", boundaries=[[ring]])
                    ],
                )
            }
        )
        simplified = model.simplify(1.0)
        assert simplified.objects["b"].geometries[0].boundaries == [], (
            "the collapsed surface must be dropped"
        )
        document = simplified.to_cityjson()
        geometries = document["CityObjects"]["b"]["geometry"]
        assert all(not g.get("boundaries") for g in geometries), (
            "no degenerate polygon may be emitted"
        )

    def test_surface_dropped_keeps_semantic_values_aligned(self):
        """Dropping a collapsed surface drops its semantic value too, keeping the arrays aligned.

        Test scenario:
            Two surfaces, the first collapses; the survivor keeps its own semantic value (not the dropped
            one's), so semantics stay aligned with boundaries.
        """
        thin = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.0, 0.0)]  # collapses
        wide = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0)]  # survives
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[[thin], [wide]],
            semantics=[],
            semantic_values=[0, 1],
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])}
        )
        result = model.simplify(1.0).objects["b"].geometries[0]
        assert len(result.boundaries) == 1, "only the surviving surface remains"
        assert result.semantic_values == [1], (
            "the survivor's semantic value is kept, aligned"
        )

    def test_non_positive_cell_size_raises(self):
        """A non-positive cell size is rejected.

        Test scenario:
            ``simplify(0)`` raises ``ValueError``.
        """
        model = _grid_model(1)
        with pytest.raises(ValueError, match="cell_size must be positive"):
            model.simplify(0)

    def test_inputs_are_not_mutated(self):
        """``simplify`` returns a new model and leaves the input untouched.

        Test scenario:
            The original ring is unchanged after simplifying.
        """
        ring = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (1.0, 1.0, 0.0)]
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(type="MultiSurface", lod="2", boundaries=[[ring]])
                    ],
                )
            }
        )
        model.simplify(1.0)
        assert model.objects["b"].geometries[0].boundaries[0][0] == ring, (
            "input must be unchanged"
        )


class TestTilesetLOD:
    """Tests for ``to_3dtiles(simplify=...)`` per-level LOD."""

    def _write_lod(self, out: Path, ratio: float = 0.1) -> tuple[dict, Path]:
        tileset = _grid_model(8).to_3dtiles(out, max_objects_per_tile=2, simplify=ratio)
        return tileset, out

    def test_interior_tiles_carry_simplified_content_and_replace_refine(self, tmp_path):
        """With ``simplify`` the root is a REPLACE interior tile that carries content.

        Test scenario:
            The subdivided root refines REPLACE and has its own (coarse) content, unlike the default.
        """
        tileset, _out = self._write_lod(tmp_path)
        root = tileset["root"]
        assert root["refine"] == "REPLACE", "an LOD tree refines REPLACE"
        assert "content" in root, (
            "an interior LOD tile must carry a coarse content file"
        )
        assert "children" in root, "the root is subdivided"

    def test_default_is_add_with_empty_interior(self, tmp_path):
        """Without ``simplify`` the tree keeps the original ADD refinement and empty interior tiles.

        Test scenario:
            The default multi-tile write refines ADD and the subdivided root has no content.
        """
        tileset = _grid_model(8).to_3dtiles(tmp_path, max_objects_per_tile=2)
        assert tileset["root"]["refine"] == "ADD", "default refinement is ADD"
        assert "content" not in tileset["root"], "default interior tiles are empty"

    def test_lod_tileset_round_trips_without_duplicates(self, tmp_path):
        """Reading an LOD tileset yields each feature once (the reader skips superseded coarse tiles).

        Test scenario:
            The 8-object LOD tileset reads back as exactly 8 objects, with no duplicate-feature warning.
        """
        _tileset, out = self._write_lod(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error"
            )  # any duplicate-feature warning becomes a failure
            back = parse_3dtiles(str(out / "tileset.json"))
        assert len(back.objects) == 8, f"expected 8 features, got {len(back.objects)}"

    def test_simplify_with_implicit_tiling_raises(self, tmp_path):
        """``simplify`` is unsupported with implicit tiling and raises.

        Test scenario:
            Combining ``simplify`` with ``implicit_tiling=True`` is rejected with a clear message.
        """
        model = _grid_model(8)
        with pytest.raises(
            ValueError, match="only supported for the explicit multi-tile mode"
        ):
            model.to_3dtiles(tmp_path, implicit_tiling=True, simplify=0.1)

    def test_coarse_tile_drops_zigzag_collapsing_to_two_vertices(self):
        """A ring that snaps to an alternating ``A,B,A,B`` (2 distinct) is dropped from a coarse tile (R2-L1).

        Test scenario:
            ``_simplify_rings`` uses the same distinct-vertex drop criterion as ``CityModel.simplify``, so a
            zig-zag that keeps >= 3 *vertices* but only 2 *distinct* ones does not survive as a sliver.
        """
        from pyramids_citymodels.writers.threedtiles import _simplify_rings

        obj = CityObject(id="z", type="Building", geometries=[])
        zigzag = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.9, 0.0, 0.0),
        ]  # snaps to 0,1,0,1 -> 2 distinct
        assert _simplify_rings([(obj, [zigzag])], 1.0) == [], (
            "a 2-distinct-vertex zig-zag must be dropped, not kept as a sliver"
        )

    def test_simplify_without_subdivision_warns(self, tmp_path):
        """``simplify`` on a model that writes a single tile warns that it had no effect (L3).

        Test scenario:
            With no ``max_objects_per_tile`` the model is one tile, so there is no interior level to
            coarsen — ``simplify`` is a silent no-op unless the writer warns.
        """
        model = _grid_model(4)
        with pytest.warns(UserWarning, match="simplify had no effect"):
            model.to_3dtiles(tmp_path, simplify=0.1)

    def test_non_positive_simplify_raises(self, tmp_path):
        """A non-positive ``simplify`` ratio is rejected.

        Test scenario:
            ``simplify=0`` raises ``ValueError``.
        """
        model = _grid_model(8)
        with pytest.raises(ValueError, match="simplify must be a positive ratio"):
            model.to_3dtiles(tmp_path, max_objects_per_tile=2, simplify=0)
