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
            assert all(abs(component) % 2.0 == 0.0 for component in point), (
                f"{point} not on the grid"
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

    def _write_lod(self, ratio: float = 0.1) -> tuple[dict, Path]:
        out = Path(tempfile.mkdtemp())
        tileset = _grid_model(8).to_3dtiles(out, max_objects_per_tile=2, simplify=ratio)
        return tileset, out

    def test_interior_tiles_carry_simplified_content_and_replace_refine(self):
        """With ``simplify`` the root is a REPLACE interior tile that carries content.

        Test scenario:
            The subdivided root refines REPLACE and has its own (coarse) content, unlike the default.
        """
        tileset, _out = self._write_lod()
        root = tileset["root"]
        assert root["refine"] == "REPLACE", "an LOD tree refines REPLACE"
        assert "content" in root, (
            "an interior LOD tile must carry a coarse content file"
        )
        assert "children" in root, "the root is subdivided"

    def test_default_is_add_with_empty_interior(self):
        """Without ``simplify`` the tree keeps the original ADD refinement and empty interior tiles.

        Test scenario:
            The default multi-tile write refines ADD and the subdivided root has no content.
        """
        tileset = _grid_model(8).to_3dtiles(
            Path(tempfile.mkdtemp()), max_objects_per_tile=2
        )
        assert tileset["root"]["refine"] == "ADD", "default refinement is ADD"
        assert "content" not in tileset["root"], "default interior tiles are empty"

    def test_lod_tileset_round_trips_without_duplicates(self):
        """Reading an LOD tileset yields each feature once (the reader skips superseded coarse tiles).

        Test scenario:
            The 8-object LOD tileset reads back as exactly 8 objects, with no duplicate-feature warning.
        """
        _tileset, out = self._write_lod()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error"
            )  # any duplicate-feature warning becomes a failure
            back = parse_3dtiles(str(out / "tileset.json"))
        assert len(back.objects) == 8, f"expected 8 features, got {len(back.objects)}"

    def test_simplify_with_implicit_tiling_raises(self):
        """``simplify`` is unsupported with implicit tiling and raises.

        Test scenario:
            Combining ``simplify`` with ``implicit_tiling=True`` is rejected with a clear message.
        """
        out = Path(tempfile.mkdtemp())
        with pytest.raises(
            ValueError, match="only supported for the explicit multi-tile mode"
        ):
            _grid_model(8).to_3dtiles(out, implicit_tiling=True, simplify=0.1)

    def test_non_positive_simplify_raises(self):
        """A non-positive ``simplify`` ratio is rejected.

        Test scenario:
            ``simplify=0`` raises ``ValueError``.
        """
        out = Path(tempfile.mkdtemp())
        with pytest.raises(ValueError, match="simplify must be a positive ratio"):
            _grid_model(8).to_3dtiles(out, max_objects_per_tile=2, simplify=0)
