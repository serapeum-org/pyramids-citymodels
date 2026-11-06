"""Tests for 3D-Tiles tileset building (feature #15 — tiling / packaging).

`CityModel.to_3dtiles(dir)` writes a single-tile tileset: a `tileset.json` (z-up, a `box` bounding
volume) plus a batched `content.b3dm` (one batch per city object, with a batch table of each
object's id/type/attributes). The output round-trips through this package's own 3D-Tiles reader back
to one footprint per batch, so the writer and reader agree on the b3dm/glb container.
"""

import json
import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import looks_like_3dtiles, parse_3dtiles
from pyramids_citymodels.writers.threedtiles import build_b3dm, build_tileset


def _square(x: float, y: float) -> list[list[tuple[float, float, float]]]:
    """A closed unit-ish square surface at ``(x, y)`` on the z=0 plane."""
    return [
        [
            (x, y, 0.0),
            (x + 10.0, y, 0.0),
            (x + 10.0, y + 10.0, 0.0),
            (x, y + 10.0, 0.0),
            (x, y, 0.0),
        ]
    ]


def _two_buildings() -> CityModel:
    return CityModel(
        objects={
            "houseA": CityObject(
                id="houseA",
                type="Building",
                attributes={"height": 12},
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[_square(1000.0, 2000.0)],
                    )
                ],
            ),
            "houseB": CityObject(
                id="houseB",
                type="Building",
                attributes={"height": 7},
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[_square(1100.0, 2000.0)],
                    )
                ],
            ),
        }
    )


def _write(model: CityModel) -> Path:
    directory = Path(tempfile.mkdtemp())
    model.to_3dtiles(directory)
    return directory


def test_writes_tileset_and_b3dm_files() -> None:
    directory = _write(_two_buildings())
    assert (directory / "tileset.json").exists()
    assert (directory / "content.b3dm").exists()


def test_tileset_is_zup_and_points_at_content() -> None:
    directory = _write(_two_buildings())
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    assert tileset["asset"]["gltfUpAxis"] == "Z"
    assert tileset["root"]["content"]["uri"] == "content.b3dm"
    assert len(tileset["root"]["boundingVolume"]["box"]) == 12
    assert tileset["geometricError"] > 0


def test_b3dm_has_the_b3dm_magic_and_is_recognised() -> None:
    directory = _write(_two_buildings())
    data = (directory / "content.b3dm").read_bytes()
    assert data[:4] == b"b3dm"
    assert looks_like_3dtiles(directory / "content.b3dm")
    assert looks_like_3dtiles(directory / "tileset.json")


def test_round_trips_back_to_one_footprint_per_building() -> None:
    directory = _write(_two_buildings())
    back = parse_3dtiles(directory / "tileset.json")
    assert sorted(back.objects) == ["houseA", "houseB"]
    assert back.source_format == "3DTiles"


def test_batch_table_attributes_round_trip() -> None:
    directory = _write(_two_buildings())
    back = parse_3dtiles(directory / "tileset.json")
    assert back.objects["houseA"].attributes["height"] == 12
    assert back.objects["houseB"].attributes["height"] == 7
    assert back.objects["houseA"].type == "Building"


def test_coordinates_are_placed_in_world_frame() -> None:
    # z-up + node translation means the read-back footprint sits at the original coordinates, not
    # rotated (which a mis-declared up-axis would do).
    back = parse_3dtiles(_write(_two_buildings()) / "tileset.json")
    xs = [
        point[0]
        for surface in back.objects["houseA"].geometries[0].boundaries
        for ring in surface
        for point in ring
    ]
    assert min(xs) >= 1000.0 and max(xs) <= 1010.0


def test_empty_model_writes_a_valid_contentless_tileset() -> None:
    directory = _write(CityModel())
    assert (directory / "tileset.json").exists()
    assert not (directory / "content.b3dm").exists()
    tileset = json.loads((directory / "tileset.json").read_text(encoding="utf-8"))
    assert "content" not in tileset["root"]
    assert parse_3dtiles(directory / "tileset.json").objects == {}


def test_build_b3dm_returns_none_for_a_geometryless_model() -> None:
    assert (
        build_b3dm(CityModel(objects={"x": CityObject(id="x", type="Building")}))
        is None
    )


def test_build_tileset_has_no_content_without_geometry() -> None:
    tileset = build_tileset(CityModel(), None)
    assert "content" not in tileset["root"]
    assert tileset["geometricError"] == 0.0


def test_attribute_named_id_or_type_does_not_clobber_identity_columns() -> None:
    # An attribute literally called "id"/"type" must not overwrite the batch table's identity
    # columns, which carry the object's own id and type.
    model = CityModel(
        objects={
            "real": CityObject(
                id="real",
                type="Building",
                attributes={"id": "attr-id", "type": "attr-type", "height": 3},
                geometries=[
                    Geometry(
                        type="MultiSurface", lod="2", boundaries=[_square(0.0, 0.0)]
                    )
                ],
            )
        }
    )
    tileset = build_tileset(model, "content.b3dm")
    assert tileset["root"]["content"]["uri"] == "content.b3dm"
    back = parse_3dtiles(_write(model) / "tileset.json")
    # The object keeps its own id (the batch table's id column wins over the attribute column).
    assert "real" in back.objects
    assert back.objects["real"].attributes["height"] == 3


def test_degenerate_surfaces_are_skipped_but_valid_ones_kept() -> None:
    # An empty surface and a two-vertex (sub-areal) surface contribute no geometry; a valid surface
    # on the same object still makes it a batch.
    geometry = Geometry(
        type="MultiSurface",
        lod="2",
        # surfaces: an empty one, a two-vertex (sub-areal) one, then a valid square surface
        # (_square returns one surface = a list of one ring).
        boundaries=[[], [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]], _square(0.0, 0.0)],
    )
    model = CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])}
    )
    assert build_b3dm(model) is not None  # the valid surface still produces a tile
    back = parse_3dtiles(_write(model) / "tileset.json")
    assert "b" in back.objects


def test_object_with_only_degenerate_surfaces_contributes_no_tile() -> None:
    geometry = Geometry(
        type="MultiSurface", lod="2", boundaries=[[], [[(0.0, 0.0, 0.0)]]]
    )
    model = CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])}
    )
    assert build_b3dm(model) is None
