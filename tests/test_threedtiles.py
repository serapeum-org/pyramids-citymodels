"""Tests for the 3D Tiles reader and driver (issue #16)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import struct
from pathlib import Path

import geopandas as gpd
import pytest

from pyramids_citymodels.drivers.cityjson import CityJsonDriver
from pyramids_citymodels.drivers.threedtiles import ThreeDTilesDriver
from pyramids_citymodels.errors import UnsupportedGeometryError
from pyramids_citymodels.readers.threedtiles import (
    _parse_glb,
    looks_like_3dtiles,
    parse_3dtiles,
)

DATA = Path(__file__).parent / "data"
TILESET = DATA / "tileset.json"
B3DM = DATA / "building.b3dm"


class TestSniffing:
    """``looks_like_3dtiles`` claims tilesets and b3dm, declines look-alikes, never raises."""

    def test_tileset_json_is_recognised(self) -> None:
        assert looks_like_3dtiles(TILESET) is True

    def test_b3dm_binary_is_recognised(self) -> None:
        assert looks_like_3dtiles(B3DM) is True

    def test_cityjson_sharing_json_suffix_is_declined(self) -> None:
        assert (
            looks_like_3dtiles(
                '{"type": "CityJSON", "CityObjects": {}, "vertices": []}'
            )
            is False
        )

    def test_geojson_is_declined(self) -> None:
        assert (
            looks_like_3dtiles('{"type": "FeatureCollection", "features": []}') is False
        )

    @pytest.mark.parametrize("source", ["", "not a tileset", 123, None, b"\x00\x01"])
    def test_hostile_inputs_never_raise(self, source: object) -> None:
        assert looks_like_3dtiles(source) is False


class TestGlbContainer:
    """The glTF binary container splits into its JSON and BIN chunks."""

    def test_glb_parses_into_json_and_binary(self) -> None:
        data = B3DM.read_bytes()
        offset = struct.calcsize("<4sIIIIII")
        magic, _v, _len, ftj, ftb, btj, btb = struct.unpack_from("<4sIIIIII", data, 0)
        assert magic == b"b3dm"
        offset += ftj + ftb + btj + btb
        gltf, binary = _parse_glb(data[offset:])
        assert gltf["asset"]["version"] == "2.0"
        assert len(binary) > 0

    def test_non_glb_payload_raises(self) -> None:
        with pytest.raises(UnsupportedGeometryError, match="glTF binary"):
            _parse_glb(b"not a glb payload at all")


class TestParseTileset:
    """A tileset walks to its b3dm content and yields one footprint feature per batch id."""

    def test_two_features_with_batch_attributes(self) -> None:
        model = parse_3dtiles(TILESET)
        assert sorted(model.objects) == ["b0", "b1"]
        assert model.source_format == "3DTiles"
        # A 3D Tiles tileset is typically ECEF; this reader does not project, so crs stays None.
        assert model.crs is None
        assert model.objects["b1"].attributes["height"] == 20.0

    def test_footprints_match_the_known_square_extents(self) -> None:
        model = parse_3dtiles(TILESET)
        ring = model.objects["b0"].geometries[0].boundaries[0][0]
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        assert (min(xs), min(ys), max(xs), max(ys)) == (0.0, 0.0, 2.0, 2.0)

    def test_b3dm_read_directly(self) -> None:
        model = parse_3dtiles(B3DM)
        assert sorted(model.objects) == ["b0", "b1"]

    def test_in_memory_tileset_without_base_dir_is_empty(self) -> None:
        """A dict/string tileset has no directory, so its content.uri cannot be followed."""
        model = parse_3dtiles(
            '{"asset": {"version": "1.0"}, "geometricError": 0, '
            '"root": {"content": {"uri": "building.b3dm"}}}'
        )
        assert model.objects == {}

    def test_unreadable_source_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="cannot read 3D Tiles"):
            parse_3dtiles(12345)


class TestDriver:
    """The driver dispatches a tileset read and conforms to the registry contract."""

    def test_read_returns_feature_collection_with_footprints(self) -> None:
        frame = ThreeDTilesDriver().read(TILESET)
        assert isinstance(frame, gpd.GeoDataFrame)
        assert len(frame) == 2
        assert set(frame["city_object_id"]) == {"b0", "b1"}
        assert (frame.geometry.area == 4.0).all()  # each 2x2 square footprint

    def test_batch_table_columns_survive_onto_the_frame(self) -> None:
        frame = ThreeDTilesDriver().read(TILESET)
        assert set(frame["height"]) == {10.0, 20.0}

    def test_can_open_claims_tileset_but_not_cityjson(self) -> None:
        assert ThreeDTilesDriver.can_open(TILESET) is True
        assert CityJsonDriver.can_open(TILESET) is False

    def test_can_open_declines_cityjson_that_cityjson_claims(self) -> None:
        cityjson = DATA / "minimal.city.json"
        assert ThreeDTilesDriver.can_open(cityjson) is False
        assert CityJsonDriver.can_open(cityjson) is True


class TestUnsupported:
    """Compressed meshes and point-cloud tiles are rejected or routed out, never mis-read."""

    def test_draco_compressed_gltf_is_rejected(self, tmp_path: Path) -> None:
        tile = _b3dm_with_compressed_gltf()
        path = tmp_path / "draco.b3dm"
        path.write_bytes(tile)
        with pytest.raises(UnsupportedGeometryError, match="mesh-compression"):
            parse_3dtiles(path)

    def test_pnts_tile_is_skipped_with_a_warning(self, tmp_path: Path) -> None:
        tileset = tmp_path / "tileset.json"
        (tmp_path / "points.pnts").write_bytes(b"pnts" + b"\x00" * 24)
        tileset.write_text(
            '{"asset": {"version": "1.0"}, "geometricError": 0, '
            '"root": {"content": {"uri": "points.pnts"}}}',
            encoding="utf-8",
        )
        with pytest.warns(UserWarning, match="pnts"):
            model = parse_3dtiles(tileset)
        assert model.objects == {}


def _b3dm_with_compressed_gltf() -> bytes:
    """A b3dm whose glTF declares a required Draco extension (no real mesh needed)."""
    import json

    gltf = {
        "asset": {"version": "2.0"},
        "extensionsRequired": ["KHR_draco_mesh_compression"],
    }
    json_chunk = json.dumps(gltf).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    glb = (
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_chunk))
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
    )
    header = struct.pack("<4sIIIIII", b"b3dm", 1, 28 + len(glb), 0, 0, 0, 0)
    return header + glb
