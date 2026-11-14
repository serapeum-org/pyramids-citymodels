"""Tests for writing Draco-compressed 3D Tiles glTF content (issue #82).

`to_3dtiles(content="gltf", compression="draco")` compresses each tile's mesh with
`KHR_draco_mesh_compression` (via the optional `[draco]` extra). Positions are quantized while the
`_FEATURE_ID_0` feature id rides as a lossless generic attribute, so the tileset round-trips through
this package's own reader to the same features and footprints as the uncompressed write — in every
tiling mode and with `simplify` — while the geometry is meaningfully smaller. The whole module needs
`DracoPy`, so it is marked `draco` and deselected on a bare install.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.errors import UnsupportedGeometryError
from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import parse_3dtiles

pytestmark = pytest.mark.draco


def _square(
    x: float, y: float, z: float = 0.0
) -> list[list[tuple[float, float, float]]]:
    """A closed 4-unit square surface at ``(x, y)`` on the ``z`` plane."""
    ring = [(x, y, z), (x + 4, y, z), (x + 4, y + 4, z), (x, y + 4, z)]
    return [ring]


def _model(count: int = 2, surfaces: int = 1) -> CityModel:
    """``count`` buildings on a diagonal, each a stack of ``surfaces`` squares with distinctive attrs."""
    return CityModel(
        objects={
            f"o{i}": CityObject(
                id=f"o{i}",
                type="Building" if i % 2 == 0 else "Bridge",
                attributes={"height": 3.5 + i, "floors": i},
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            _square(i * 50.0, i * 50.0, 3.0 * s)
                            for s in range(surfaces)
                        ],
                    )
                ],
            )
            for i in range(count)
        }
    )


def _footprint(obj: CityObject) -> tuple[float, float, float, float]:
    """The rounded (minx, miny, maxx, maxy) extent of a read-back object's surfaces."""
    xs = [
        p[0]
        for g in obj.geometries
        for surf in g.boundaries
        for ring in surf
        for p in ring
    ]
    ys = [
        p[1]
        for g in obj.geometries
        for surf in g.boundaries
        for ring in surf
        for p in ring
    ]
    return (round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2))


def _footprints(model: CityModel) -> list[tuple[float, float, float, float]]:
    """Every object's footprint extent, sorted — order-independent comparison across a round-trip."""
    return sorted(_footprint(obj) for obj in model.objects.values())


def _glb_document(glb: bytes) -> dict:
    """Parse the JSON chunk out of a ``.glb`` for structural assertions."""
    magic, _version, length = struct.unpack_from("<III", glb, 0)
    assert magic == 0x46546C67, "glb must start with the glTF magic"
    json_length, json_type = struct.unpack_from("<II", glb, 12)
    assert json_type == 0x4E4F534A, "the first chunk must be JSON"
    return json.loads(glb[20 : 20 + json_length])


def _write(model: CityModel, **kwargs) -> Path:
    """Write ``model`` as a tileset into a fresh temp dir and return the directory."""
    out = Path(tempfile.mkdtemp())
    model.to_3dtiles(out, **kwargs)
    return out


class TestDracoWriteRoundTrip:
    """A Draco-compressed tileset reads back to the same features as its uncompressed twin."""

    def test_single_tile_matches_uncompressed(self) -> None:
        """A single Draco tile recovers the same object ids, types, attributes and footprints.

        Test scenario:
            The Draco and uncompressed single-tile writes of one model read back to identical
            feature sets, proving POSITION and the _FEATURE_ID_0 feature id survive compression.
        """
        model = _model(2)
        draco = parse_3dtiles(
            _write(model, content="gltf", compression="draco") / "tileset.json"
        )
        plain = parse_3dtiles(_write(model, content="gltf") / "tileset.json")
        assert set(draco.objects) == set(plain.objects), (
            "same object ids as uncompressed"
        )
        assert _footprints(draco) == _footprints(plain), (
            "same footprints as uncompressed"
        )
        assert {o.type for o in draco.objects.values()} == {"Building", "Bridge"}, (
            "types kept"
        )

    def test_feature_ids_stay_paired_across_many_features(self) -> None:
        """Each of 40 features keeps its own footprint — feature ids are not scrambled by Draco.

        Test scenario:
            With 40 well-separated buildings in one tile, every object reads back at (near) its own
            input footprint — within position-quantization tolerance, far below the 50-unit spacing —
            which can only hold if each vertex kept its correct feature id. Uses an id-keyed,
            tolerant comparison because Draco quantizes positions (but not the feature id).
        """
        model = _model(40)
        back = parse_3dtiles(
            _write(model, content="gltf", compression="draco") / "tileset.json"
        )
        assert len(back.objects) == 40, "all 40 features recovered"
        for object_id, expected in model.objects.items():
            drift = max(
                abs(a - b)
                for a, b in zip(
                    _footprint(back.objects[object_id]), _footprint(expected)
                )
            )
            assert drift < 0.3, (
                f"{object_id} drifted {drift} — far below the 50-unit feature spacing"
            )

    @pytest.mark.parametrize(
        "mode",
        [
            {"max_objects_per_tile": 1},
            {"implicit_tiling": True, "max_objects_per_tile": 1},
            {"max_objects_per_tile": 1, "simplify": 0.02},
            {"metadata": "typed"},
        ],
        ids=["explicit-multitile", "implicit", "simplify-lod", "typed-metadata"],
    )
    def test_draco_composes_with_tiling_and_options(self, mode: dict) -> None:
        """Draco composes with every tiling mode, per-tile LOD and the typed-metadata shape.

        Args:
            mode: The extra ``to_3dtiles`` keywords to combine with ``compression="draco"``.

        Test scenario:
            Each mode's Draco tileset recovers all four objects with the same footprints as the
            uncompressed write of the same model and mode.
        """
        model = _model(4)
        draco = parse_3dtiles(
            _write(model, content="gltf", compression="draco", **mode) / "tileset.json"
        )
        plain = parse_3dtiles(_write(model, content="gltf", **mode) / "tileset.json")
        assert set(draco.objects) == set(plain.objects), (
            "same object ids as uncompressed"
        )
        assert _footprints(draco) == _footprints(plain), (
            "same footprints as uncompressed"
        )

    def test_attributes_round_trip_exactly(self) -> None:
        """Every object's integer ``floors`` attribute round-trips exactly through a Draco tile.

        Test scenario:
            Reading back a 6-feature Draco tileset yields the input ``floors`` values 0..5 unchanged,
            confirming the metadata property table is unaffected by geometry compression.
        """
        back = parse_3dtiles(
            _write(_model(6), content="gltf", compression="draco") / "tileset.json"
        )
        floors = sorted(obj.attributes["floors"] for obj in back.objects.values())
        assert floors == [0, 1, 2, 3, 4, 5], "integer attributes survive exactly"


class TestDracoCompression:
    """Draco shrinks real geometry while leaving the uncompressed path untouched."""

    def test_draco_shrinks_a_dense_mesh(self) -> None:
        """The Draco ``.glb`` is smaller than the uncompressed one for a multi-surface model.

        Test scenario:
            A 60-building model with three surfaces each compresses to well under the uncompressed
            glb size, demonstrating real mesh compression (not just a valid but larger file).
        """
        model = _model(60, surfaces=3)
        draco = (
            _write(model, content="gltf", compression="draco") / "content.glb"
        ).read_bytes()
        plain = (_write(model, content="gltf") / "content.glb").read_bytes()
        assert len(draco) < len(plain), (
            f"draco {len(draco)}B should be < plain {len(plain)}B"
        )

    def test_uncompressed_output_is_byte_identical(self) -> None:
        """``compression=None`` (default) produces the exact bytes of omitting the argument.

        Test scenario:
            Passing ``compression=None`` explicitly and not passing it at all yield identical glb
            bytes, proving the new branch leaves the uncompressed path unchanged.
        """
        model = _model(3)
        explicit = (
            _write(model, content="gltf", compression=None) / "content.glb"
        ).read_bytes()
        default = (_write(model, content="gltf") / "content.glb").read_bytes()
        assert explicit == default, (
            "compression=None must not alter the uncompressed output"
        )


class TestDracoGlbStructure:
    """The written glb declares the Draco extension the way the spec (and decoders) expect."""

    def _draco_doc(self) -> dict:
        """The glTF document of a single Draco tile."""
        return _glb_document(
            (
                _write(_model(2), content="gltf", compression="draco") / "content.glb"
            ).read_bytes()
        )

    def test_extension_is_required_and_used(self) -> None:
        """``KHR_draco_mesh_compression`` is in ``extensionsRequired`` and ``extensionsUsed``.

        Test scenario:
            A decoder that ignores the extension cannot show the mesh, so it must be required, and it
            must also appear in the used list alongside the metadata extensions.
        """
        doc = self._draco_doc()
        assert doc["extensionsRequired"] == ["KHR_draco_mesh_compression"], "required"
        assert "KHR_draco_mesh_compression" in doc["extensionsUsed"], "used"

    def test_geometry_accessors_have_no_bufferview(self) -> None:
        """The POSITION / _FEATURE_ID_0 / index accessors drop ``bufferView`` under Draco.

        Test scenario:
            Per the extension spec the compressed stream supplies the data, so the three geometry
            accessors declare only the decompressed shape and carry no bufferView.
        """
        accessors = self._draco_doc()["accessors"]
        assert all("bufferView" not in a for a in accessors[:3]), (
            "no bufferView on geometry accessors"
        )
        assert accessors[0]["type"] == "VEC3", (
            "POSITION accessor still declares its shape"
        )

    def test_draco_bufferview_and_attribute_map(self) -> None:
        """The Draco bufferView carries no target and the extension maps both glTF attributes.

        Test scenario:
            The primitive's extension references a bufferView (the Draco stream, no ARRAY_BUFFER
            target) and maps POSITION and _FEATURE_ID_0 to Draco unique ids the reader resolves.
        """
        doc = self._draco_doc()
        primitive = doc["meshes"][0]["primitives"][0]
        draco_ext = primitive["extensions"]["KHR_draco_mesh_compression"]
        assert "target" not in doc["bufferViews"][draco_ext["bufferView"]], (
            "draco view has no target"
        )
        assert set(draco_ext["attributes"]) == {"POSITION", "_FEATURE_ID_0"}, (
            "both attributes mapped"
        )


class TestDracoWriteValidation:
    """Invalid ``compression`` combinations fail fast with a clear ValueError."""

    def test_draco_requires_gltf_content(self) -> None:
        """``compression="draco"`` with ``content="b3dm"`` raises (b3dm-embedded Draco is a follow-up).

        Test scenario:
            Draco inside a b3dm wrapper is out of scope for this feature, so the combination is
            rejected before anything is written.
        """
        out = Path(tempfile.mkdtemp())
        with pytest.raises(ValueError, match="content='gltf'"):
            _model(1).to_3dtiles(out, content="b3dm", compression="draco")

    def test_unknown_compression_raises(self) -> None:
        """An unknown ``compression`` value raises naming the accepted values.

        Test scenario:
            Any value other than None / "draco" is rejected with a message listing what is allowed.
        """
        out = Path(tempfile.mkdtemp())
        with pytest.raises(ValueError, match="compression must be None or 'draco'"):
            _model(1).to_3dtiles(out, content="gltf", compression="gzip")


class TestDracoBackendAbsent:
    """Without the draco extra, requesting Draco output fails with an actionable error."""

    def test_missing_dracopy_raises_install_hint(self, monkeypatch) -> None:
        """Writing Draco content with ``DracoPy`` unimportable raises the "install the draco extra" error.

        Test scenario:
            ``DracoPy`` is masked in ``sys.modules`` so the lazy import fails; requesting Draco output
            then raises ``UnsupportedGeometryError`` naming the extra, rather than an ImportError.
        """
        monkeypatch.setitem(
            sys.modules, "DracoPy", None
        )  # force ImportError on `import DracoPy`
        out = Path(tempfile.mkdtemp())
        with pytest.raises(
            UnsupportedGeometryError, match=r"pyramids-citymodels\[draco\]"
        ):
            _model(1).to_3dtiles(out, content="gltf", compression="draco")
