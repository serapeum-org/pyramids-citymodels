"""Tests for embedding texture image bytes in the mesh exports (issue #74).

With ``embed_textures=True`` a readable local texture image is made part of the export: glTF embeds it
as a base64 ``data:`` URI (self-contained ``.gltf``), while OBJ and COLLADA copy the image file beside
the output and reference it by basename. An image that cannot be read (missing, remote URL, unreadable)
falls back to a plain URI reference — never a crash — and the default (reference-only) is unchanged.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.writers.collada import write_collada
from pyramids_citymodels.writers.gltf import write_gltf
from pyramids_citymodels.writers.obj import write_obj

_PNG = b"\x89PNG\r\n\x1a\n_fake_png_bytes_for_testing_"


def _textured_model(image: str = "wall.png") -> CityModel:
    """A one-surface model textured by texture 0 referencing ``image``."""
    geometry = Geometry(
        type="MultiSurface",
        lod="2",
        boundaries=[[[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]],
        appearance={"texture": {"": {"values": [[[0, 0, 1, 2]]]}}},
    )
    return CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
        appearances={
            "textures": [{"type": "PNG", "image": image}],
            "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        },
    )


class TestGltfEmbedding:
    """Tests for glTF data-URI texture embedding."""

    def test_readable_image_is_embedded_as_data_uri(self, tmp_path):
        """A readable local image is embedded as a base64 PNG ``data:`` URI.

        Test scenario:
            With ``embed_textures=True`` and ``texture_base`` pointing at the image, the glTF image URI is
            a ``data:image/png;base64,`` payload.
        """
        (tmp_path / "wall.png").write_bytes(_PNG)
        document = write_gltf(
            _textured_model(),
            materials=True,
            embed_textures=True,
            texture_base=tmp_path,
        )
        uri = document["images"][0]["uri"]
        assert uri.startswith("data:image/png;base64,"), (
            f"expected an embedded data URI, got {uri[:40]!r}"
        )

    def test_default_references_image_by_uri(self):
        """Without ``embed_textures`` the glTF references the image by its URI (unchanged default).

        Test scenario:
            A plain textured export keeps the CityJSON image reference.
        """
        document = write_gltf(_textured_model(), materials=True)
        assert document["images"][0]["uri"] == "wall.png", (
            "default must reference the image by URI"
        )

    def test_unreadable_image_falls_back_to_uri(self, tmp_path):
        """An image that does not resolve to a file falls back to a URI reference, no crash.

        Test scenario:
            ``embed_textures=True`` but the image is missing — the glTF keeps the URI reference.
        """
        document = write_gltf(
            _textured_model("missing.png"),
            materials=True,
            embed_textures=True,
            texture_base=tmp_path,
        )
        assert document["images"][0]["uri"] == "missing.png", (
            "a missing image must stay a URI reference"
        )

    def test_remote_image_is_not_embedded(self, tmp_path):
        """A remote (``http://``) image is not fetched; it stays a URI reference.

        Test scenario:
            An absolute URL image is never read from disk even with ``embed_textures``.
        """
        model = _textured_model("http://example.com/wall.png")
        document = write_gltf(
            model, materials=True, embed_textures=True, texture_base=tmp_path
        )
        assert document["images"][0]["uri"] == "http://example.com/wall.png", (
            "a URL must not be embedded"
        )


class TestObjEmbedding:
    """Tests for OBJ texture copying."""

    def test_readable_image_is_copied_beside_the_obj(self, tmp_path):
        """A readable image is copied beside the ``.obj`` and referenced by basename in the ``.mtl``.

        Test scenario:
            ``embed_textures=True`` copies the source image into the output dir; ``map_Kd`` is the basename.
        """
        src = tmp_path / "src"
        src.mkdir()
        (src / "wall.png").write_bytes(_PNG)
        out = tmp_path / "out"
        out.mkdir()
        write_obj(
            _textured_model(),
            out / "model.obj",
            materials=True,
            embed_textures=True,
            texture_base=src,
        )
        assert (out / "wall.png").read_bytes() == _PNG, (
            "the image must be copied beside the .obj"
        )
        assert "map_Kd wall.png" in (out / "model.mtl").read_text(), (
            "map_Kd must reference the basename"
        )

    def test_unreadable_image_is_not_copied(self, tmp_path):
        """A missing image is left as a verbatim reference and not copied, no crash.

        Test scenario:
            ``embed_textures=True`` with a missing image keeps ``map_Kd`` verbatim and copies nothing.
        """
        out = tmp_path / "out"
        out.mkdir()
        write_obj(
            _textured_model("nope.png"),
            out / "m.obj",
            materials=True,
            embed_textures=True,
            texture_base=tmp_path,
        )
        assert "map_Kd nope.png" in (out / "m.mtl").read_text(), (
            "a missing image stays a verbatim reference"
        )
        assert not (out / "nope.png").exists(), (
            "nothing should be copied for a missing image"
        )


class TestEmbedCollisions:
    """Tests that basename collisions / pre-existing files are not clobbered (M2, L1)."""

    def _two_texture_model(self) -> CityModel:
        """Two surfaces, each textured by a different texture whose images share the basename wall.png."""
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[
                [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]],
                [[(2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0)]],
            ],
            appearance={"texture": {"": {"values": [[[0, 0, 1, 2]], [[1, 0, 1, 2]]]}}},
        )
        return CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "textures": [{"image": "a/wall.png"}, {"image": "b/wall.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )

    def test_same_basename_textures_do_not_clobber(self, tmp_path):
        """Two images named ``wall.png`` from different dirs are copied to distinct files (M2).

        Test scenario:
            ``a/wall.png`` (b"AAAA") and ``b/wall.png`` (b"BBBB") both embed; the output keeps both
            distinct, and the two ``map_Kd`` references point at different files.
        """
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "wall.png").write_bytes(b"AAAA")
        (tmp_path / "b" / "wall.png").write_bytes(b"BBBB")
        out = tmp_path / "out"
        out.mkdir()
        write_obj(
            self._two_texture_model(),
            out / "m.obj",
            materials=True,
            embed_textures=True,
            texture_base=tmp_path,
        )
        refs = [
            line.split()[1]
            for line in (out / "m.mtl").read_text().splitlines()
            if line.startswith("map_Kd")
        ]
        assert len(set(refs)) == 2, (
            f"two distinct textures must map to two distinct files, got {refs}"
        )
        copied = {(out / ref).read_bytes() for ref in refs}
        assert copied == {b"AAAA", b"BBBB"}, (
            "both source images must survive distinctly"
        )

    def test_pre_existing_file_is_not_overwritten(self, tmp_path):
        """A texture does not clobber an unrelated pre-existing file of the same name (L1).

        Test scenario:
            ``out/wall.png`` already holds user data; embedding a different ``wall.png`` writes a
            disambiguated name and leaves the user file intact.
        """
        src = tmp_path / "src"
        src.mkdir()
        (src / "wall.png").write_bytes(b"NEW")
        out = tmp_path / "out"
        out.mkdir()
        (out / "wall.png").write_bytes(b"EXISTING_USER_FILE")
        write_obj(
            _textured_model(),
            out / "m.obj",
            materials=True,
            embed_textures=True,
            texture_base=src,
        )
        assert (out / "wall.png").read_bytes() == b"EXISTING_USER_FILE", (
            "the pre-existing file must be kept"
        )
        ref = next(
            line.split()[1]
            for line in (out / "m.mtl").read_text().splitlines()
            if line.startswith("map_Kd")
        )
        assert (out / ref).read_bytes() == b"NEW", (
            "the texture must be written under a disambiguated name"
        )


class TestImageCopierInPlace:
    """Tests that ImageCopier references a source already beside the output without re-copying (R2-L2)."""

    def test_image_already_in_output_dir_is_referenced_in_place(self, tmp_path):
        """When ``texture_base`` is the output dir and the image already lives there, no copy is made.

        Test scenario:
            The image is already ``out/wall.png``; embedding references ``wall.png`` in place and does not
            create a disambiguated ``wall_1.png``.
        """
        out = tmp_path / "out"
        out.mkdir()
        (out / "wall.png").write_bytes(_PNG)
        write_obj(
            _textured_model(),
            out / "m.obj",
            materials=True,
            embed_textures=True,
            texture_base=out,
        )
        assert "map_Kd wall.png" in (out / "m.mtl").read_text(), (
            "the in-place image must be referenced as-is"
        )
        assert not (out / "wall_1.png").exists(), (
            "no redundant copy of the in-place image may be made"
        )
        assert (out / "wall.png").read_bytes() == _PNG, (
            "the in-place image is untouched"
        )


class TestColladaEmbedding:
    """Tests for COLLADA texture copying."""

    def test_readable_image_is_copied_beside_the_dae(self, tmp_path):
        """A readable image is copied beside the ``.dae`` and ``init_from`` references the basename.

        Test scenario:
            ``embed_textures=True`` copies the image into the output dir; the library image is the basename.
        """
        src = tmp_path / "src"
        src.mkdir()
        (src / "wall.png").write_bytes(_PNG)
        out = tmp_path / "out"
        out.mkdir()
        write_collada(
            _textured_model(),
            out / "model.dae",
            materials=True,
            embed_textures=True,
            texture_base=src,
        )
        assert (out / "wall.png").read_bytes() == _PNG, (
            "the image must be copied beside the .dae"
        )
        assert "<init_from>wall.png</init_from>" in (out / "model.dae").read_text(), (
            "init_from is the basename"
        )
