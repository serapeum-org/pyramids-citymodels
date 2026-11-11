"""Tests for textured / PBR Wavefront OBJ export (issue #71).

``write_obj(model, materials=True)`` / ``CityModel.to_obj(path, materials=True)`` carries the IR
appearances into the OBJ: surfaces are grouped by ``(material, texture)`` into ``usemtl`` groups, a
companion ``<stem>.mtl`` is written beside the ``.obj`` (``mtllib``) with each material's ``Kd`` diffuse
colour (from the CityJSON ``diffuseColor``) and ``map_Kd`` texture image, and a textured surface emits
``vt`` texture coordinates and ``f v/vt`` faces. Because the ``.mtl`` is a sidecar, ``materials=True``
needs a path. The plain-geometry default (``materials=False``) is unchanged.
"""

import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.writers.obj import write_obj


def _triangle(x: float = 0.0) -> list[list[tuple[float, float, float]]]:
    """A single triangular surface offset along x."""
    return [[(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 1.0, 0.0)]]


def _write(model: CityModel) -> tuple[str, str | None]:
    """Write ``model`` with materials to a temp dir; return ``(obj_text, mtl_text or None)``."""
    out = Path(tempfile.mkdtemp()) / "model.obj"
    write_obj(model, out, materials=True)
    mtl = out.with_suffix(".mtl")
    return out.read_text(), (mtl.read_text() if mtl.exists() else None)


def _material_model(
    diffuse: list[float] | None = None, image: str | None = None
) -> CityModel:
    """A one-surface model referencing material 0 (optionally textured by texture 0)."""
    appearance: dict = {"material": {"": {"values": [0]}}}
    appearances: dict = {
        "materials": [{"name": "m", **({"diffuseColor": diffuse} if diffuse else {})}]
    }
    if image is not None:
        appearance["texture"] = {"": {"values": [[[0, 0, 1, 2]]]}}
        appearances["textures"] = [{"type": "PNG", "image": image}]
        appearances["vertices-texture"] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    geometry = Geometry(
        type="MultiSurface", lod="2", boundaries=[_triangle()], appearance=appearance
    )
    return CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
        appearances=appearances,
    )


class TestPlainDefault:
    """Tests that the default (materials=False) path is unchanged."""

    def test_default_has_no_mtllib(self):
        """Without materials the OBJ carries no ``mtllib`` and keeps the ``o`` object markers.

        Test scenario:
            A material-bearing model written plain emits object-grouped faces and no material library.
        """
        text = write_obj(_material_model([1.0, 0.0, 0.0]))
        assert "mtllib" not in text, (
            "the plain OBJ must not reference a material library"
        )
        assert "o b" in text.splitlines(), "the plain OBJ keeps object group markers"


class TestObjMaterials:
    """Tests for OBJ material / texture export."""

    def test_diffuse_becomes_kd_and_mtllib_is_referenced(self):
        """A material's ``diffuseColor`` becomes ``Kd`` in a sidecar the OBJ references via ``mtllib``.

        Test scenario:
            A red material writes ``mtllib model.mtl`` + ``usemtl mat_0``; the ``.mtl`` has ``Kd 1.0 0.0 0.0``.
        """
        obj, mtl = _write(_material_model([1.0, 0.0, 0.0]))
        assert "mtllib model.mtl" in obj, "the OBJ must reference the .mtl sidecar"
        assert "usemtl mat_0" in obj, "the surface's material group must be selected"
        assert mtl is not None, "a material model must write a .mtl sidecar"
        assert "Kd 1.0 0.0 0.0" in mtl, f"Kd should be the diffuseColor, got:\n{mtl}"

    def test_texture_emits_vt_map_kd_and_vertex_texcoord_faces(self):
        """A textured surface emits ``vt``, ``map_Kd`` and ``f v/vt`` faces.

        Test scenario:
            A textured material writes ``map_Kd wall.png``, three ``vt`` lines, and a face with ``v/vt``.
        """
        obj, mtl = _write(_material_model([0.0, 0.0, 1.0], image="wall.png"))
        assert "map_Kd wall.png" in mtl, "the texture image must be a map_Kd"
        assert any(line.startswith("vt ") for line in obj.splitlines()), (
            "textured OBJ needs vt lines"
        )
        face = next(line for line in obj.splitlines() if line.startswith("f "))
        assert "/" in face, f"a textured face must reference v/vt, got {face!r}"

    def test_distinct_materials_make_distinct_usemtl_groups(self):
        """Two surfaces with different materials produce two ``newmtl`` and two ``usemtl`` groups.

        Test scenario:
            Surface 0 -> red, surface 1 -> blue; the .mtl declares mat_0 and mat_1, both used.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(0.0), _triangle(5.0)],
            appearance={"material": {"": {"values": [0, 1]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "materials": [
                    {"name": "red", "diffuseColor": [1.0, 0.0, 0.0]},
                    {"name": "blue", "diffuseColor": [0.0, 0.0, 1.0]},
                ]
            },
        )
        obj, mtl = _write(model)
        assert mtl.count("newmtl ") == 2, f"expected two materials, got:\n{mtl}"
        assert "usemtl mat_0" in obj and "usemtl mat_1" in obj, (
            "both material groups must be selected"
        )

    def test_mixed_model_emits_plain_faces_before_any_usemtl(self):
        """An unthemed surface in a mixed model is not left under a prior ``usemtl`` (M1).

        Test scenario:
            Surface 0 is red, surface 1 has a ``null`` material; because OBJ ``usemtl`` is stateful, the
            plain face must be written before any ``usemtl`` (default material), not after mat_0 where it
            would silently render red.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(0.0), _triangle(5.0)],
            appearance={"material": {"": {"values": [0, None]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "materials": [{"name": "red", "diffuseColor": [1.0, 0.0, 0.0]}]
            },
        )
        obj, _mtl = _write(model)
        lines = obj.splitlines()
        first_usemtl = next(
            (i for i, line in enumerate(lines) if line.startswith("usemtl")), len(lines)
        )
        plain_faces = [line for line in lines[:first_usemtl] if line.startswith("f ")]
        assert plain_faces, (
            "the unthemed surface's face must precede any usemtl (default material)"
        )

    def test_string_return_with_materials_raises(self):
        """``materials=True`` with no path raises, since the ``.mtl`` sidecar needs a file path.

        Test scenario:
            ``write_obj(model, materials=True)`` (path=None) is rejected with a clear message.
        """
        with pytest.raises(ValueError, match="needs a path to write the .mtl sidecar"):
            write_obj(_material_model([1.0, 0.0, 0.0]), materials=True)

    def test_no_appearance_model_writes_no_sidecar(self):
        """``materials=True`` on a model with no appearances writes plain faces and no ``.mtl``.

        Test scenario:
            A bare triangle writes an OBJ with no ``mtllib`` and no sidecar file.
        """
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(type="MultiSurface", lod="2", boundaries=_triangle())
                    ],
                )
            }
        )
        obj, mtl = _write(model)
        assert "mtllib" not in obj, (
            "a model with no appearances needs no material library"
        )
        assert mtl is None, "no .mtl sidecar should be written"

    def test_material_without_diffuse_falls_back_to_grey(self):
        """A material with no ``diffuseColor`` degrades to the default grey ``Kd``, never a crash.

        Test scenario:
            A material carrying only a name yields the default grey Kd.
        """
        _obj, mtl = _write(_material_model(None))
        assert "Kd 0.8 0.8 0.8" in mtl, (
            f"a material with no diffuseColor should be grey, got:\n{mtl}"
        )

    def test_shared_uv_is_deduplicated_in_vt(self):
        """Two textured surfaces referencing the same UVs share the ``vt`` pool (3 vt, not 6).

        Test scenario:
            Both surfaces of one textured material reference the same three UV indices, so the OBJ emits
            three ``vt`` lines, not six.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(0.0), _triangle(5.0)],
            appearance={
                "material": {"": {"values": [0, 0]}},
                "texture": {"": {"values": [[[0, 0, 1, 2]], [[0, 0, 1, 2]]]}},
            },
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "materials": [{"name": "m", "diffuseColor": [1.0, 0.0, 0.0]}],
                "textures": [{"type": "PNG", "image": "t.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        obj, _mtl = _write(model)
        vt_lines = [line for line in obj.splitlines() if line.startswith("vt ")]
        assert len(vt_lines) == 3, (
            f"shared UVs should dedupe to 3 vt lines, got {len(vt_lines)}"
        )

    def test_missing_image_writes_kd_without_map(self):
        """A material with a texture whose entry has no image writes ``Kd`` but no ``map_Kd``.

        Test scenario:
            The texture pool entry lacks ``image``; the surface is untextured (no map_Kd, no vt).
        """
        model = _material_model([1.0, 0.0, 0.0], image="x")
        model.appearances["textures"] = [{"type": "PNG"}]  # drop the image
        obj, mtl = _write(model)
        assert "map_Kd" not in mtl, "a texture with no image must not emit map_Kd"
        assert not any(line.startswith("vt ") for line in obj.splitlines()), (
            "no UVs without an image"
        )
