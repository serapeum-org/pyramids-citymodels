"""Tests for textured / PBR glTF mesh export (issue #63).

``write_gltf(model, materials=True)`` / ``CityModel.to_gltf(materials=True)`` carries the IR appearances
into the glTF: surfaces are grouped by ``(material, texture)`` into one primitive each, a material's
``baseColorFactor`` comes from the CityJSON ``diffuseColor``, and a texture adds a ``TEXCOORD_0``
attribute (from the ``vertices-texture`` UV pool) plus a ``baseColorTexture`` referenced by image URI.
The positions-only default (``materials=False``) is unchanged, and a model without appearances still
yields the single positions-only primitive.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.writers.gltf import write_gltf


def _triangle(x: float = 0.0) -> list[list[tuple[float, float, float]]]:
    """A single triangular surface offset along x."""
    return [[(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 1.0, 0.0)]]


def _material_model(diffuse: list[float] | None = None) -> CityModel:
    """A one-surface model whose surface references material 0 (``diffuse`` or no ``diffuseColor``)."""
    entry: dict = {"name": "m"}
    if diffuse is not None:
        entry["diffuseColor"] = diffuse
    geometry = Geometry(
        type="MultiSurface",
        lod="2",
        boundaries=[_triangle()],
        appearance={"material": {"": {"values": [0]}}},
    )
    return CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
        appearances={"materials": [entry]},
    )


class TestPositionsOnlyDefault:
    """Tests that the default (materials=False) path is unchanged."""

    def test_default_has_no_materials(self):
        """Without ``materials=True`` a model with appearances still exports a positions-only mesh.

        Test scenario:
            The default write of a material-bearing model declares no ``materials`` and one primitive
            with only a POSITION attribute.
        """
        document = write_gltf(_material_model([1.0, 0.0, 0.0]))
        assert "materials" not in document, "default export must not emit materials"
        primitive = document["meshes"][0]["primitives"][0]
        assert set(primitive["attributes"]) == {"POSITION"}, (
            "default primitive is positions-only"
        )


class TestMaterials:
    """Tests for PBR material export (baseColorFactor from diffuseColor)."""

    def test_diffuse_color_becomes_base_color_factor(self):
        """A CityJSON ``diffuseColor`` becomes an opaque glTF ``baseColorFactor``.

        Test scenario:
            A red material [1,0,0] exports baseColorFactor [1,0,0,1] on the surface's primitive.
        """
        document = write_gltf(_material_model([1.0, 0.0, 0.0]), materials=True)
        pbr = document["materials"][0]["pbrMetallicRoughness"]
        assert pbr["baseColorFactor"] == [1.0, 0.0, 0.0, 1.0], (
            f"got {pbr['baseColorFactor']}"
        )

    def test_material_without_diffuse_gets_default_color(self):
        """A material lacking ``diffuseColor`` falls back to the default opaque grey.

        Test scenario:
            A material with only a name still yields a valid baseColorFactor (length-4, alpha 1).
        """
        document = write_gltf(_material_model(None), materials=True)
        base = document["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"]
        assert base == [0.8, 0.8, 0.8, 1.0], f"expected default grey, got {base}"

    def test_distinct_materials_make_distinct_primitives(self):
        """Two surfaces with different materials export as two primitives with their own colours.

        Test scenario:
            Surface 0 -> red material, surface 1 -> blue material; the mesh has two primitives and two
            materials with the matching base colours.
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
        document = write_gltf(model, materials=True)
        assert len(document["meshes"][0]["primitives"]) == 2, (
            "expected one primitive per material"
        )
        colors = {
            tuple(m["pbrMetallicRoughness"]["baseColorFactor"])
            for m in document["materials"]
        }
        assert colors == {(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)}, f"got {colors}"

    def test_no_appearance_model_is_positions_only(self):
        """``materials=True`` on a model without appearances still yields a valid positions-only mesh.

        Test scenario:
            A bare triangle exports one primitive, no materials, POSITION-only — graceful degradation.
        """
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(type="MultiSurface", lod="2", boundaries=[_triangle()])
                    ],
                )
            }
        )
        document = write_gltf(model, materials=True)
        assert "materials" not in document, (
            "a model with no appearances needs no materials"
        )
        assert set(document["meshes"][0]["primitives"][0]["attributes"]) == {"POSITION"}

    def test_empty_model_is_mesh_less(self):
        """An empty model exports a valid, mesh-less glTF even with materials requested.

        Test scenario:
            No surfaces -> an empty scene, regardless of ``materials``.
        """
        document = write_gltf(CityModel(), materials=True)
        assert document["scenes"] == [{"nodes": []}], (
            f"expected empty scene, got {document['scenes']}"
        )


class TestTextures:
    """Tests for texture export (TEXCOORD_0 + baseColorTexture + image URI)."""

    def _textured_model(self) -> CityModel:
        """A one-surface model textured by texture 0 with three UV coordinates."""
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle()],
            appearance={"texture": {"": {"values": [[[0, 0, 1, 2]]]}}},
        )
        return CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "textures": [{"type": "PNG", "image": "wall.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )

    def test_texture_adds_texcoord_and_base_color_texture(self):
        """A textured surface gets a TEXCOORD_0 attribute and a baseColorTexture pointing at the image.

        Test scenario:
            The primitive carries TEXCOORD_0; the material has a baseColorTexture; the image URI is the
            CityJSON ``image``; a default sampler exists.
        """
        document = write_gltf(self._textured_model(), materials=True)
        primitive = document["meshes"][0]["primitives"][0]
        assert "TEXCOORD_0" in primitive["attributes"], (
            "a textured primitive needs TEXCOORD_0"
        )
        pbr = document["materials"][primitive["material"]]["pbrMetallicRoughness"]
        assert "baseColorTexture" in pbr, "a textured material needs a baseColorTexture"
        assert document["images"][0]["uri"] == "wall.png", (
            f"got {document['images'][0]}"
        )
        assert document["samplers"] == [{}], "a default sampler should be declared"

    def test_shared_texture_is_referenced_once(self):
        """Two surfaces using the same texture share one glTF image/texture.

        Test scenario:
            Both surfaces reference texture 0; the document declares exactly one image and one texture.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(0.0), _triangle(5.0)],
            appearance={"texture": {"": {"values": [[[0, 0, 1, 2]], [[0, 0, 1, 2]]]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "textures": [{"type": "PNG", "image": "wall.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = write_gltf(model, materials=True)
        assert len(document["images"]) == 1, (
            "the shared texture image must be declared once"
        )
        assert len(document["textures"]) == 1, (
            "the shared texture must be declared once"
        )

    def test_mismatched_uv_count_falls_back_to_untextured(self):
        """A texture whose UV count does not match the ring is treated as untextured (no TEXCOORD_0).

        Test scenario:
            The ring has three vertices but only two UV indices are given, so the surface is meshed
            without a texture rather than mis-mapping UVs.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle()],
            appearance={"texture": {"": {"values": [[[0, 0, 1]]]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "textures": [{"type": "PNG", "image": "wall.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = write_gltf(model, materials=True)
        assert (
            "TEXCOORD_0" not in document["meshes"][0]["primitives"][0]["attributes"]
        ), "a UV/ring count mismatch must not produce texture coordinates"

    def test_out_of_range_uv_index_does_not_crash(self):
        """A UV index outside the pool is ignored (untextured), never an IndexError.

        Test scenario:
            UV index 9 exceeds the three-entry pool, so the surface is meshed untextured without raising.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle()],
            appearance={"texture": {"": {"values": [[[0, 0, 1, 9]]]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "textures": [{"type": "PNG", "image": "wall.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = write_gltf(model, materials=True)
        assert (
            "TEXCOORD_0" not in document["meshes"][0]["primitives"][0]["attributes"]
        ), "an out-of-range UV index must fall back to untextured"

    def test_texture_without_image_is_untextured(self):
        """A texture whose entry has no ``image`` contributes no glTF image, so the surface is untextured.

        Test scenario:
            Texture 0 has no ``image`` key; the surface is meshed without TEXCOORD_0 and no images are
            declared.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle()],
            appearance={"texture": {"": {"values": [[[0, 0, 1, 2]]]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "textures": [{"type": "PNG"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = write_gltf(model, materials=True)
        assert "images" not in document, (
            "a texture with no image must not declare a glTF image"
        )
        assert "TEXCOORD_0" not in document["meshes"][0]["primitives"][0]["attributes"]

    def test_one_texture_across_two_materials_shares_one_image(self):
        """Two differently-materialled surfaces sharing a texture declare that image/texture once.

        Test scenario:
            Surface 0 (material 0) and surface 1 (material 1) both reference texture 0, forming two
            primitives; the texture's image is reused (one image, one texture) via the dedup cache.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(0.0), _triangle(5.0)],
            appearance={
                "material": {"": {"values": [0, 1]}},
                "texture": {"": {"values": [[[0, 0, 1, 2]], [[0, 0, 1, 2]]]}},
            },
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "materials": [
                    {"name": "red", "diffuseColor": [1.0, 0.0, 0.0]},
                    {"name": "blue", "diffuseColor": [0.0, 0.0, 1.0]},
                ],
                "textures": [{"type": "PNG", "image": "wall.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = write_gltf(model, materials=True)
        assert len(document["meshes"][0]["primitives"]) == 2, (
            "expected one primitive per material"
        )
        assert len(document["images"]) == 1, (
            "the shared texture image must be declared once"
        )
        assert len(document["textures"]) == 1, (
            "the shared texture must be declared once"
        )

    def test_shared_vertex_is_welded_within_a_group(self):
        """Two surfaces of one group that meet at a corner share that vertex (the pool welds it).

        Test scenario:
            Two appearance-less triangles share the vertex (0,0,0); with materials=True they land in one
            group, so the welded pool holds 5 positions (3 + 3 - 1), not 6.
        """
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(
                            type="MultiSurface",
                            lod="2",
                            boundaries=[
                                [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]],
                                [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)]],
                            ],
                        )
                    ],
                )
            }
        )
        document = write_gltf(model, materials=True)
        assert document["accessors"][0]["count"] == 5, (
            f"shared corner should weld to 5 positions, got {document['accessors'][0]['count']}"
        )

    def test_degenerate_ring_is_skipped(self):
        """A surface whose exterior ring has fewer than three vertices is skipped, not meshed.

        Test scenario:
            A geometry with one valid triangle and one two-vertex ring exports a single triangle's
            worth of positions (the degenerate surface contributes none).
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(), [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]],
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])}
        )
        document = write_gltf(model, materials=True)
        assert document["accessors"][0]["count"] == 3, (
            "only the valid triangle's 3 vertices should mesh"
        )

    def test_malformed_texture_leaf_is_untextured(self):
        """A texture leaf whose head is not an integer index is treated as untextured, not an error.

        Test scenario:
            The texture entry starts with a string instead of a texture index, so the surface meshes
            without a texture.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle()],
            appearance={"texture": {"": {"values": [[["x", 0, 1, 2]]]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "textures": [{"type": "PNG", "image": "wall.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = write_gltf(model, materials=True)
        assert "TEXCOORD_0" not in document["meshes"][0]["primitives"][0]["attributes"]
