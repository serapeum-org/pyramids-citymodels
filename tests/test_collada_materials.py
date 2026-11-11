"""Tests for textured / PBR COLLADA export (issue #71).

``write_collada(model, materials=True)`` / ``CityModel.to_collada(materials=True)`` carries the IR
appearances into the ``.dae``: surfaces are grouped by ``(material, texture)`` into per-material
``<triangles>`` bound through ``library_materials`` / ``library_effects`` (diffuse ``<color>`` from the
CityJSON ``diffuseColor``), and a textured group adds a ``TEXCOORD`` input plus a sampler over a
``library_images`` image referenced by its ``image`` URI. The positions-only default is unchanged.
"""

from xml.dom import minidom

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.writers.collada import write_collada


def _triangle(x: float = 0.0) -> list[list[tuple[float, float, float]]]:
    """A single triangular surface offset along x."""
    return [[(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 1.0, 0.0)]]


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


class TestColladaDefault:
    """Tests that the default (materials=False) path is unchanged."""

    def test_default_has_no_material_library(self):
        """Without materials the ``.dae`` carries no ``library_materials``.

        Test scenario:
            A material-bearing model written plain emits a positions-only mesh and no material library.
        """
        dae = write_collada(_material_model([1.0, 0.0, 0.0]))
        assert "library_materials" not in dae, (
            "the plain .dae must not declare materials"
        )


class TestColladaMaterials:
    """Tests for COLLADA material / texture export."""

    def test_output_is_well_formed_and_declares_material(self):
        """A materialled export is well-formed XML declaring material/effect libraries.

        Test scenario:
            A red material produces parseable XML with library_effects + library_materials and a
            material-bound ``<triangles>``.
        """
        dae = write_collada(_material_model([1.0, 0.0, 0.0]), materials=True)
        minidom.parseString(dae)  # raises if not well-formed
        assert "library_effects" in dae, "a materialled .dae declares effects"
        assert "library_materials" in dae, "a materialled .dae declares materials"
        assert 'material="sym_0"' in dae, (
            "the triangles must bind their material symbol"
        )

    def test_diffuse_color_is_the_effect_color(self):
        """The CityJSON ``diffuseColor`` becomes the lambert diffuse ``<color>`` (RGBA).

        Test scenario:
            A red material yields a diffuse colour of ``1.0 0.0 0.0 1.0``.
        """
        dae = write_collada(_material_model([1.0, 0.0, 0.0]), materials=True)
        assert "<color>1.0 0.0 0.0 1.0</color>" in dae, f"unexpected colour in:\n{dae}"

    def test_texture_adds_image_sampler_and_texcoord(self):
        """A textured surface adds a library image, a sampler, a TEXCOORD input and a bind.

        Test scenario:
            The textured material references the image, binds a sampler via ``<texture>``, the
            ``<triangles>`` has a TEXCOORD input, and the instance_material binds the UV set.
        """
        dae = write_collada(
            _material_model([0.0, 0.0, 1.0], image="wall.png"), materials=True
        )
        minidom.parseString(dae)
        assert "library_images" in dae and "wall.png" in dae, (
            "the texture image must be declared"
        )
        assert 'semantic="TEXCOORD"' in dae, (
            "a textured triangles needs a TEXCOORD input"
        )
        assert "<texture " in dae, (
            "the effect diffuse must reference the sampler texture"
        )
        assert "bind_vertex_input" in dae, (
            "the UV set must be bound on the instance_material"
        )

    def test_distinct_materials_make_distinct_triangles(self):
        """Two surfaces with different materials produce two material-bound ``<triangles>``.

        Test scenario:
            Surface 0 -> red, surface 1 -> blue; the mesh has two ``<triangles material=...>`` and two
            ``<material>`` definitions.
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
        dae = write_collada(model, materials=True)
        document = minidom.parseString(dae)
        assert len(document.getElementsByTagName("triangles")) == 2, (
            "expected one triangles per material"
        )
        assert len(document.getElementsByTagName("material")) == 2, (
            "expected two materials"
        )

    def test_distinct_textures_make_distinct_images(self):
        """Two surfaces with different textures declare two ``<image>`` entries.

        Test scenario:
            Surface 0 -> texture 0 (a.png), surface 1 -> texture 1 (b.png); library_images has both.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(0.0), _triangle(5.0)],
            appearance={
                "material": {"": {"values": [0, 1]}},
                "texture": {"": {"values": [[[0, 0, 1, 2]], [[1, 0, 1, 2]]]}},
            },
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "materials": [
                    {"name": "a", "diffuseColor": [1.0, 0.0, 0.0]},
                    {"name": "b", "diffuseColor": [0.0, 1.0, 0.0]},
                ],
                "textures": [{"image": "a.png"}, {"image": "b.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = minidom.parseString(write_collada(model, materials=True))
        assert len(document.getElementsByTagName("image")) == 2, (
            "two textures must declare two images"
        )

    def test_textured_and_untextured_groups_coexist(self):
        """A textured group and a colour-only group share one mesh; only the textured one has TEXCOORD.

        Test scenario:
            Surface 0 is textured, surface 1 is colour-only; exactly one ``<triangles>`` carries a
            TEXCOORD input.
        """
        geometry = Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_triangle(0.0), _triangle(5.0)],
            appearance={
                "material": {"": {"values": [0, 1]}},
                "texture": {"": {"values": [[[0, 0, 1, 2]], None]}},
            },
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "materials": [
                    {"name": "a", "diffuseColor": [1.0, 0.0, 0.0]},
                    {"name": "b", "diffuseColor": [0.0, 1.0, 0.0]},
                ],
                "textures": [{"image": "t.png"}],
                "vertices-texture": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            },
        )
        document = minidom.parseString(write_collada(model, materials=True))
        with_texcoord = [
            triangles
            for triangles in document.getElementsByTagName("triangles")
            if any(
                inp.getAttribute("semantic") == "TEXCOORD"
                for inp in triangles.getElementsByTagName("input")
            )
        ]
        assert len(with_texcoord) == 1, (
            "exactly one group should carry a TEXCOORD input"
        )

    def test_solid_geometry_is_materialled(self):
        """A ``Solid`` (deeper nesting) pairs material values in lock-step into per-material triangles.

        Test scenario:
            A one-shell Solid of two surfaces with materials [0, 1] yields two material-bound triangles.
        """
        shell = [_triangle(0.0), _triangle(5.0)]
        geometry = Geometry(
            type="Solid",
            lod="2",
            boundaries=[shell],
            appearance={"material": {"": {"values": [[0, 1]]}}},
        )
        model = CityModel(
            objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
            appearances={
                "materials": [
                    {"name": "a", "diffuseColor": [1.0, 0.0, 0.0]},
                    {"name": "b", "diffuseColor": [0.0, 0.0, 1.0]},
                ]
            },
        )
        document = minidom.parseString(write_collada(model, materials=True))
        assert len(document.getElementsByTagName("triangles")) == 2, (
            "a Solid's two surfaces -> two groups"
        )

    def test_material_without_diffuse_is_grey(self):
        """A material with no ``diffuseColor`` degrades to the default grey colour, not a crash.

        Test scenario:
            A material carrying only a name yields the default grey diffuse colour.
        """
        dae = write_collada(_material_model(None), materials=True)
        assert "<color>0.8 0.8 0.8 1.0</color>" in dae, (
            f"expected grey colour in:\n{dae}"
        )

    def test_missing_image_is_untextured_color(self):
        """A texture whose entry has no image degrades to a plain colour effect (no sampler).

        Test scenario:
            The texture pool entry lacks ``image``; the effect is a ``<color>``, not a ``<texture>``.
        """
        model = _material_model([1.0, 0.0, 0.0], image="x")
        model.appearances["textures"] = [{"type": "PNG"}]  # drop the image
        dae = write_collada(model, materials=True)
        assert "<texture " not in dae, "a texture with no image must not bind a sampler"
        assert "<color>" in dae, "the material must fall back to a plain colour"

    def test_empty_model_is_valid_and_geometry_less(self):
        """An empty model exports a well-formed, geometry-less ``.dae`` even with materials requested.

        Test scenario:
            No objects -> parseable XML with a visual scene but no ``<triangles>`` and no materials.
        """
        dae = write_collada(CityModel(), materials=True)
        document = minidom.parseString(dae)
        assert len(document.getElementsByTagName("triangles")) == 0, (
            "an empty model has no triangles"
        )
        assert "library_materials" not in dae, "an empty model declares no materials"

    def test_no_appearance_model_is_valid(self):
        """``materials=True`` on a model with no appearances still yields a valid positions mesh.

        Test scenario:
            A bare triangle exports well-formed XML with one ``<triangles>`` and no material library.
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
        dae = write_collada(model, materials=True)
        document = minidom.parseString(dae)
        assert "library_materials" not in dae, (
            "a model with no appearances needs no materials"
        )
        assert len(document.getElementsByTagName("triangles")) == 1, (
            "the geometry is still written"
        )
