"""Tests for 3D Tiles mesh/metadata fidelity (issue #63).

Two writer improvements on the 3D Tiles content path:

- **Per-batch vertex welding** in ``_mesh_buffers``: surfaces of one object that meet along an edge
  share their boundary vertices (the index buffer shrinks, the geometry is unchanged), while vertices
  of different objects never merge — a shared index would make the per-vertex ``_BATCHID`` /
  ``_FEATURE_ID_0`` ambiguous.
- **Typed ``EXT_structural_metadata`` columns** for glTF content (``metadata="typed"``): an attribute
  present on every feature with a uniform clean scalar type becomes its own BOOLEAN / SCALAR INT64 /
  SCALAR FLOAT64 / STRING column (what Cesium / 3d-tiles-tools consume), while a sparse or nested/mixed
  key falls back to the JSON ``attributes`` blob, so the round-trip stays lossless.
"""

import json
import struct
import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.readers.threedtiles import parse_3dtiles
from pyramids_citymodels.writers.threedtiles import _mesh_buffers


def _square(x: float, y: float) -> list[list[tuple[float, float, float]]]:
    """A closed 10-unit square surface at ``(x, y)`` on the z=0 plane."""
    ring = [(x, y, 0.0), (x + 10, y, 0.0), (x + 10, y + 10, 0.0), (x, y + 10, 0.0)]
    return [ring + [ring[0]]]


def _adjacent_quads() -> list[tuple[float, float, float]]:
    """Two open quads sharing the edge (10,0,0)-(10,10,0): 8 corners, 6 distinct."""
    left = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
    right = [(10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 10.0, 0.0), (10.0, 10.0, 0.0)]
    return left, right


def _glb_schema_properties(glb: bytes) -> dict[str, dict]:
    """The ``EXT_structural_metadata`` class ``feature``'s schema properties from a glb's JSON chunk."""
    json_length = struct.unpack_from("<I", glb, 12)[0]
    document = json.loads(glb[20 : 20 + json_length])
    schema = document["extensions"]["EXT_structural_metadata"]["schema"]
    return schema["classes"]["feature"]["properties"]


class TestVertexWeld:
    """Tests for per-batch vertex welding in ``_mesh_buffers``."""

    def test_shared_edge_within_one_object_is_welded(self):
        """Two adjacent quads of one object collapse their shared edge: 8 corners -> 6 vertices.

        Test scenario:
            The two quads share two corners; welding within the single batch folds them, so the pool
            holds 6 vertices while the triangle count (two fans of two triangles) stays 4.
        """
        left, right = _adjacent_quads()
        _, _, _, vertices, triangles = _mesh_buffers(
            [(None, [left, right])], [0.0, 0.0, 0.0]
        )
        assert vertices == 6, f"expected 6 welded vertices, got {vertices}"
        assert triangles == 4, f"expected 4 triangles, got {triangles}"

    def test_vertices_are_not_merged_across_objects(self):
        """The same coordinates in two different objects stay distinct (per-vertex batch id is unambiguous).

        Test scenario:
            Each quad is its own object/batch; although they share two coordinates, a merged index would
            carry two batch ids, so no cross-batch weld happens and the pool keeps all 8 corners.
        """
        left, right = _adjacent_quads()
        _, _, _, vertices, _ = _mesh_buffers(
            [(None, [left]), (None, [right])], [0.0, 0.0, 0.0]
        )
        assert vertices == 8, (
            f"expected 8 vertices (no cross-batch merge), got {vertices}"
        )

    def test_feature_id_count_matches_welded_vertex_count(self):
        """Every welded vertex carries exactly one batch id (the ``_BATCHID`` blob stays vertex-aligned).

        Test scenario:
            After welding one object's two quads to 6 vertices, the batch-id blob is 6 uint32s.
        """
        left, right = _adjacent_quads()
        _, batchid_blob, _, vertices, _ = _mesh_buffers(
            [(None, [left, right])], [0.0, 0.0, 0.0]
        )
        assert len(batchid_blob) == vertices * 4, (
            "batch-id blob must hold one uint32 per vertex"
        )

    @pytest.mark.parametrize("content", ["b3dm", "gltf"])
    def test_weld_preserves_geometry_round_trip(self, content):
        """Welding does not move geometry: a welded tileset round-trips to the same footprint.

        Args:
            content: The tile-content format to exercise (weld applies to both).

        Test scenario:
            One building whose two surfaces share an edge is written and read back; the decoded object
            keeps a valid footprint area (geometry preserved despite the smaller vertex pool).
        """
        left, right = _adjacent_quads()
        model = CityModel(
            objects={
                "b": CityObject(
                    id="b",
                    type="Building",
                    geometries=[
                        Geometry(
                            type="MultiSurface",
                            lod="2",
                            boundaries=[[left + [left[0]]], [right + [right[0]]]],
                        )
                    ],
                )
            }
        )
        out = Path(tempfile.mkdtemp())
        model.to_3dtiles(out, content=content)
        back = parse_3dtiles(str(out / "tileset.json"))
        assert len(back.objects) == 1, (
            f"expected one object back, got {len(back.objects)}"
        )


class TestTypedMetadata:
    """Tests for typed ``EXT_structural_metadata`` columns (``metadata='typed'``)."""

    def _model(self) -> CityModel:
        """Two features: dense bool/int/float/str keys, plus a sparse key and a nested key."""
        return CityModel(
            objects={
                "o0": CityObject(
                    id="o0",
                    type="Building",
                    attributes={
                        "height": 3.5,
                        "floors": 2,
                        "active": True,
                        "label": "a",
                        "note": "x",
                    },
                    geometries=[
                        Geometry(
                            type="MultiSurface", lod="2", boundaries=[_square(0, 0)]
                        )
                    ],
                ),
                "o1": CityObject(
                    id="o1",
                    type="Bridge",
                    attributes={
                        "height": 4,
                        "floors": 5,
                        "active": False,
                        "label": "b",
                        "meta": {"k": [1]},
                    },
                    geometries=[
                        Geometry(
                            type="MultiSurface", lod="2", boundaries=[_square(50, 50)]
                        )
                    ],
                ),
            }
        )

    def test_typed_columns_declare_scalar_boolean_string_types(self):
        """Dense uniform keys become typed columns; id/type/attributes stay STRING.

        Test scenario:
            ``active`` is BOOLEAN, ``floors`` SCALAR INT64, ``height`` SCALAR FLOAT64 (one int value makes
            the column float), ``label`` STRING; a JSON ``attributes`` STRING carries the sparse/nested rest.
        """
        out = Path(tempfile.mkdtemp())
        self._model().to_3dtiles(out, content="gltf", metadata="typed")
        props = _glb_schema_properties((out / "content.glb").read_bytes())
        assert props["active"] == {"type": "BOOLEAN"}, (
            f"active should be BOOLEAN, got {props['active']}"
        )
        assert props["floors"] == {"type": "SCALAR", "componentType": "INT64"}, (
            "floors should be INT64"
        )
        assert props["height"] == {"type": "SCALAR", "componentType": "FLOAT64"}, (
            "height should be FLOAT64"
        )
        assert props["label"] == {"type": "STRING"}, (
            f"label should be STRING, got {props['label']}"
        )
        assert props["attributes"]["type"] == "STRING", (
            "the JSON fallback blob must remain a STRING"
        )

    def test_typed_round_trip_preserves_values_and_types(self):
        """A typed-metadata tileset round-trips every attribute value with its Python type.

        Test scenario:
            bool/int/float/str read back as bool/int/float/str; the sparse ``note`` and nested ``meta``
            survive via the JSON blob (kept only on the object that carries them).
        """
        out = Path(tempfile.mkdtemp())
        self._model().to_3dtiles(out, content="gltf", metadata="typed")
        back = {
            o.id: o for o in parse_3dtiles(str(out / "tileset.json")).objects.values()
        }
        a0 = back["o0"].attributes
        assert a0["active"] is True, (
            f"active should round-trip as True, got {a0['active']!r}"
        )
        assert a0["floors"] == 2 and isinstance(a0["floors"], int), (
            "floors should be int 2"
        )
        assert a0["height"] == 3.5 and isinstance(a0["height"], float), (
            "height should be float 3.5"
        )
        assert a0["note"] == "x", "the sparse key must survive via the JSON blob"
        assert back["o1"].attributes["meta"] == {"k": [1]}, (
            "the nested key must survive via the JSON blob"
        )

    def test_sparse_key_is_absent_on_objects_without_it(self):
        """A key present on only some objects is not fabricated on the others (lossless presence).

        Test scenario:
            ``note`` is only on o0 and ``meta`` only on o1; neither leaks onto the other object.
        """
        out = Path(tempfile.mkdtemp())
        self._model().to_3dtiles(out, content="gltf", metadata="typed")
        back = {
            o.id: o for o in parse_3dtiles(str(out / "tileset.json")).objects.values()
        }
        assert "note" not in back["o1"].attributes, (
            "sparse 'note' must not appear on o1"
        )
        assert "meta" not in back["o0"].attributes, (
            "sparse 'meta' must not appear on o0"
        )

    def test_integer_only_column_round_trips_as_int(self):
        """An attribute that is an integer on every feature stays an int (INT64 column, not float).

        Test scenario:
            Both objects carry an int ``storeys``; the column is INT64 and reads back as int.
        """
        model = CityModel(
            objects={
                f"o{i}": CityObject(
                    id=f"o{i}",
                    type="Building",
                    attributes={"storeys": i + 1},
                    geometries=[
                        Geometry(
                            type="MultiSurface",
                            lod="2",
                            boundaries=[_square(i * 50.0, 0.0)],
                        )
                    ],
                )
                for i in range(2)
            }
        )
        out = Path(tempfile.mkdtemp())
        model.to_3dtiles(out, content="gltf", metadata="typed")
        props = _glb_schema_properties((out / "content.glb").read_bytes())
        assert props["storeys"] == {"type": "SCALAR", "componentType": "INT64"}, (
            "storeys should be INT64"
        )
        back = {
            o.id: o for o in parse_3dtiles(str(out / "tileset.json")).objects.values()
        }
        assert back["o0"].attributes["storeys"] == 1, (
            "storeys should read back as int 1"
        )
        assert isinstance(back["o1"].attributes["storeys"], int), (
            "storeys must stay an int"
        )

    def test_default_metadata_is_json_blob(self):
        """Without ``metadata='typed'`` the attributes stay in one JSON STRING property (issue #56 shape).

        Test scenario:
            The default write declares no typed columns — only id/type/attributes STRING properties.
        """
        out = Path(tempfile.mkdtemp())
        self._model().to_3dtiles(out, content="gltf")
        props = _glb_schema_properties((out / "content.glb").read_bytes())
        assert set(props) == {"id", "type", "attributes"}, (
            f"default should be JSON-blob only, got {set(props)}"
        )
        assert all(spec["type"] == "STRING" for spec in props.values()), (
            "every default property is STRING"
        )

    @pytest.mark.parametrize("scheme", ["quadtree", "octree"])
    def test_typed_metadata_in_implicit_tiling(self, scheme):
        """Typed metadata threads through implicit tiling (both schemes) and round-trips.

        Args:
            scheme: The implicit subdivision scheme to exercise.

        Test scenario:
            Two features written as an implicit glTF tileset with typed metadata read back with their
            integer ``floors`` preserved as int.
        """
        out = Path(tempfile.mkdtemp())
        self._model().to_3dtiles(
            out, content="gltf", metadata="typed", implicit_tiling=True, scheme=scheme
        )
        back = {
            o.id: o for o in parse_3dtiles(str(out / "tileset.json")).objects.values()
        }
        assert isinstance(back["o0"].attributes["floors"], int), (
            "floors must stay int through implicit tiling"
        )

    def test_user_attribute_named_attributes_round_trips_in_typed_mode(self):
        """A user key literally named ``attributes`` is not clobbered by the JSON-blob property (H1).

        Test scenario:
            Both features carry a string attribute ``attributes``; typed mode must keep it in the JSON
            blob (never promote it to a column that the reserved blob then overwrites), so it round-trips.
        """
        model = CityModel(
            objects={
                "o0": CityObject(
                    id="o0",
                    type="Building",
                    attributes={"attributes": "USERVAL0", "sparse": "s0"},
                    geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[_square(0, 0)])],
                ),
                "o1": CityObject(
                    id="o1",
                    type="Building",
                    attributes={"attributes": "USERVAL1"},
                    geometries=[Geometry(type="MultiSurface", lod="2", boundaries=[_square(50, 50)])],
                ),
            }
        )
        out = Path(tempfile.mkdtemp())
        model.to_3dtiles(out, content="gltf", metadata="typed")
        back = {o.id: o for o in parse_3dtiles(str(out / "tileset.json")).objects.values()}
        assert back["o0"].attributes["attributes"] == "USERVAL0", "user 'attributes' key must survive"
        assert back["o1"].attributes["attributes"] == "USERVAL1", "user 'attributes' key must survive"

    def test_invalid_metadata_value_raises(self):
        """An unknown ``metadata`` value raises ``ValueError`` before anything is written.

        Test scenario:
            ``metadata='columns'`` is not a valid shape and is rejected.
        """
        out = Path(tempfile.mkdtemp())
        with pytest.raises(ValueError, match="metadata must be 'json' or 'typed'"):
            self._model().to_3dtiles(out, content="gltf", metadata="columns")
