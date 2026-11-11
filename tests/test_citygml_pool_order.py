"""Tests for CityGML appearance-pool composition on round-trip (issue #75).

The CityGML writer emits each distinct painted pool entry once (shared — one ``app:X3DMaterial`` /
``app:ParameterizedTexture`` with one ``app:target`` per painted polygon) and, since #75, in pool-index
order, so a CityJSON→CityGML→CityJSON round-trip preserves the painted pool's *relative* composition.
The documented residual (CityGML has no global indexed pool) is that absolute index values and an
unpainted pool entry are not carried.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.writers.citygml import write_citygml


def _square(x: float) -> list[list[tuple[float, float, float]]]:
    """A closed unit square at ``x`` on the z=0 plane."""
    return [[(x, 0.0, 0.0), (x + 1, 0.0, 0.0), (x + 1, 1.0, 0.0), (x, 1.0, 0.0)]]


def _model(values: list, materials: list[dict]) -> CityModel:
    """A two-surface building with the given per-surface material ``values`` and materials pool."""
    geometry = Geometry(
        type="MultiSurface",
        lod="2",
        boundaries=[_square(0.0), _square(2.0)],
        appearance={"material": {"": {"values": values}}},
    )
    return CityModel(
        objects={"b": CityObject(id="b", type="Building", geometries=[geometry])},
        appearances={"materials": materials},
    )


_MATERIALS = [
    {"name": "red", "diffuseColor": [1.0, 0.0, 0.0]},
    {"name": "green", "diffuseColor": [0.0, 1.0, 0.0]},
    {"name": "blue", "diffuseColor": [0.0, 0.0, 1.0]},
]


class TestCityGmlPoolOrder:
    """Tests for appearance pool ordering / sharing through a CityGML round-trip."""

    def test_painted_pool_relative_order_is_preserved(self, tmp_path):
        """Painted materials re-read in their original pool order, not first-painted order.

        Test scenario:
            Surface 0 uses index 2 (blue), surface 1 uses index 0 (red); re-reading lists red before
            blue (their pool order), not blue before red (the order they are first painted).
        """
        model = _model([2, 0], _MATERIALS)
        model.to_citygml(tmp_path / "m.gml")
        back = CityModel.from_file(tmp_path / "m.gml")
        names = [
            material.get("name") for material in back.appearances.get("materials", [])
        ]
        assert names == ["red", "blue"], f"expected pool order [red, blue], got {names}"

    def test_a_shared_material_is_emitted_once_with_two_targets(self):
        """A material painting two surfaces is one ``app:X3DMaterial`` with two ``app:target`` s.

        Test scenario:
            Both surfaces use material 0; the CityGML carries a single X3DMaterial targeting both polygons.
        """
        xml = write_citygml(_model([0, 0], _MATERIALS))
        assert xml.count("<app:X3DMaterial") == 1, (
            "a shared material must be defined once"
        )
        assert xml.count("<app:target>") == 2, (
            "the shared material must target both painted polygons"
        )

    def test_unpainted_pool_entry_is_dropped(self, tmp_path):
        """An unpainted pool entry has no gml:id anchor and is dropped (documented residual).

        Test scenario:
            Index 1 (green) is painted by no surface; the re-read pool carries only the painted materials.
        """
        model = _model([2, 0], _MATERIALS)  # green (index 1) unpainted
        model.to_citygml(tmp_path / "m.gml")
        back = CityModel.from_file(tmp_path / "m.gml")
        names = [
            material.get("name") for material in back.appearances.get("materials", [])
        ]
        assert "green" not in names, (
            "an unpainted pool entry cannot be anchored in CityGML (residual)"
        )
        assert len(names) == 2, "only the two painted materials survive"
