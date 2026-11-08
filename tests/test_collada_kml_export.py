"""Tests for the COLLADA and KML mesh-export writers (issue #53).

`CityModel.to_collada()` writes a triangulated positions-only COLLADA 1.4.1 `.dae`; `to_kml()` writes
a Placemark per object with one Polygon per surface. Both are plain-geometry, stdlib-XML writers (no
new dependency), routed by `to_file` via `.dae` / `.kml`.
"""

import tempfile
from pathlib import Path
from xml.etree import ElementTree

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pyramids_citymodels.ir import CityModel, CityObject, Geometry


def _model() -> CityModel:
    surface = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
    return CityModel(
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[surface])
                ],
            )
        }
    )


def test_collada_is_well_formed_with_one_triangle() -> None:
    dae = _model().to_collada()
    root = ElementTree.fromstring(dae)  # parses → well-formed XML
    assert root.tag.endswith("COLLADA")
    assert 'triangles count="1"' in dae
    assert "mesh0-positions-array" in dae


def test_collada_empty_model_is_valid_and_geometryless() -> None:
    dae = CityModel().to_collada()
    root = ElementTree.fromstring(dae)
    assert root.tag.endswith("COLLADA")
    assert "<geometry" not in dae  # no geometry
    # The empty library_geometries is omitted entirely (the 1.4.1 XSD requires >=1 geometry in it).
    assert "library_geometries" not in dae


def test_kml_is_well_formed_with_placemark_and_polygon() -> None:
    kml = _model().to_kml()
    root = ElementTree.fromstring(kml)
    assert root.tag.endswith("kml")
    assert "<Placemark>" in kml
    assert "<Polygon>" in kml
    assert "<coordinates>" in kml


def test_kml_coordinates_are_lon_lat_alt_and_closed() -> None:
    ns = "{http://www.opengis.net/kml/2.2}"
    root = ElementTree.fromstring(_model().to_kml())
    coords = root.find(f".//{ns}coordinates")
    assert coords is not None and coords.text is not None
    pairs = coords.text.split()
    assert pairs[0] == "0.0,0.0,0.0"
    assert pairs[0] == pairs[-1]  # LinearRing closed


def test_kml_empty_model_is_valid() -> None:
    kml = CityModel().to_kml()
    root = ElementTree.fromstring(kml)
    assert root.tag.endswith("kml")
    assert "<Placemark>" not in kml


def test_to_file_routes_dae_and_kml() -> None:
    directory = Path(tempfile.mkdtemp())
    assert _model().to_file(directory / "m.dae") is None
    assert _model().to_file(directory / "m.kml") is None
    assert (directory / "m.dae").read_text(encoding="utf-8").count("COLLADA") >= 1
    assert "<Placemark>" in (directory / "m.kml").read_text(encoding="utf-8")
