"""Validate emitted CityGML against the **official OGC XSDs** (issues #48/#49/#50).

These tests prove the writer's CityGML 2.0 and 3.0 output is schema-valid — not merely that it
round-trips through this package's own reader. They are marked ``vendor_data`` because they fetch the
published OGC schemas over the network (the CityGML 2.0 modules from ``schemas.opengis.net`` and the
CityGML 3.0 encoding from the OGC GitHub), so the default ``-m "not vendor_data"`` run skips them;
they run in the dedicated CI job. ``lxml`` (a dev-only dependency) does the XSD validation — the writer
itself uses the stdlib ElementTree and declares no runtime XML dependency.
"""

import urllib.request

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

lxml_etree = pytest.importorskip("lxml.etree")  # dev-only; skip cleanly if absent

from pyramids_citymodels.ir import (  # noqa: E402
    CityModel,
    CityObject,
    Geometry,
    SemanticSurface,
)

pytestmark = pytest.mark.vendor_data

_SCHEMAS_20 = "https://schemas.opengis.net/citygml/{module}/2.0/{file}.xsd"
_MODS_20 = {
    "http://www.opengis.net/citygml/2.0": "https://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd",
    "http://www.opengis.net/citygml/building/2.0": _SCHEMAS_20.format(
        module="building", file="building"
    ),
    "http://www.opengis.net/citygml/bridge/2.0": _SCHEMAS_20.format(
        module="bridge", file="bridge"
    ),
    "http://www.opengis.net/citygml/tunnel/2.0": _SCHEMAS_20.format(
        module="tunnel", file="tunnel"
    ),
    "http://www.opengis.net/citygml/cityfurniture/2.0": _SCHEMAS_20.format(
        module="cityfurniture", file="cityFurniture"
    ),
    "http://www.opengis.net/citygml/vegetation/2.0": _SCHEMAS_20.format(
        module="vegetation", file="vegetation"
    ),
    "http://www.opengis.net/citygml/landuse/2.0": _SCHEMAS_20.format(
        module="landuse", file="landUse"
    ),
    "http://www.opengis.net/citygml/transportation/2.0": _SCHEMAS_20.format(
        module="transportation", file="transportation"
    ),
    "http://www.opengis.net/citygml/generics/2.0": _SCHEMAS_20.format(
        module="generics", file="generics"
    ),
    "http://www.opengis.net/citygml/waterbody/2.0": _SCHEMAS_20.format(
        module="waterbody", file="waterBody"
    ),
    "http://www.opengis.net/citygml/appearance/2.0": _SCHEMAS_20.format(
        module="appearance", file="appearance"
    ),
}
_B30 = (
    "https://raw.githubusercontent.com/opengeospatial/CityGML-3.0Encodings/master/"
    "Moved_to_CITYGML-3.0Encoding_CityGML/Schema/"
)
_MODS_30 = {
    "http://www.opengis.net/citygml/3.0": _B30 + "core.xsd",
    "http://www.opengis.net/citygml/building/3.0": _B30 + "building.xsd",
    "http://www.opengis.net/citygml/construction/3.0": _B30 + "construction.xsd",
    "http://www.opengis.net/citygml/appearance/3.0": _B30 + "appearance.xsd",
}


class _UrllibResolver(lxml_etree.Resolver):
    """Resolve schema imports over the network (libxml2's own fetch is disabled in this build)."""

    def resolve(self, url, pub_id, context):  # noqa: D102
        if url.startswith(("http://", "https://")):
            with urllib.request.urlopen(url, timeout=120) as response:
                return self.resolve_string(response.read(), context, base_url=url)
        return None


def _build(mods: dict[str, str]):
    imports = "".join(
        f'<import namespace="{ns}" schemaLocation="{loc}"/>' for ns, loc in mods.items()
    )
    parser = lxml_etree.XMLParser(no_network=False)
    parser.resolvers.add(_UrllibResolver())
    wrapper = f'<schema xmlns="http://www.w3.org/2001/XMLSchema">{imports}</schema>'
    return lxml_etree.XMLSchema(lxml_etree.fromstring(wrapper.encode(), parser))


@pytest.fixture(scope="module")
def schema_20():
    return _build(_MODS_20)


@pytest.fixture(scope="module")
def schema_30():
    return _build(_MODS_30)


def _assert_valid(schema, xml: str) -> None:
    document = lxml_etree.fromstring(xml.encode())
    if not schema.validate(document):
        errors = "\n".join(f"L{e.line}: {e.message}" for e in schema.error_log)
        pytest.fail(f"CityGML is not XSD-valid:\n{errors}")


def _wall() -> list:
    return [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]]


def _model(version: str, object_type: str, geometry: Geometry) -> CityModel:
    return CityModel(
        version=version,
        objects={"x": CityObject(id="x", type=object_type, geometries=[geometry])},
    )


def _semantic(version: str, object_type: str, surface_type: str) -> CityModel:
    return _model(
        version,
        object_type,
        Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_wall()],
            semantics=[SemanticSurface(type=surface_type)],
            semantic_values=[0],
        ),
    )


def _plain(version: str, object_type: str) -> CityModel:
    return _model(
        version,
        object_type,
        Geometry(type="MultiSurface", lod="2", boundaries=[_wall()]),
    )


@pytest.mark.parametrize(
    "object_type, surface",
    [
        ("Building", "WallSurface"),
        ("Bridge", "RoofSurface"),
        ("Tunnel", "GroundSurface"),
    ],
)
def test_20_boundary_surfaces_are_valid_in_their_module(
    schema_20, object_type, surface
):
    # #49: a bridge/tunnel boundary surface is emitted in its OWN module (brid:/tun:), not bldg:.
    _assert_valid(schema_20, _semantic("2.0", object_type, surface).to_citygml())


@pytest.mark.parametrize(
    "object_type",
    [
        "CityFurniture",
        "SolitaryVegetationObject",
        "PlantCover",
        "LandUse",
        "Road",
        "GenericCityObject",
    ],
)
def test_20_plain_geometry_types_are_valid(schema_20, object_type):
    # #49: each direct-geometry type uses its module's own geometry slot (lodNGeometry / lodNMultiSurface).
    _assert_valid(schema_20, _plain("2.0", object_type).to_citygml())


def test_20_building_solid_is_valid(schema_20):
    _assert_valid(
        schema_20,
        _model(
            "2.0", "Building", Geometry(type="Solid", lod="2", boundaries=[[_wall()]])
        ).to_citygml(),
    )


def test_20_measure_attribute_is_valid(schema_20):
    # #48: a generic attribute with a unit is emitted as a schema-valid gen:measureAttribute.
    model = CityModel(
        version="2.0",
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                attributes={"customLength": 5.0, "measuredHeight": 10.0},
                attribute_units={"customLength": "km", "measuredHeight": "m"},
            )
        },
    )
    _assert_valid(schema_20, model.to_citygml())


@pytest.mark.parametrize("surface", ["WallSurface", "RoofSurface"])
def test_30_boundary_surfaces_are_valid(schema_30, surface):
    # #50: CityGML 3.0 puts the surface in the Construction module under core:boundary.
    _assert_valid(schema_30, _semantic("3.0", "Building", surface).to_citygml())


def test_30_plain_and_solid_are_valid(schema_30):
    _assert_valid(schema_30, _plain("3.0", "Building").to_citygml())
    _assert_valid(
        schema_30,
        _model(
            "3.0", "Building", Geometry(type="Solid", lod="2", boundaries=[[_wall()]])
        ).to_citygml(),
    )


def _mixed(version: str) -> CityModel:
    # One geometry carrying BOTH a semantic boundary surface and a plain surface (value None) — the
    # content-order case (R2-M1): geometry vs boundedBy must be sequenced correctly per version.
    roof = [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
    return _model(
        version,
        "Building",
        Geometry(
            type="MultiSurface",
            lod="2",
            boundaries=[_wall(), roof],
            semantics=[SemanticSurface(type="RoofSurface")],
            semantic_values=[None, 0],
        ),
    )


@pytest.mark.parametrize("version", ["2.0", "3.0"])
def test_mixed_semantic_and_plain_geometry_is_valid(request, version):
    # R2-M1: boundedBy/boundary and the aggregated plain geometry must be ordered per the version's
    # content sequence (2.0: geometry before boundedBy; 3.0: boundary before geometry).
    schema = request.getfixturevalue("schema_20" if version == "2.0" else "schema_30")
    _assert_valid(schema, _mixed(version).to_citygml())


def test_two_same_lod_geometries_aggregate_to_one_property(schema_20):
    # R2-L1: two LOD-2 plain geometries on one object share one lod2MultiSurface (maxOccurs=1).
    roof = [[(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]]
    model = CityModel(
        version="2.0",
        objects={
            "b": CityObject(
                id="b",
                type="Building",
                geometries=[
                    Geometry(type="MultiSurface", lod="2", boundaries=[_wall()]),
                    Geometry(type="MultiSurface", lod="2", boundaries=[roof]),
                ],
            )
        },
    )
    _assert_valid(schema_20, model.to_citygml())
