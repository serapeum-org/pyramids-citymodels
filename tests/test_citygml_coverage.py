"""Branch-coverage tests for the CityGML reader's source handling and edge cases."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path
from xml.etree import ElementTree

import pytest

from pyramids_citymodels.readers.citygml import (
    _gml_id,
    _local,
    _srs_dimension,
    looks_like_citygml,
    parse_citygml,
)

DATA = Path(__file__).parent / "data"
CITYGML = DATA / "minimal_building.gml"

_NS = (
    'xmlns:core="http://www.opengis.net/citygml/2.0" '
    'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" '
    'xmlns:gml="http://www.opengis.net/gml"'
)


def _building(
    poslist: str, *, srs: str = ' srsDimension="3"', surface: str = "WallSurface"
) -> str:
    """A one-surface CityGML building wrapping the given posList."""
    return f"""<core:CityModel {_NS}>
      <core:cityObjectMember><bldg:Building gml:id="B">
        <bldg:boundedBy><bldg:{surface}>
          <bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>
            <gml:Polygon><gml:exterior><gml:LinearRing>
              <gml:posList{srs}>{poslist}</gml:posList>
            </gml:LinearRing></gml:exterior></gml:Polygon>
          </gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>
        </bldg:{surface}></bldg:boundedBy>
      </bldg:Building></core:cityObjectMember>
    </core:CityModel>"""


class TestLocalName:
    """Tests for the private ``_local`` tag-name helper."""

    def test_non_string_tag_yields_empty(self) -> None:
        """A non-string tag (e.g. an ElementTree comment factory) maps to an empty name.

        Test scenario:
            ``_local`` on a non-str returns ``""`` rather than raising.
        """
        assert _local(ElementTree.Comment) == ""
        assert _local("{ns}Building") == "Building"


class TestGmlId:
    """Tests for the private ``_gml_id`` attribute reader."""

    def test_skips_non_id_attributes(self) -> None:
        """A non-id attribute is passed over to find ``gml:id``.

        Test scenario:
            An element with a leading non-id attribute still resolves its ``gml:id``.
        """
        element = ElementTree.fromstring(
            '<b xmlns:gml="http://www.opengis.net/gml" foo="1" gml:id="X"/>'
        )
        assert _gml_id(element) == "X"

    def test_returns_none_without_id(self) -> None:
        """An element with no ``gml:id`` returns ``None``."""
        assert _gml_id(ElementTree.fromstring("<b/>")) is None


class TestSrsDimension:
    """Tests for the private ``_srs_dimension`` reader."""

    def test_defaults_when_absent(self) -> None:
        """An element without ``srsDimension`` returns the default."""
        assert _srs_dimension(ElementTree.fromstring("<p/>"), default=3) == 3

    def test_invalid_value_returns_default(self) -> None:
        """A non-integer ``srsDimension`` falls back to the default instead of raising."""
        element = ElementTree.fromstring('<p srsDimension="x"/>')
        assert _srs_dimension(element, default=3) == 3


class TestLooksLikeCityGml:
    """Tests for the ``looks_like_citygml`` content sniff."""

    def test_namespace_only_match(self) -> None:
        """A citygml-namespaced root whose local name is not a known marker still matches.

        Test scenario:
            A ``CityObjectGroup`` element (not in the marker set) is recognized by its namespace.
        """
        xml = '<core:CityObjectGroup xmlns:core="http://www.opengis.net/citygml/2.0"/>'
        assert looks_like_citygml(xml) is True

    def test_open_file_handle(self) -> None:
        """An open file handle is accepted as a source.

        Test scenario:
            Passing an open handle to the fixture returns ``True``.
        """
        with open(CITYGML, encoding="utf-8") as handle:
            assert looks_like_citygml(handle) is True

    def test_unsupported_type_returns_false(self) -> None:
        """A source of an unreadable type returns ``False`` (the TypeError is swallowed)."""
        assert looks_like_citygml(12345) is False

    def test_bytes_source_is_accepted_like_read(self) -> None:
        """``looks_like_citygml`` accepts ``bytes``, agreeing with what ``parse_citygml`` reads (INF5).

        Test scenario:
            The fixture read as raw bytes is still recognized as CityGML.
        """
        assert looks_like_citygml(CITYGML.read_bytes()) is True

    def test_plain_gml_with_a_building_is_not_claimed(self) -> None:
        """A non-CityGML GML document that merely contains a ``Building`` is declined (SF2).

        Test scenario:
            A ``gml``-namespaced ``Building`` (no CityGML namespace) must not be sniffed as
            CityGML, or it would hijack plain GML once the registry dispatches.
        """
        xml = (
            '<gml:FeatureCollection xmlns:gml="http://www.opengis.net/gml">'
            "<gml:featureMember><gml:Building/></gml:featureMember></gml:FeatureCollection>"
        )
        assert looks_like_citygml(xml) is False


class TestParseCityGml:
    """Tests for ``parse_citygml`` source handling and degenerate inputs."""

    def test_parse_from_open_file_handle(self) -> None:
        """An open file handle parses to a model.

        Test scenario:
            Passing an open handle yields the building object.
        """
        with open(CITYGML, encoding="utf-8") as handle:
            model = parse_citygml(handle)
        assert "B1" in model.objects

    def test_unsupported_type_raises(self) -> None:
        """A source of an unsupported type raises ``TypeError``."""
        with pytest.raises(TypeError):
            parse_citygml(12345)

    def test_poslist_without_srs_dimension_defaults_to_3d(self) -> None:
        """A posList without ``srsDimension`` is read as 3D by default.

        Test scenario:
            Nine numbers with no declared dimension become three 3D coordinates.
        """
        model = parse_citygml(_building("0 0 0 1 0 0 1 1 0", srs=""))
        ring = model.objects["B"].geometries[0].boundaries[0][0]
        assert ring[0] == (0.0, 0.0, 0.0)
        assert len(ring[0]) == 3

    def test_invalid_srs_dimension_falls_back(self) -> None:
        """An invalid ``srsDimension`` does not break parsing.

        Test scenario:
            ``srsDimension="x"`` falls back to the 3D default and still parses the ring.
        """
        model = parse_citygml(_building("0 0 0 1 0 0 1 1 0", srs=' srsDimension="x"'))
        assert len(model.objects["B"].geometries[0].boundaries[0][0]) == 3

    def test_degenerate_polygon_is_skipped(self) -> None:
        """A boundary polygon with fewer than three points produces no geometry.

        Test scenario:
            A two-point exterior ring leaves the building without a geometry.
        """
        model = parse_citygml(_building("0 0 0 1 0 0"))
        assert model.objects["B"].geometries == []

    def test_orphan_boundary_surface_without_building_is_ignored(self) -> None:
        """A boundary surface outside any building is attributed to nothing and dropped.

        Test scenario:
            A top-level RoofSurface yields a model with no city objects, without raising.
        """
        xml = f"""<core:CityModel {_NS}>
          <bldg:RoofSurface><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>
            <gml:Polygon><gml:exterior><gml:LinearRing>
              <gml:posList srsDimension="3">0 0 0 1 0 0 1 1 0</gml:posList>
            </gml:LinearRing></gml:exterior></gml:Polygon>
          </gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:RoofSurface>
        </core:CityModel>"""
        model = parse_citygml(xml)
        assert model.objects == {}

    def test_xlink_referenced_geometry_is_resolved(self) -> None:
        """A boundary surface whose geometry is an ``xlink:href`` is resolved, not dropped (SF3).

        Test scenario:
            The WallSurface references a shared polygon by ``xlink:href`` (the polygon is defined
            in the building's solid, not inline under the surface). The surface must still carry
            the referenced geometry.
        """
        xlink = 'xmlns:xlink="http://www.w3.org/1999/xlink"'
        xml = f"""<core:CityModel {_NS} {xlink}>
          <core:cityObjectMember><bldg:Building gml:id="B">
            <bldg:boundedBy><bldg:WallSurface>
              <bldg:lod2MultiSurface><gml:MultiSurface>
                <gml:surfaceMember xlink:href="#shared_poly"/>
              </gml:MultiSurface></bldg:lod2MultiSurface>
            </bldg:WallSurface></bldg:boundedBy>
            <bldg:lod2Solid><gml:Solid><gml:exterior><gml:CompositeSurface>
              <gml:surfaceMember><gml:Polygon gml:id="shared_poly"><gml:exterior>
                <gml:LinearRing>
                  <gml:posList srsDimension="3">0 0 0 1 0 0 1 0 1</gml:posList>
                </gml:LinearRing>
              </gml:exterior></gml:Polygon></gml:surfaceMember>
            </gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod2Solid>
          </bldg:Building></core:cityObjectMember>
        </core:CityModel>"""
        model = parse_citygml(xml)
        geometries = model.objects["B"].geometries
        assert len(geometries) == 1, "the href-referenced wall surface must be resolved"
        assert geometries[0].semantics[0].type == "WallSurface"
        assert len(geometries[0].boundaries) == 1

    def test_polygon_shared_by_two_surfaces_is_collected_once(self) -> None:
        """A polygon reachable from two boundary surfaces of one feature is collected once (SF1 r2).

        Test scenario:
            A WallSurface owns an inline polygon; a RoofSurface of the same building references it
            by xlink:href. The polygon must appear once, keeping the first (document-order) surface's
            semantic type, not be duplicated with an arbitrary type.
        """
        xlink = 'xmlns:xlink="http://www.w3.org/1999/xlink"'
        xml = f"""<core:CityModel {_NS} {xlink}>
          <core:cityObjectMember><bldg:Building gml:id="B">
            <bldg:boundedBy><bldg:WallSurface>
              <bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>
                <gml:Polygon gml:id="p"><gml:exterior><gml:LinearRing>
                  <gml:posList srsDimension="3">0 0 0 1 0 0 1 0 1</gml:posList>
                </gml:LinearRing></gml:exterior></gml:Polygon>
              </gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface>
            </bldg:WallSurface></bldg:boundedBy>
            <bldg:boundedBy><bldg:RoofSurface>
              <bldg:lod2MultiSurface><gml:MultiSurface>
                <gml:surfaceMember xlink:href="#p"/>
              </gml:MultiSurface></bldg:lod2MultiSurface>
            </bldg:RoofSurface></bldg:boundedBy>
          </bldg:Building></core:cityObjectMember>
        </core:CityModel>"""
        geometry = parse_citygml(xml).objects["B"].geometries[0]
        assert len(geometry.boundaries) == 1, "a shared polygon must be collected once"
        assert [surface.type for surface in geometry.semantics] == ["WallSurface"]

    def test_numeric_attributes_are_coerced(self) -> None:
        """Known-numeric building and generic attributes become numbers, not strings (SF5).

        Test scenario:
            ``yearOfConstruction`` and a ``gen:intAttribute`` become ``int``; a
            ``gen:doubleAttribute`` becomes ``float``.
        """
        gen = 'xmlns:gen="http://www.opengis.net/citygml/generics/2.0"'
        xml = f"""<core:CityModel {_NS} {gen}>
          <core:cityObjectMember><bldg:Building gml:id="B">
            <bldg:yearOfConstruction>1985</bldg:yearOfConstruction>
            <gen:intAttribute name="floors"><gen:value>3</gen:value></gen:intAttribute>
            <gen:doubleAttribute name="area"><gen:value>12.5</gen:value></gen:doubleAttribute>
          </bldg:Building></core:cityObjectMember>
        </core:CityModel>"""
        attributes = parse_citygml(xml).objects["B"].attributes
        assert attributes["yearOfConstruction"] == 1985
        assert isinstance(attributes["yearOfConstruction"], int)
        assert attributes["floors"] == 3
        assert isinstance(attributes["floors"], int)
        assert attributes["area"] == 12.5
        assert isinstance(attributes["area"], float)
