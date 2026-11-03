"""Tests for read-time subset & filtering (feature #5).

An :class:`~pyramids_citymodels.options.ObjectFilter` lets a reader skip non-matching city objects
*as it parses* (before their geometry is decoded), rather than reading a whole file and discarding
rows afterwards. These tests cover the value object itself, both readers applying it, the driver
``ids``/``types``/``attributes`` keywords, and :meth:`CityModel.from_file`.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

from pathlib import Path

import pytest

from pyramids_citymodels.drivers.citygml import CityGmlDriver
from pyramids_citymodels.drivers.cityjson import CityJsonDriver
from pyramids_citymodels.ir import CityModel
from pyramids_citymodels.options import ObjectFilter
from pyramids_citymodels.readers.citygml import parse_citygml
from pyramids_citymodels.readers.cityjson import parse_cityjson

DATA = Path(__file__).parent / "data"
# minimal.city.json objects: bldg1 (Building), bldg1_part (BuildingPart).
CITYJSON = DATA / "minimal.city.json"
# minimal_building.gml: one Building, gml:id="B1".
CITYGML = DATA / "minimal_building.gml"


# --- ObjectFilter value object ------------------------------------------------------------------


def test_coerce_returns_none_when_no_criterion() -> None:
    assert ObjectFilter.coerce() is None
    assert ObjectFilter.coerce(ids=None, types=None, attributes=None) is None


def test_coerce_treats_empty_iterables_as_no_filter() -> None:
    # An empty id/type list must mean "do not filter on this", never "reject everything".
    assert ObjectFilter.coerce(ids=[], types=[], attributes={}) is None


def test_coerce_accepts_a_lone_string_as_one_member() -> None:
    selector = ObjectFilter.coerce(types="Building")
    assert selector is not None
    assert selector.types == frozenset({"Building"})


def test_coerce_builds_sets_and_attribute_map() -> None:
    selector = ObjectFilter.coerce(ids=["a", "b"], attributes={"storeys": 3})
    assert selector.ids == frozenset({"a", "b"})
    assert selector.attributes == {"storeys": 3}


def test_matches_requires_every_set_criterion() -> None:
    selector = ObjectFilter.coerce(types=["Building"], attributes={"storeys": 3})
    assert selector.matches("b1", "Building", {"storeys": 3})
    assert not selector.matches("b1", "Building", {"storeys": 5})  # wrong attribute
    assert not selector.matches("w1", "WallSurface", {"storeys": 3})  # wrong type


def test_matches_ignores_unset_criteria() -> None:
    assert ObjectFilter.coerce(ids="x").matches("x", "AnyType", {"any": "attr"})


# --- CityJSON reader ----------------------------------------------------------------------------


def test_cityjson_selector_by_id_keeps_only_matching_objects() -> None:
    model = parse_cityjson(CITYJSON, ObjectFilter.coerce(ids="bldg1"))
    assert set(model.objects) == {"bldg1"}
    # The kept object is fully read — geometry still dereferenced.
    assert model.objects["bldg1"].geometries


def test_cityjson_selector_by_type() -> None:
    model = parse_cityjson(CITYJSON, ObjectFilter.coerce(types="BuildingPart"))
    assert set(model.objects) == {"bldg1_part"}


def test_cityjson_selector_by_attribute() -> None:
    model = parse_cityjson(
        CITYJSON, ObjectFilter.coerce(attributes={"function": "residential"})
    )
    assert set(model.objects) == {"bldg1"}


def test_cityjson_selector_no_match_yields_empty_model() -> None:
    model = parse_cityjson(CITYJSON, ObjectFilter.coerce(types="Bridge"))
    assert model.objects == {}


def test_cityjson_none_selector_reads_everything() -> None:
    model = parse_cityjson(CITYJSON, None)
    assert set(model.objects) == {"bldg1", "bldg1_part"}


def test_cityjson_filtering_a_child_leaves_a_dangling_parent_ref() -> None:
    # Filtering is per object and does not re-link: a kept part still names its (dropped) parent.
    model = parse_cityjson(CITYJSON, ObjectFilter.coerce(ids="bldg1_part"))
    assert set(model.objects) == {"bldg1_part"}
    assert model.objects["bldg1_part"].parents == ["bldg1"]
    assert "bldg1" not in model.objects


# --- CityGML reader -----------------------------------------------------------------------------


def test_citygml_selector_by_type_keeps_building() -> None:
    model = parse_citygml(CITYGML, ObjectFilter.coerce(types="Building"))
    assert set(model.objects) == {"B1"}
    assert model.objects["B1"].geometries  # surfaces still parsed for the kept feature


def test_citygml_selector_by_id() -> None:
    model = parse_citygml(CITYGML, ObjectFilter.coerce(ids="B1"))
    assert set(model.objects) == {"B1"}


def test_citygml_selector_no_match_yields_empty_model() -> None:
    model = parse_citygml(CITYGML, ObjectFilter.coerce(types="BuildingPart"))
    assert model.objects == {}


# --- Drivers ------------------------------------------------------------------------------------


def test_cityjson_driver_types_keyword_filters_the_frame() -> None:
    assert len(CityJsonDriver().read(CITYJSON, types="Building")) >= 1
    assert len(CityJsonDriver().read(CITYJSON, types="Bridge")) == 0


def test_cityjson_driver_ids_keyword_filters_the_frame() -> None:
    # bldg1_part has no geometry of its own, so an id subset to it alone yields no areal rows.
    frame = CityJsonDriver().read(CITYJSON, ids="bldg1")
    assert len(frame) >= 1


def test_citygml_driver_types_keyword_filters_the_frame() -> None:
    assert len(CityGmlDriver().read(CITYGML, types="Building")) >= 1
    assert len(CityGmlDriver().read(CITYGML, types="BuildingPart")) == 0


def test_driver_subset_kwargs_are_consumed_not_forwarded_to_flatten() -> None:
    # ids/types/attributes must be popped before flatten, else to_feature_collection rejects them.
    frame = CityJsonDriver().read(CITYJSON, ids="bldg1", strategy="footprint")
    assert len(frame) >= 1


# --- CityModel.from_file ------------------------------------------------------------------------


def test_from_file_applies_selector() -> None:
    model = CityModel.from_file(CITYJSON, ObjectFilter.coerce(ids="bldg1"))
    assert set(model.objects) == {"bldg1"}


def test_from_file_without_selector_reads_everything() -> None:
    model = CityModel.from_file(CITYJSON)
    assert set(model.objects) == {"bldg1", "bldg1_part"}


# --- bbox prefilter (#32) -----------------------------------------------------------------------

# A CityJSON doc with three objects: one whose declared extent is inside the query bbox, one far
# outside, and one with no declared extent at all.
_BBOX_CITYJSON = {
    "type": "CityJSON",
    "version": "2.0",
    "vertices": [],
    "CityObjects": {
        "near": {"type": "Building", "geographicalExtent": [0, 0, 0, 10, 10, 5]},
        "far": {
            "type": "Building",
            "geographicalExtent": [1000, 1000, 0, 1010, 1010, 5],
        },
        "noext": {"type": "Building"},
    },
}

# The CityGML counterpart: each feature declares its own gml:Envelope.
_BBOX_CITYGML = """<core:CityModel
  xmlns:core="http://www.opengis.net/citygml/2.0"
  xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
  xmlns:gml="http://www.opengis.net/gml">
  <core:cityObjectMember><bldg:Building gml:id="near">
    <gml:boundedBy><gml:Envelope srsDimension="3">
      <gml:lowerCorner>0 0 0</gml:lowerCorner>
      <gml:upperCorner>10 10 5</gml:upperCorner>
    </gml:Envelope></gml:boundedBy>
  </bldg:Building></core:cityObjectMember>
  <core:cityObjectMember><bldg:Building gml:id="far">
    <gml:boundedBy><gml:Envelope srsDimension="3">
      <gml:lowerCorner>1000 1000 0</gml:lowerCorner>
      <gml:upperCorner>1010 1010 5</gml:upperCorner>
    </gml:Envelope></gml:boundedBy>
  </bldg:Building></core:cityObjectMember>
</core:CityModel>"""


def test_coerce_normalizes_bbox_corner_order() -> None:
    assert ObjectFilter.coerce(bbox=(10, 10, 0, 0)).bbox == (0.0, 0.0, 10.0, 10.0)


def test_coerce_rejects_a_non_four_number_bbox() -> None:
    with pytest.raises(ValueError):
        ObjectFilter.coerce(bbox=(0, 0, 10))


def test_rejects_extent_only_on_a_proven_miss() -> None:
    box = ObjectFilter.coerce(bbox=(0, 0, 10, 10))
    assert box.rejects_extent((20, 20, 30, 30))  # disjoint
    assert not box.rejects_extent((5, 5, 15, 15))  # overlaps
    assert not box.rejects_extent((-5, -5, 1, 1))  # touches/overlaps corner
    assert not box.rejects_extent(None)  # unknown extent kept
    assert not ObjectFilter.coerce(ids="a").rejects_extent(
        (99, 99, 100, 100)
    )  # no bbox


def test_cityjson_bbox_skips_only_proven_outside_objects() -> None:
    model = parse_cityjson(_BBOX_CITYJSON, ObjectFilter.coerce(bbox=(0, 0, 20, 20)))
    # 'far' is skipped (its extent is disjoint); 'noext' is kept (no declared extent).
    assert set(model.objects) == {"near", "noext"}


def test_cityjson_bbox_combines_with_type() -> None:
    model = parse_cityjson(
        _BBOX_CITYJSON, ObjectFilter.coerce(types="Building", bbox=(0, 0, 20, 20))
    )
    assert set(model.objects) == {"near", "noext"}


def test_citygml_bbox_skips_proven_outside_features() -> None:
    model = parse_citygml(_BBOX_CITYGML, ObjectFilter.coerce(bbox=(0, 0, 20, 20)))
    assert set(model.objects) == {"near"}  # 'far' envelope is disjoint


def test_cityjson_driver_bbox_keyword() -> None:
    # 'near'/'noext' have no geometry, so the frame is empty, but the subset itself must not error.
    frame = CityJsonDriver().read(_BBOX_CITYJSON, bbox=(0, 0, 20, 20))
    assert len(frame) == 0


def test_bbox_none_selector_reads_all_three() -> None:
    model = parse_cityjson(_BBOX_CITYJSON, None)
    assert set(model.objects) == {"near", "far", "noext"}
