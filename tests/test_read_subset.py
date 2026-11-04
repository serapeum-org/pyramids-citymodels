"""Tests for read-time subset & filtering (feature #5).

An :class:`~pyramids_citymodels.options.ObjectFilter` lets a reader skip non-matching city objects
*as it parses* (before their geometry is decoded), rather than reading a whole file and discarding
rows afterwards. These tests cover the value object itself, both readers applying it, the driver
``ids``/``types``/``attributes`` keywords, and :meth:`CityModel.from_file`.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import json
from pathlib import Path

import numpy as np
import pytest

from pyramids_citymodels.drivers.citygml import CityGmlDriver
from pyramids_citymodels.drivers.cityjson import CityJsonDriver
from pyramids_citymodels.drivers.threedtiles import ThreeDTilesDriver
from pyramids_citymodels.ir import CityModel, CityObject, Geometry
from pyramids_citymodels.options import (
    ObjectFilter,
    _apply_operator,
    _attribute_matches,
)
from pyramids_citymodels.readers import threedtiles as td
from pyramids_citymodels.readers.citygml import parse_citygml
from pyramids_citymodels.readers.cityjson import _extent_2d, parse_cityjson
from pyramids_citymodels.readers.threedtiles import (
    _bounding_volume_extent,
    _geometry_extent_2d,
    _subtree_outside_bbox,
    parse_3dtiles,
)

DATA = Path(__file__).parent / "data"
# minimal.city.json objects: bldg1 (Building, attrs measuredHeight=10.0, function=residential),
# bldg1_part (BuildingPart).
CITYJSON = DATA / "minimal.city.json"
# minimal_building.gml: one Building, gml:id="B1".
CITYGML = DATA / "minimal_building.gml"
# tileset.json -> building.b3dm: features b0 (Building, height 10.0, footprint (0,0)-(2,2)) and
# b1 (Building, height 20.0, footprint (5,5)-(7,7)).
TILESET = DATA / "tileset.json"


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


def test_coerce_rejects_a_non_numeric_bbox() -> None:
    with pytest.raises(ValueError, match="four numbers"):
        ObjectFilter.coerce(bbox=("a", "b", "c", "d"))


def test_rejects_extent_only_on_a_proven_miss() -> None:
    box = ObjectFilter.coerce(bbox=(0, 0, 10, 10))
    assert box.rejects_extent((20, 20, 30, 30))  # disjoint
    assert not box.rejects_extent((5, 5, 15, 15))  # overlaps
    assert not box.rejects_extent((-5, -5, 1, 1))  # touches/overlaps corner
    assert not box.rejects_extent(None)  # unknown extent kept
    assert not ObjectFilter.coerce(ids="a").rejects_extent(
        (99, 99, 100, 100)
    )  # no bbox


def test_extent_2d_normalizes_6_and_4_value_extents() -> None:
    assert _extent_2d([0, 1, 2, 10, 11, 12]) == (0.0, 1.0, 10.0, 11.0)  # 6-value
    assert _extent_2d([0, 1, 10, 11]) == (0.0, 1.0, 10.0, 11.0)  # 4-value


def test_extent_2d_is_none_for_short_or_non_numeric() -> None:
    assert _extent_2d([0, 1, 2]) is None  # too short
    assert _extent_2d("not a list") is None
    assert _extent_2d(["a", "b", "c", "d"]) is None  # non-numeric


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


def _citygml_with_envelope(corners: str) -> str:
    """One Building gml:id='M' whose gml:Envelope body is ``corners`` (possibly malformed)."""
    return f"""<core:CityModel
      xmlns:core="http://www.opengis.net/citygml/2.0"
      xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
      xmlns:gml="http://www.opengis.net/gml">
      <core:cityObjectMember><bldg:Building gml:id="M">
        <gml:boundedBy><gml:Envelope srsDimension="3">{corners}</gml:Envelope></gml:boundedBy>
      </bldg:Building></core:cityObjectMember>
    </core:CityModel>"""


@pytest.mark.parametrize(
    "corners",
    [
        "<gml:lowerCorner>0 0 0</gml:lowerCorner>",  # upperCorner missing
        "<gml:lowerCorner>a b c</gml:lowerCorner><gml:upperCorner>d e f</gml:upperCorner>",  # non-numeric
        "<gml:lowerCorner>0</gml:lowerCorner><gml:upperCorner>1</gml:upperCorner>",  # 1-D corners
    ],
)
def test_citygml_malformed_envelope_keeps_the_feature(corners: str) -> None:
    # A half-read / non-numeric / 1-D envelope yields no usable extent, so the conservative
    # prefilter keeps the feature rather than pruning it on a bbox it cannot prove it misses.
    model = parse_citygml(
        _citygml_with_envelope(corners),
        ObjectFilter.coerce(bbox=(1000, 1000, 1010, 1010)),
    )
    assert set(model.objects) == {"M"}


def test_cityjson_driver_bbox_keyword() -> None:
    # 'near'/'noext' have no geometry, so the frame is empty, but the subset itself must not error.
    frame = CityJsonDriver().read(_BBOX_CITYJSON, bbox=(0, 0, 20, 20))
    assert len(frame) == 0


def test_bbox_none_selector_reads_all_three() -> None:
    model = parse_cityjson(_BBOX_CITYJSON, None)
    assert set(model.objects) == {"near", "far", "noext"}


# --- exclude / negation -------------------------------------------------------------------------


def test_coerce_exclude_from_mapping_builds_nested_filter() -> None:
    selector = ObjectFilter.coerce(exclude={"types": "Bridge"})
    assert selector is not None
    assert selector.exclude is not None
    assert selector.exclude.types == frozenset({"Bridge"})


def test_coerce_exclude_from_lone_id() -> None:
    selector = ObjectFilter.coerce(exclude="b1")
    assert selector.exclude.ids == frozenset({"b1"})


def test_coerce_exclude_accepts_an_existing_filter() -> None:
    inner = ObjectFilter.coerce(ids="x")
    selector = ObjectFilter.coerce(exclude=inner)
    assert selector.exclude is inner


def test_matches_rejects_an_excluded_object() -> None:
    selector = ObjectFilter.coerce(exclude={"types": "Bridge"})
    assert selector.matches("b", "Building", {})
    assert not selector.matches("x", "Bridge", {})


def test_exclude_combines_with_keep_criteria() -> None:
    # Keep Buildings, but drop the one with function=utility.
    selector = ObjectFilter.coerce(
        types="Building", exclude={"attributes": {"function": "utility"}}
    )
    assert selector.matches("a", "Building", {"function": "residential"})
    assert not selector.matches("b", "Building", {"function": "utility"})
    assert not selector.matches("c", "Bridge", {})  # still fails the keep-type test


def test_cityjson_exclude_drops_matching_objects() -> None:
    model = parse_cityjson(CITYJSON, ObjectFilter.coerce(exclude={"types": "Building"}))
    assert set(model.objects) == {"bldg1_part"}


def test_cityjson_driver_exclude_keyword() -> None:
    full = len(CityJsonDriver().read(CITYJSON))
    excluded = len(CityJsonDriver().read(CITYJSON, exclude={"types": "Building"}))
    assert 0 <= excluded < full  # dropping the Building reduces the frame


# --- CQL2-style attribute predicates ------------------------------------------------------------


def test_attribute_predicate_comparisons() -> None:
    assert _attribute_matches(12.0, {"ge": 10})
    assert _attribute_matches(12.0, {"gt": 10, "lt": 20})
    assert not _attribute_matches(8, {"ge": 10})
    assert not _attribute_matches(25, {"gt": 10, "lt": 20})


def test_attribute_predicate_in_and_not_in() -> None:
    assert _attribute_matches("residential", {"in": ["residential", "commercial"]})
    assert not _attribute_matches("industrial", {"in": ["residential", "commercial"]})
    assert _attribute_matches("industrial", {"nin": ["residential", "commercial"]})


def test_attribute_in_with_a_string_operand_is_membership_not_substring() -> None:
    # A lone string operand is one member, not a substring test: "res" is NOT in ["residential"]
    # (review L3 — otherwise a typo'd {"in": "residential"} silently substring-matches).
    assert not _apply_operator("in", "res", "residential")
    assert _apply_operator("in", "residential", "residential")
    assert _apply_operator("nin", "res", "residential")
    assert not _apply_operator("nin", "residential", "residential")


def test_attribute_predicate_like() -> None:
    assert _attribute_matches("residential", {"like": "resid%"})
    assert _attribute_matches("B1", {"like": "B_"})
    assert not _attribute_matches("commercial", {"like": "resid%"})


def test_attribute_like_on_a_missing_value_is_a_non_match() -> None:
    # A missing attribute must not be stringified to "None" and matched — a `like` on an absent
    # value is a non-match, like every other operator (review M2).
    assert not _attribute_matches(None, {"like": "%"})
    assert not _attribute_matches(None, {"like": "N%"})
    assert not _apply_operator("like", None, "%")


def test_attribute_predicate_between_and_ne() -> None:
    assert _attribute_matches(15, {"between": [10, 20]})
    assert not _attribute_matches(25, {"between": [10, 20]})
    assert _attribute_matches("a", {"ne": "b"})
    assert not _attribute_matches("a", {"ne": "a"})


def test_attribute_comparison_on_missing_value_is_a_non_match_not_error() -> None:
    # A predicate must never raise on heterogeneous/absent data — it just fails to match.
    assert not _attribute_matches(None, {"ge": 10})
    assert not _attribute_matches("text", {"gt": 10})


def test_attribute_eq_operator_explicitly() -> None:
    # The explicit {"eq": ...} spelling is equivalent to a bare scalar.
    assert _attribute_matches("residential", {"eq": "residential"})
    assert not _attribute_matches("commercial", {"eq": "residential"})


def test_attribute_membership_on_a_non_iterable_operand_is_a_non_match() -> None:
    # `in` / `nin` against a non-iterable operand must not raise — it is simply no match.
    assert not _apply_operator("in", "x", 5)
    assert not _apply_operator("nin", "x", 5)


def test_attribute_between_on_a_malformed_operand_is_a_non_match() -> None:
    # A `between` whose operand is not a [low, high] pair, or is incomparable, never raises.
    assert not _apply_operator("between", 5, [1, 2, 3])  # not unpackable to two
    assert not _apply_operator("between", "x", [None, None])  # incomparable


def test_attribute_literal_dict_is_still_matched_by_equality() -> None:
    # A mapping whose keys are not all operators is a literal value, compared with ==.
    assert _attribute_matches({"a": 1}, {"a": 1})
    assert not _attribute_matches({"a": 1}, {"a": 2})


def test_attribute_mapping_with_a_non_operator_key_is_a_literal_not_a_predicate() -> (
    None
):
    # A typo'd / unknown operator key makes the whole mapping a literal value (==), never a
    # predicate — so it does not raise, it simply only matches an identical mapping.
    assert not _attribute_matches(5, {"approx": 5})
    assert _attribute_matches({"approx": 5}, {"approx": 5})


def test_apply_operator_rejects_an_unknown_operator() -> None:
    # The operator dispatch itself guards against an unknown op (defensive; the predicate gate
    # only ever hands it known operators).
    with pytest.raises(ValueError, match="unknown attribute predicate operator"):
        _apply_operator("approx", 5, 5)


def test_cityjson_attribute_predicate_filters_by_height() -> None:
    assert set(
        parse_cityjson(
            CITYJSON, ObjectFilter.coerce(attributes={"measuredHeight": {"ge": 10}})
        ).objects
    ) == {"bldg1"}
    assert (
        parse_cityjson(
            CITYJSON, ObjectFilter.coerce(attributes={"measuredHeight": {"gt": 10}})
        ).objects
        == {}
    )


def test_cityjson_driver_attribute_predicate_keyword() -> None:
    frame = CityJsonDriver().read(
        CITYJSON, attributes={"function": {"in": ["residential", "commercial"]}}
    )
    assert len(frame) >= 1


# --- random sampling ----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, -1, 1.5, -0.5, "half"])
def test_coerce_rejects_an_invalid_sample(bad: object) -> None:
    with pytest.raises(ValueError):
        ObjectFilter.coerce(sample=bad)


def test_sample_ids_count_is_capped_at_the_population() -> None:
    selector = ObjectFilter.coerce(sample=10, seed=0)
    assert selector.sample_ids(["a", "b", "c"]) == {"a", "b", "c"}


def test_sample_ids_count_draws_exactly_n() -> None:
    kept = ObjectFilter.coerce(sample=2, seed=1).sample_ids(["a", "b", "c", "d"])
    assert len(kept) == 2
    assert kept <= {"a", "b", "c", "d"}


def test_sample_ids_fraction_rounds() -> None:
    kept = ObjectFilter.coerce(sample=0.5, seed=1).sample_ids(["a", "b", "c", "d"])
    assert len(kept) == 2


def test_sample_ids_is_reproducible_for_a_seed() -> None:
    ids = [str(n) for n in range(20)]
    first = ObjectFilter.coerce(sample=5, seed=42).sample_ids(ids)
    second = ObjectFilter.coerce(sample=5, seed=42).sample_ids(ids)
    assert first == second


def test_sample_ids_without_sample_keeps_all() -> None:
    assert ObjectFilter.coerce(ids="a").sample_ids(["a", "b", "c"]) == {"a", "b", "c"}


def test_cityjson_sample_thins_the_model() -> None:
    model = parse_cityjson(CITYJSON, ObjectFilter.coerce(sample=1, seed=0))
    assert len(model.objects) == 1
    assert set(model.objects) <= {"bldg1", "bldg1_part"}


def test_cityjson_sample_combines_with_type() -> None:
    # Two Buildings but sample=1 -> exactly one kept.
    doc = {
        "type": "CityJSON",
        "version": "2.0",
        "vertices": [],
        "CityObjects": {
            "a": {"type": "Building"},
            "b": {"type": "Building"},
            "r": {"type": "Road"},
        },
    }
    model = parse_cityjson(doc, ObjectFilter.coerce(types="Building", sample=1, seed=0))
    assert len(model.objects) == 1
    assert set(model.objects) <= {"a", "b"}


def test_citygml_sample_thins_features() -> None:
    model = parse_citygml(CITYGML, ObjectFilter.coerce(sample=0.0))
    assert model.objects == {}  # fraction 0 keeps none


# --- 3D Tiles subset (metadata, exact footprint bbox, prefilter prune) --------------------------


def test_threedtiles_bbox_keeps_only_overlapping_features() -> None:
    model = parse_3dtiles(TILESET, ObjectFilter.coerce(bbox=(0, 0, 3, 3)))
    assert set(model.objects) == {"b0"}  # b1 footprint (5,5)-(7,7) is disjoint


def test_threedtiles_exclude_and_attribute_predicate() -> None:
    assert set(
        parse_3dtiles(TILESET, ObjectFilter.coerce(exclude={"ids": "b1"})).objects
    ) == {"b0"}
    assert set(
        parse_3dtiles(
            TILESET, ObjectFilter.coerce(attributes={"height": {"ge": 15}})
        ).objects
    ) == {"b1"}


def test_threedtiles_driver_bbox_keyword() -> None:
    frame = ThreeDTilesDriver().read(TILESET, bbox=(0, 0, 3, 3))
    assert len(frame) == 1


def test_threedtiles_from_file_applies_selector() -> None:
    model = CityModel.from_file(TILESET, ObjectFilter.coerce(types="Building"))
    assert set(model.objects) == {"b0", "b1"}


def test_geometry_extent_2d_of_a_footprint() -> None:
    obj = CityObject(
        id="x",
        type="Building",
        geometries=[
            Geometry(
                type="MultiSurface",
                lod=None,
                boundaries=[[[(0.0, 0.0), (4.0, 1.0), (2.0, 5.0), (0.0, 0.0)]]],
            )
        ],
    )
    assert _geometry_extent_2d(obj) == (0.0, 0.0, 4.0, 5.0)


def test_bounding_volume_extent_of_a_box() -> None:
    tile = {"boundingVolume": {"box": [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]}}
    assert _bounding_volume_extent(tile, np.eye(4)) == (-10.0, -10.0, 10.0, 10.0)


def test_bounding_volume_extent_is_none_for_region_or_sphere() -> None:
    assert (
        _bounding_volume_extent({"boundingVolume": {"region": [0] * 6}}, np.eye(4))
        is None
    )
    assert _bounding_volume_extent({"boundingVolume": {}}, np.eye(4)) is None


def test_bounding_volume_extent_is_none_for_a_non_numeric_box() -> None:
    # A malformed box (12 entries but not numbers) is treated as "no placeable volume", not an error.
    tile = {"boundingVolume": {"box": ["x"] * 12}}
    assert _bounding_volume_extent(tile, np.eye(4)) is None


def test_subtree_pruned_when_box_proven_outside() -> None:
    # The box bounding volume is z-up in tile space regardless of gltfUpAxis, and footprints reach
    # the same z-up world frame, so pruning is valid for any up-axis — not gated on z-up (review M1).
    tile = {"boundingVolume": {"box": [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]}}
    far = ObjectFilter.coerce(bbox=(100, 100, 110, 110))
    near = ObjectFilter.coerce(bbox=(0, 0, 5, 5))
    assert _subtree_outside_bbox(tile, np.eye(4), far)  # disjoint -> prune
    assert not _subtree_outside_bbox(tile, np.eye(4), near)  # overlaps -> keep
    assert not _subtree_outside_bbox(tile, np.eye(4), None)  # no selector -> keep
    assert not _subtree_outside_bbox(
        {"boundingVolume": {"region": [0] * 6}}, np.eye(4), far
    )  # non-box volume -> cannot place -> keep


@pytest.mark.parametrize("up_axis", ["Y", "Z", "X"])
def test_bbox_prunes_subtree_before_decoding_content(
    up_axis: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prove the bbox is a real prefilter for ANY up-axis (including the Y-up default — review M1): a
    # disjoint subtree's b3dm is never decoded. Stub the tile decode so a feature is produced
    # regardless of geometry, then assert the stub is not called when the subtree is pruned.
    calls: list[str] = []

    def fake_parse(
        data: bytes, transform: object, axis: object = "Y", source_id: str = ""
    ) -> list[CityObject]:
        calls.append(source_id)
        return [
            CityObject(
                id="f",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod=None,
                        boundaries=[[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]]],
                    )
                ],
            )
        ]

    monkeypatch.setattr(td, "_parse_b3dm_tile", fake_parse)
    (tmp_path / "tile.b3dm").write_bytes(b"b3dm" + b"\x00" * 8)
    tileset = {
        "asset": {"version": "1.0", "gltfUpAxis": up_axis},
        "geometricError": 1.0,
        "root": {
            "boundingVolume": {"box": [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10]},
            "geometricError": 0.0,
            "content": {"uri": "tile.b3dm"},
        },
    }
    path = tmp_path / "tileset.json"
    path.write_text(json.dumps(tileset), encoding="utf-8")

    pruned = parse_3dtiles(path, ObjectFilter.coerce(bbox=(100, 100, 110, 110)))
    assert calls == []  # subtree bounding volume is disjoint -> never decoded
    assert pruned.objects == {}

    kept = parse_3dtiles(path)  # no bbox -> content is decoded
    assert calls  # the stub ran this time
    assert set(kept.objects) == {"f"}
