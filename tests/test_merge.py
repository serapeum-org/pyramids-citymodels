"""Tests for ``CityModel.merge`` — IR-level aggregation of several models (feature #15)."""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import pytest

from pyramids_citymodels.ir import CityModel, CityObject, Geometry


def _model(**objects: str) -> CityModel:
    """A model whose objects are ``id=type`` pairs, with no geometry."""
    return CityModel(
        objects={
            object_id: CityObject(id=object_id, type=object_type)
            for object_id, object_type in objects.items()
        }
    )


def test_merge_unions_disjoint_objects() -> None:
    merged = _model(a="Building").merge(_model(b="Bridge"), _model(c="Tunnel"))
    assert sorted(merged.objects) == ["a", "b", "c"]
    assert merged.objects["b"].type == "Bridge"


def test_merge_returns_a_new_model_and_leaves_inputs_untouched() -> None:
    a = _model(a="Building")
    b = _model(b="Bridge")
    merged = a.merge(b)
    assert merged is not a and merged is not b
    assert sorted(a.objects) == ["a"]
    assert sorted(b.objects) == ["b"]  # inputs are not mutated


def test_merge_rekeys_a_colliding_id() -> None:
    merged = _model(x="Building").merge(_model(x="Bridge"))
    assert sorted(merged.objects) == ["x", "x#2"]
    assert merged.objects["x"].type == "Building"  # the first keeps the plain id
    assert merged.objects["x#2"].type == "Bridge"


def test_merge_rewrites_parent_and_child_links_of_a_rekeyed_clone() -> None:
    first = _model(x="Building")
    second = CityModel(
        objects={
            "x": CityObject(id="x", type="Building", children=["p"]),
            "p": CityObject(id="p", type="BuildingPart", parents=["x"]),
        }
    )
    merged = first.merge(second)
    assert sorted(merged.objects) == ["p", "x", "x#2"]
    assert merged.objects["x#2"].children == ["p"]  # child link follows the rename
    assert merged.objects["p"].parents == ["x#2"]  # back-link follows it too


def test_merge_three_way_collision_suffixes_incrementally() -> None:
    merged = _model(x="Building").merge(_model(x="Bridge"), _model(x="Tunnel"))
    assert sorted(merged.objects) == ["x", "x#2", "x#3"]


def test_merge_takes_crs_and_version_from_the_first_declaring_model() -> None:
    a = CityModel(version="2.0")  # no CRS
    b = CityModel(crs="EPSG:7415", version="1.1")
    merged = a.merge(b)
    assert merged.crs == "EPSG:7415"  # first non-None CRS
    assert merged.version == "2.0"  # first non-None version (a's)


def test_merge_warns_on_conflicting_crs_and_keeps_the_first() -> None:
    a = CityModel(objects={"a": CityObject(id="a", type="Building")}, crs="EPSG:7415")
    b = CityModel(objects={"b": CityObject(id="b", type="Building")}, crs="EPSG:28992")
    with pytest.warns(UserWarning, match="different CRS"):
        merged = a.merge(b)
    assert merged.crs == "EPSG:7415"
    assert sorted(merged.objects) == [
        "a",
        "b",
    ]  # objects still merged despite the conflict


def test_merge_source_format_is_common_value_or_mixed() -> None:
    cj = CityModel(source_format="CityJSON")
    gml = CityModel(source_format="CityGML")
    assert cj.merge(CityModel(source_format="CityJSON")).source_format == "CityJSON"
    assert cj.merge(gml).source_format == "mixed"


def test_merge_shallow_merges_metadata_and_extensions_first_wins() -> None:
    a = CityModel(metadata={"title": "A"}, extensions={"e1": {}})
    b = CityModel(metadata={"title": "B", "author": "x"}, extensions={"e2": {}})
    merged = a.merge(b)
    assert merged.metadata == {"title": "A", "author": "x"}  # A wins the shared key
    assert sorted(merged.extensions) == ["e1", "e2"]


def test_merge_keeps_first_appearances_and_warns_on_a_second() -> None:
    a = CityModel(appearances={"materials": [{"name": "a"}]})
    b = CityModel(appearances={"materials": [{"name": "b"}]})
    with pytest.warns(UserWarning, match="appearance"):
        merged = a.merge(b)
    assert merged.appearances == {"materials": [{"name": "a"}]}


def test_merge_takes_second_appearances_when_first_has_none() -> None:
    a = CityModel()
    b = CityModel(appearances={"materials": [{"name": "b"}]})
    assert a.merge(b).appearances == {"materials": [{"name": "b"}]}


def test_merge_preserves_geometry() -> None:
    geometry = Geometry(type="MultiSurface", lod="2", boundaries=[[[(0.0, 0.0, 0.0)]]])
    a = _model(a="Building")
    b = CityModel(
        objects={"g": CityObject(id="g", type="Building", geometries=[geometry])}
    )
    merged = a.merge(b)
    assert merged.objects["g"].geometries[0].boundaries == [[[(0.0, 0.0, 0.0)]]]


def test_merge_does_not_alias_geometry_objects_with_the_inputs() -> None:
    # merge promises a new, independent model; a later edit to a merged geometry must not reach the
    # source (review M2 — geometries were shallow-copied, aliasing the Geometry instances).
    geometry = Geometry(type="MultiSurface", lod="2", boundaries=[[[(0.0, 0.0, 0.0)]]])
    b = CityModel(
        objects={"g": CityObject(id="g", type="Building", geometries=[geometry])}
    )
    merged = CityModel().merge(b)
    assert merged.objects["g"].geometries[0] is not b.objects["g"].geometries[0]
    merged.objects["g"].geometries[0].boundaries.append("leak")
    assert b.objects["g"].geometries[0].boundaries == [
        [[(0.0, 0.0, 0.0)]]
    ]  # source untouched


def test_merge_with_no_others_is_a_copy() -> None:
    a = _model(a="Building")
    merged = a.merge()
    assert merged is not a
    assert sorted(merged.objects) == ["a"]
    merged.objects["a"].type = "Bridge"
    assert a.objects["a"].type == "Building"  # the copy is independent
