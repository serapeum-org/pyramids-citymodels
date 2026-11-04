"""Read options: how the 3D, hierarchical city model is projected onto flat features.

A GeoDataFrame is 2D-centric, flat and one-geometry-per-row; a city model is 3D, hierarchical
and graph-linked. There is no single right projection, so the projection is exposed as options
rather than hardcoded — the lesson the QGIS CityJSON Loader and 3DCityDB both teach (issues #9
and #10):

- :class:`LoadStrategy` — *what a row is*: one row per city object (its footprint) or one row
  per semantic surface (wall/roof/ground as individual features).
- :class:`GeometryDimension` — *how Z is carried*: dropped for a 2D footprint, or kept for 2.5D
  surfaces. A ``footprint`` load is always 2D regardless, since a footprint is a plan projection.
- ``lod`` — which level of detail to select when a city object carries several.
- ``child_inherits_parent_attributes`` — whether a ``BuildingPart`` row also carries its
  ``Building``'s attributes.

:class:`ReadOptions` bundles these, and :meth:`ReadOptions.coerce` accepts either the enums or
their plain-string values so a driver can pass user kwargs straight through.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)

import operator
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "GeometryDimension",
    "LoadStrategy",
    "ObjectFilter",
    "ReadOptions",
]


class LoadStrategy(StrEnum):
    """What a row in the output represents."""

    #: One row per city object, geometry = its 2D footprint (its ground surfaces).
    FOOTPRINT = "footprint"
    #: One row per semantic surface — wall/roof/ground each become an individual feature.
    SEMANTIC_SURFACES = "semantic_surfaces"


class GeometryDimension(StrEnum):
    """How Z is carried in the output geometry."""

    #: Drop Z — 2D geometry.
    XY = "xy"
    #: Keep Z — 2.5D geometry (coordinates retain their height).
    XYZ = "xyz"


def _as_str_set(value: Any) -> "frozenset[str] | None":
    """Coerce an id/type criterion to a ``frozenset[str]``, or ``None`` when empty.

    A lone ``str`` is one member (not a set of its characters); any other iterable is taken
    member-wise. An empty value (``None``, ``[]``) becomes ``None`` — "do not filter on this" — so
    an empty criterion never silently rejects every object.

    Raises:
        ValueError: If ``value`` is neither ``None``, a string, nor an iterable of items — mirroring
            the curated errors ``bbox``/``sample`` raise, rather than a bare ``TypeError``.

    Examples:
        >>> _as_str_set("Building")
        frozenset({'Building'})
        >>> sorted(_as_str_set(["a", "b"]))
        ['a', 'b']
        >>> _as_str_set([]) is None
        True
    """
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value}) or None
    try:
        members = frozenset(str(item) for item in value)
    except TypeError as error:
        raise ValueError(
            f"ids/types must be a string or an iterable of strings, got {value!r}"
        ) from error
    return members or None


def _as_bbox(value: Any) -> "tuple[float, float, float, float] | None":
    """Coerce a bbox criterion to a normalized ``(xmin, ymin, xmax, ymax)``, or ``None`` when empty.

    Accepts any 4-number sequence and orders the corners, so a caller may pass them in either
    order. An empty value becomes ``None`` — "do not filter by bbox".

    Raises:
        ValueError: If ``value`` is not a sequence of exactly four numbers.

    Examples:
        >>> _as_bbox((10, 0, 0, 10))
        (0.0, 0.0, 10.0, 10.0)
        >>> _as_bbox(None) is None
        True
    """
    if value is None:
        return None
    try:
        coords = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "bbox must be a sequence of four numbers (xmin, ymin, xmax, ymax)"
        ) from error
    if len(coords) != 4:
        raise ValueError(
            f"bbox must be (xmin, ymin, xmax, ymax) — four numbers, got {len(coords)}"
        )
    xmin, ymin, xmax, ymax = coords
    return (min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))


def _as_sample(value: Any) -> "int | float | None":
    """Coerce a ``sample`` criterion to a count (``int``) or a fraction (``float``), or ``None``.

    An ``int`` is a maximum object *count*; a ``float`` is a *fraction* of the matching objects and
    must lie in ``[0.0, 1.0]``. ``bool`` is rejected (``True``/``False`` are ``int`` in Python, and
    a boolean sample size is always a mistake). ``None`` means "do not sample".

    Raises:
        ValueError: If ``value`` is a bool, a negative count, a fraction outside ``[0, 1]``, or not
            a number at all.

    Examples:
        >>> _as_sample(3)
        3
        >>> _as_sample(0.25)
        0.25
        >>> _as_sample(None) is None
        True
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("sample must be an int count or a float fraction, not a bool")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"sample count must be >= 0, got {value}")
        return value
    if isinstance(value, float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"sample fraction must be in [0.0, 1.0], got {value}")
        return value
    raise ValueError(
        f"sample must be an int count or a float fraction in [0, 1], got {value!r}"
    )


#: The comparison operators a CQL2-style attribute predicate supports, mapped to their callables.
_COMPARISON_OPERATORS = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
}
#: Every operator key a predicate mapping may use. A mapping whose keys are *all* in this set is
#: read as a predicate; any other mapping is compared for literal equality — so a city object whose
#: attribute value is itself a JSON object is matchable by ``==`` *unless every one of its keys is an
#: operator name* (e.g. a literal ``{"eq": 1}`` is read as a predicate, not matched by equality; wrap
#: it as ``{"eq": {"eq": 1}}`` to match such a value literally).
_PREDICATE_OPERATORS = frozenset(
    {"eq", "ne", "in", "nin", "not_in", "like", "between", *_COMPARISON_OPERATORS}
)


def _like_to_regex(pattern: str) -> "re.Pattern[str]":
    r"""Compile a SQL/CQL2-style ``LIKE`` pattern to a case-insensitive regex.

    ``%`` and ``*`` match any run of characters; ``_`` and ``?`` match a single character; every
    other character is matched literally. The match is anchored (full-string).

    Examples:
        >>> bool(_like_to_regex("resid%").fullmatch("residential"))
        True
        >>> bool(_like_to_regex("B_").fullmatch("B1"))
        True
        >>> bool(_like_to_regex("resid%").fullmatch("commercial"))
        False
    """
    out = []
    for char in pattern:
        if char in "%*":
            out.append(".*")
        elif char in "_?":
            out.append(".")
        else:
            out.append(re.escape(char))
    return re.compile("".join(out), re.IGNORECASE | re.DOTALL)


def _apply_operator(op: str, actual: Any, operand: Any) -> bool:
    """Evaluate one predicate operator against an object's actual attribute value.

    A comparison whose operands are not orderable (e.g. a missing attribute, ``None``, compared with
    ``>``) is a non-match rather than an error, so a predicate never raises on heterogeneous data.

    Raises:
        ValueError: If ``op`` is not a recognized operator.
    """
    if op == "eq":
        return bool(actual == operand)
    if op == "ne":
        return bool(actual != operand)
    if op in _COMPARISON_OPERATORS:
        try:
            return bool(_COMPARISON_OPERATORS[op](actual, operand))
        except TypeError:
            return False
    if op in ("in", "nin", "not_in"):
        # A lone string operand is one member, not a sequence of characters, so membership is an
        # equality test against it — never a substring match (``"res" in "residential"``).
        members = [operand] if isinstance(operand, str) else operand
        try:
            present = actual in members
        except TypeError:
            # A non-iterable operand (a typo) leaves membership indeterminate: ``in`` cannot be
            # satisfied, but ``nin`` keeps conservatively rather than silently emptying the result.
            return op != "in"
        return present if op == "in" else not present
    if op == "between":
        try:
            low, high = operand
            return bool(low <= actual <= high)
        except (TypeError, ValueError):
            return False
    if op == "like":
        # A missing attribute is a non-match, consistent with every other operator — never
        # stringify ``None`` to ``"None"`` and pattern-match that.
        if actual is None:
            return False
        return _like_to_regex(str(operand)).fullmatch(str(actual)) is not None
    raise ValueError(
        f"unknown attribute predicate operator {op!r}; expected one of "
        f"{', '.join(sorted(_PREDICATE_OPERATORS))}"
    )


def _attribute_matches(actual: Any, expected: Any) -> bool:
    """True when an object's ``actual`` attribute value satisfies the ``expected`` criterion.

    ``expected`` is either a scalar (matched by ``==``, the back-compatible default) or a predicate
    mapping whose keys are *all* operators from :data:`_PREDICATE_OPERATORS` — then every operator
    in it must hold (AND), giving CQL2-style ``>`` / ``in`` / ``like`` / ``between`` tests.

    Examples:
        >>> _attribute_matches("residential", "residential")
        True
        >>> _attribute_matches(12.0, {"ge": 10})
        True
        >>> _attribute_matches(8, {"ge": 10})
        False
        >>> _attribute_matches("residential", {"like": "resid%"})
        True
    """
    if (
        isinstance(expected, Mapping)
        and expected
        and all(key in _PREDICATE_OPERATORS for key in expected)
    ):
        return all(
            _apply_operator(op, actual, operand) for op, operand in expected.items()
        )
    return bool(actual == expected)


@dataclass(frozen=True)
class ObjectFilter:
    """A read-time subset: which city objects a reader keeps, decided *before* geometry is built.

    This is the parse-time half of "subset & filtering". Rather than read a whole city-scale file
    and discard rows afterwards (filtering a ``FeatureCollection`` is a GIS operation that belongs
    upstream in ``pyramids-core``), a reader consults this to **skip non-matching objects as it
    parses**, so their geometry is never decoded. An object is kept only when it satisfies *every*
    criterion that is set — it matches ``ids`` AND ``types`` AND each entry of ``attributes`` AND
    ``bbox``, is **not** matched by ``exclude``, and survives any ``sample``; a criterion left unset
    (``None``/empty) is not tested, so an empty filter keeps everything.

    ``ids``/``types``/``attributes`` are **metadata** tests, decided from an object's header before
    any geometry is touched. An ``attributes`` value may be a scalar (matched by ``==``) or a
    **CQL2-style predicate mapping** — ``{"ge": 10}``, ``{"in": [...]}``, ``{"like": "resid%"}``,
    ``{"between": [lo, hi]}``, ``{"gt": .., "lt": ..}`` — whose operators must all hold (see
    :func:`_attribute_matches`). ``exclude`` is a nested metadata-only filter whose match *drops*
    an object (negation / "keep X but not Y"). ``sample`` thins the surviving objects to a random
    subset — a ``float`` fraction or an ``int`` count — reproducibly when ``seed`` is given.

    ``bbox`` is a **conservative prefilter**: an object is skipped only when a *declared extent the
    file already carries* (a CityJSON ``geographicalExtent`` or a CityGML ``gml:Envelope``) proves
    it lies wholly outside the box. An object with no declared extent is **kept**, because proving
    it is outside would require decoding the very geometry this is meant to skip; the bbox result is
    thus a superset of the exact selection, which a caller refines downstream with
    ``FeatureCollection.clip_by_rect``. The bbox is read in the file's own CRS (no reprojection here
    — that is ``pyramids-core``'s job).

    Filtering is per object and does not pull in relatives: keeping a ``BuildingPart`` whose
    ``Building`` was filtered out leaves a dangling ``parents`` id (the flattener resolves a missing
    parent to ``None``), and in CityGML a dropped feature's boundary surfaces roll up to the nearest
    *kept* ancestor feature. This is deliberate — the filter selects, it does not re-link.

    Attributes:
        ids: Keep only objects whose id is in this set, or ``None`` to not filter by id.
        types: Keep only objects whose type is in this set, or ``None`` to not filter by type.
        attributes: Keep only objects whose attributes satisfy each of these criteria (a scalar for
            equality, or a predicate mapping); an empty mapping does not filter by attribute.
        bbox: Keep only objects whose declared extent overlaps this ``(xmin, ymin, xmax, ymax)``
            (objects with no declared extent are kept), or ``None`` to not filter by bbox.
        exclude: A nested metadata-only :class:`ObjectFilter` whose match *removes* an object, or
            ``None`` to exclude nothing. Its own ``bbox``/``sample`` are ignored.
        sample: Thin the surviving objects to a random subset — a ``float`` fraction of them or an
            ``int`` maximum count — or ``None`` to keep them all.
        seed: Seed for the ``sample`` draw, for a reproducible subset; ``None`` draws freshly.

    Examples:
        - Keep only tall Buildings — a type test plus a CQL2-style attribute predicate:
            >>> keep = ObjectFilter.coerce(
            ...     types="Building", attributes={"measuredHeight": {"ge": 10}}
            ... )
            >>> keep.matches("b1", "Building", {"measuredHeight": 12.0})
            True
            >>> keep.matches("b2", "Building", {"measuredHeight": 4.0})
            False

        - Keep two types but drop one of them again with ``exclude`` (negation):
            >>> f = ObjectFilter.coerce(
            ...     types=["Building", "Bridge"], exclude={"types": "Bridge"}
            ... )
            >>> sorted(f.types)
            ['Bridge', 'Building']
            >>> f.matches("x", "Building", {})
            True
            >>> f.matches("y", "Bridge", {})
            False

        - An empty filter coerces to ``None`` (nothing to test, so the reader keeps everything):
            >>> ObjectFilter.coerce() is None
            True
    """

    ids: "frozenset[str] | None" = None
    types: "frozenset[str] | None" = None
    attributes: dict[str, Any] = field(default_factory=dict)
    bbox: "tuple[float, float, float, float] | None" = None
    exclude: "ObjectFilter | None" = None
    sample: "int | float | None" = None
    seed: int | None = None

    def __hash__(self) -> int:
        """Hash the filter by its hashable projection, consistent with equality.

        ``frozen=True`` advertises hashability, but the auto-generated hash would raise on the
        ``attributes`` ``dict``. Equal filters carry equal ``attributes`` (hence equal keys), so
        hashing the *sorted keys* (not the values, which may themselves be predicate mappings) keeps
        the ``a == b implies hash(a) == hash(b)`` contract while staying safe to compute.
        """
        return hash(
            (
                self.ids,
                self.types,
                tuple(sorted(self.attributes)),
                self.bbox,
                self.exclude,
                self.sample,
                self.seed,
            )
        )

    @classmethod
    def coerce(
        cls,
        ids: "str | Iterable[str] | None" = None,
        types: "str | Iterable[str] | None" = None,
        attributes: "Mapping[str, Any] | None" = None,
        bbox: "Iterable[float] | None" = None,
        exclude: "ObjectFilter | Mapping[str, Any] | str | Iterable[str] | None" = None,
        sample: "int | float | None" = None,
        seed: int | None = None,
    ) -> "ObjectFilter | None":
        """Build an :class:`ObjectFilter`, or ``None`` when no criterion is given.

        Accepts a lone ``str`` or any iterable of strings for ``ids``/``types``, a mapping for
        ``attributes`` (scalar values or predicate mappings), and a 4-number
        ``(xmin, ymin, xmax, ymax)`` sequence for ``bbox`` (corners in either order). ``exclude`` is
        an already-built :class:`ObjectFilter`, a mapping of ``ids``/``types``/``attributes`` to
        drop, or a lone id / iterable of ids to drop. ``sample`` is an ``int`` count or a ``float``
        fraction, drawn reproducibly under ``seed``. Returns ``None`` when every criterion is empty,
        so a reader can treat "no filter" as a cheap ``is None`` check and skip the test entirely.

        Raises:
            ValueError: If ``bbox`` is not four numbers, or ``sample`` is not a valid count/fraction.

        Examples:
            >>> ObjectFilter.coerce() is None
            True
            >>> ObjectFilter.coerce(types="Building").types
            frozenset({'Building'})
            >>> ObjectFilter.coerce(bbox=(0, 0, 10, 10)).bbox
            (0.0, 0.0, 10.0, 10.0)
            >>> ObjectFilter.coerce(exclude={"types": "Bridge"}).exclude.types
            frozenset({'Bridge'})
        """
        id_set = _as_str_set(ids)
        type_set = _as_str_set(types)
        attribute_map = dict(attributes) if attributes else {}
        box = _as_bbox(bbox)
        exclude_filter = cls._coerce_exclude(exclude)
        sample_value = _as_sample(sample)
        if (
            id_set is None
            and type_set is None
            and not attribute_map
            and box is None
            and exclude_filter is None
            and sample_value is None
        ):
            return None
        return cls(
            ids=id_set,
            types=type_set,
            attributes=attribute_map,
            bbox=box,
            exclude=exclude_filter,
            sample=sample_value,
            seed=seed,
        )

    @classmethod
    def _coerce_exclude(
        cls,
        exclude: "ObjectFilter | Mapping[str, Any] | str | Iterable[str] | None",
    ) -> "ObjectFilter | None":
        """Coerce the ``exclude`` argument to a nested metadata :class:`ObjectFilter`, or ``None``.

        An existing :class:`ObjectFilter` is used as-is; a mapping is read for its
        ``ids``/``types``/``attributes`` keys; anything else is taken as a lone id or an iterable of
        ids to drop. A ``bbox`` or ``sample`` inside an ``exclude`` mapping is ignored — exclusion
        is a metadata test only.
        """
        if exclude is None:
            return None
        if isinstance(exclude, ObjectFilter):
            return exclude
        if isinstance(exclude, Mapping):
            return cls.coerce(
                ids=exclude.get("ids"),
                types=exclude.get("types"),
                attributes=exclude.get("attributes"),
            )
        return cls.coerce(ids=exclude)

    def matches(
        self, object_id: str, object_type: str, attributes: "Mapping[str, Any]"
    ) -> bool:
        """True when a city object passes every metadata criterion this filter sets.

        Tests :attr:`ids`, :attr:`types` and each entry of :attr:`attributes` (a scalar by equality,
        a predicate mapping by its operators), then rejects the object when :attr:`exclude` matches
        it. This is the metadata gate; the ``bbox`` prefilter is :meth:`rejects_extent` and the
        ``sample`` thinning is :meth:`sample_ids`, applied by the reader around this call.

        Args:
            object_id: The object's id, tested against :attr:`ids`.
            object_type: The object's type, tested against :attr:`types`.
            attributes: The object's attributes, tested against :attr:`attributes`.

        Returns:
            ``True`` when the object is kept, ``False`` when it is filtered out.

        Examples:
            >>> keep = ObjectFilter.coerce(types=["Building"], attributes={"storeys": 3})
            >>> keep.matches("b1", "Building", {"storeys": 3})
            True
            >>> keep.matches("b2", "Building", {"storeys": 5})
            False
            >>> keep.matches("w1", "WallSurface", {"storeys": 3})
            False
            >>> tall = ObjectFilter.coerce(attributes={"measuredHeight": {"ge": 10}})
            >>> tall.matches("b1", "Building", {"measuredHeight": 12.0})
            True
            >>> not_bridge = ObjectFilter.coerce(exclude={"types": "Bridge"})
            >>> not_bridge.matches("x", "Bridge", {})
            False
        """
        if self.ids is not None and object_id not in self.ids:
            return False
        if self.types is not None and object_type not in self.types:
            return False
        for key, expected in self.attributes.items():
            if not _attribute_matches(attributes.get(key), expected):
                return False
        if self.exclude is not None and self.exclude.matches(
            object_id, object_type, attributes
        ):
            return False
        return True

    def rejects_extent(
        self, extent: "tuple[float, float, float, float] | None"
    ) -> bool:
        """True when a declared ``extent`` proves the object lies wholly outside :attr:`bbox`.

        The conservative bbox prefilter: it returns ``True`` (skip the object) only when there is a
        bbox *and* a known 2D ``extent`` *and* the two are disjoint. With no bbox, or an unknown
        extent (``None``), or any overlap, it returns ``False`` (keep the object) — so an object is
        never dropped on a bbox the reader cannot prove it misses.

        Args:
            extent: The object's declared extent as ``(xmin, ymin, xmax, ymax)``, or ``None`` when
                the file declares none for it.

        Returns:
            ``True`` to skip the object, ``False`` to keep it.

        Examples:
            >>> box = ObjectFilter.coerce(bbox=(0, 0, 10, 10))
            >>> box.rejects_extent((20, 20, 30, 30))  # disjoint -> skip
            True
            >>> box.rejects_extent((5, 5, 15, 15))  # overlaps -> keep
            False
            >>> box.rejects_extent(None)  # unknown extent -> keep
            False
            >>> ObjectFilter.coerce(ids="a").rejects_extent((20, 20, 30, 30))  # no bbox -> keep
            False
        """
        if self.bbox is None or extent is None:
            return False
        exmin, eymin, exmax, eymax = extent
        bxmin, bymin, bxmax, bymax = self.bbox
        return exmax < bxmin or exmin > bxmax or eymax < bymin or eymin > bymax

    def sample_ids(self, ids: "Sequence[Any] | Iterable[Any]") -> set:
        """Draw the random subset of already-matched object keys this filter keeps.

        Applies :attr:`sample` to the keys of the objects that already passed :meth:`matches` and
        :meth:`rejects_extent` — a reader gathers those (cheap, metadata-only) *before* decoding any
        geometry and asks this which to actually build. With no :attr:`sample` every key is kept; a
        ``float`` keeps ``round(fraction * n)`` of them and an ``int`` keeps at most that many,
        chosen with a :class:`random.Random` seeded by :attr:`seed` (so a given ``seed`` and input
        order reproduce the same subset). The keys are opaque — real object ids for CityJSON/3D
        Tiles, positional indices for anonymous CityGML features — and only membership is used.

        Args:
            ids: The keys of the objects that passed the metadata and bbox tests, in document order.

        Returns:
            The subset of keys to keep, as a set.

        Examples:
            >>> f = ObjectFilter.coerce(sample=2, seed=0)
            >>> kept = f.sample_ids(["a", "b", "c", "d"])
            >>> len(kept)
            2
            >>> kept <= {"a", "b", "c", "d"}
            True
            >>> sorted(ObjectFilter.coerce(ids="a").sample_ids(["a", "b"]))  # no sample -> keep all
            ['a', 'b']
        """
        items = list(ids)
        if self.sample is None:
            return set(items)
        count = (
            round(self.sample * len(items))
            if isinstance(self.sample, float)
            else self.sample
        )
        count = max(0, min(count, len(items)))
        return set(random.Random(self.seed).sample(items, count))


@dataclass(frozen=True)
class ReadOptions:
    """The resolved set of read options the flattener consumes.

    Attributes:
        strategy: The :class:`LoadStrategy`.
        lod: The level of detail to select, or ``None`` to take the highest each object offers.
        geometry_dimension: The :class:`GeometryDimension` (ignored for a footprint load, which
            is always 2D).
        child_inherits_parent_attributes: When ``True``, a child object's rows also carry its
            parent's attributes (the parent's values do not overwrite the child's own).
    """

    strategy: LoadStrategy = LoadStrategy.SEMANTIC_SURFACES
    lod: str | None = None
    geometry_dimension: GeometryDimension = GeometryDimension.XYZ
    child_inherits_parent_attributes: bool = False

    @classmethod
    def coerce(
        cls,
        strategy: "str | LoadStrategy" = LoadStrategy.SEMANTIC_SURFACES,
        lod: Any = None,
        geometry_dimension: "str | GeometryDimension" = GeometryDimension.XYZ,
        child_inherits_parent_attributes: bool = False,
    ) -> "ReadOptions":
        """Build a :class:`ReadOptions`, accepting enums or their plain-string values.

        Raises:
            ValueError: If ``strategy`` or ``geometry_dimension`` is not a recognized value.

        Examples:
            >>> ReadOptions.coerce(strategy="footprint").strategy is LoadStrategy.FOOTPRINT
            True
            >>> ReadOptions.coerce(geometry_dimension="xy").geometry_dimension
            <GeometryDimension.XY: 'xy'>
            >>> ReadOptions.coerce(lod=2).lod
            '2'
        """
        try:
            resolved_strategy = LoadStrategy(strategy)
        except ValueError as error:
            valid = ", ".join(repr(member.value) for member in LoadStrategy)
            raise ValueError(
                f"unknown strategy {strategy!r}; expected one of {valid}"
            ) from error
        try:
            resolved_dimension = GeometryDimension(geometry_dimension)
        except ValueError as error:
            valid = ", ".join(repr(member.value) for member in GeometryDimension)
            raise ValueError(
                f"unknown geometry_dimension {geometry_dimension!r}; expected one of {valid}"
            ) from error
        return cls(
            strategy=resolved_strategy,
            lod=None if lod is None else str(lod),
            geometry_dimension=resolved_dimension,
            child_inherits_parent_attributes=bool(child_inherits_parent_attributes),
        )
