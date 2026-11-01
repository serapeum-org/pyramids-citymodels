# ADR-0005: city-model writers live on the IR, not on `FeatureCollection`

- Status: Accepted
- Date: 2026-11-01
- Amends: [ADR-0003](0003-file-format-drivers.md) (file-format driver package)

## Context

ADR-0003 states that `pyramids-citymodels` **reads and writes** the semantic 3D city-model exchange
formats as `pyramids` objects. The read half is built: `readers/{cityjson,citygml,threedtiles}.py`
parse a file into one shared intermediate representation (the `CityModel` IR in `ir.py`), and
`flatten.to_feature_collection` turns that IR into a `pyramids` `FeatureCollection`. The write half
does not exist yet — there is no serializer for any format, and all three drivers declare
`capabilities = frozenset({"read"})`.

Before any writer is built, one architectural question has to be settled, because it decides where
every writer lives and what it can faithfully produce: **what does a writer serialize from?** There
are two candidates, and they are not equivalent:

1. **From `FeatureCollection`** — e.g. wiring a write path into `pyramids.feature.FeatureCollection.to_file`,
   the mirror of how `read_file` dispatches to a reader. This is attractive because it would make
   writing symmetric with reading through the pyramids registry.
2. **From the `CityModel` IR** — e.g. `CityModel.to_cityjson()` / `CityModel.to_file(path)`, writing
   straight from the rich in-memory model.

The decisive fact is that **`to_feature_collection` is deliberately lossy**. It flattens the object
hierarchy to one-geometry-per-row footprint/surface features and keeps only the structural columns
plus attributes; it discards the object graph (parent/child links beyond a `parent_id`), the
per-object multi-LOD geometries, the nested boundary/semantics structure, document appearances and
extensions, and the model metadata. A `FeatureCollection` cannot be turned back into a faithful
CityGML or CityJSON file. Two further facts reinforce this: the `pyramids.vector_drivers` registry is
**read-only** (its `CAPABILITIES` are `{"read", "needs_auth"}`, its hook is `read_file`, and there is
no write dispatch upstream — adding one would be an upstream change akin to the read-dispatch work in
#19); and `FeatureCollection.to_file` already exists, backed by OGR/pyogrio, which writes flat vector
formats (GeoJSON, GeoPackage, …) and has no notion of the semantic city-model model.

## Decision

**City-model writers serialize from the `CityModel` IR, and are exposed as methods on `CityModel`.
They are never fed from, nor wired into, `FeatureCollection`.**

- The public write surface is on the IR: `CityModel.to_cityjson(...)`, `CityModel.to_citygml(...)`,
  and a suffix-dispatching `CityModel.to_file(path)`. A symmetric `CityModel.from_file(path)` reader
  entry point makes conversion a single expression that never touches the flattened form:

  ```python
  CityModel.from_file("building.gml").to_file("building.city.json")
  ```

- These methods follow the repository's design rule (behaviour lives on the type that owns the data —
  see CLAUDE.md): the serializer's public entry is a `CityModel` method, backed by a `writers/`
  subpackage that mirrors `readers/` (a public `write_*` orchestrating small private helpers) when
  the work is multi-step.
- `flatten.to_feature_collection` remains a **one-way** projection: IR → `FeatureCollection`, for GIS
  analysis and mapping. There is no `FeatureCollection` → IR path, and none should be added to serve
  writing.
- We do **not** add a write capability to `pyramids.vector_drivers`, and we do **not** route
  city-model writing through `FeatureCollection.to_file`. If a pyramids-level "write a city model"
  entry point is ever wanted, it is raised as a separate upstream decision; it would still serialize
  from the IR, not from a flattened frame.

## Consequences

- **Faithful round-trips are possible.** Writing from the IR preserves the hierarchy, multi-LOD
  geometries, semantics, appearances, extensions, metadata and CRS that the IR carries — provided the
  IR is made round-trip-complete (today it dereferences CityJSON `vertices` to coordinate tuples and
  drops the `transform`, which a CityJSON writer must rebuild; tracked as a prerequisite issue).
- **Conversion is IR→IR.** CityGML ⇄ CityJSON is `read into CityModel` + `write from CityModel`, so
  each new reader and each new writer composes with the others for free (a CityJSON writer immediately
  gives CityGML→CityJSON once the CityGML reader exists, and vice versa).
- **The driver registry stays read-only and simple.** No speculative write hook, no lossy write path
  to mislead callers.
- **Scope cost.** A CityGML writer is substantially harder than a CityJSON one (XML/GML, LODn
  `boundedBy` representations, `gml:id`s, ADE), so writers are sequenced CityJSON-first.
- This ADR amends ADR-0003's "reads and writes" clause with *where* writing lives; it does not change
  the scope boundary (no GIS here; `pyramids-core` still owns the data model and its persistence of
  flat vector formats).

The work this decision governs is tracked by the writers epic (#23) and its children (#24–#27).
