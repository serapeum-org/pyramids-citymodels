# Round-trip fidelity

City-model writers serialize from the `CityModel` IR, never from the flattened `FeatureCollection`
([ADR-0005](../adr/0005-city-model-writers.md)). A conversion is therefore `read into CityModel` +
`write from CityModel`, and this page is the contract for what that round-trip preserves. It is the
target the writers (the [#23](https://github.com/serapeum-org/pyramids-citymodels/issues/23) epic and
its children) are built against.

Two things are true of the IR and shape everything below:

- It is **format-neutral**. CityJSON and CityGML both parse into the same `CityModel`, so a field only
  survives a round-trip if the IR itself carries it — not if one reader happens to keep it.
- Its geometry is **dereferenced**. Boundaries are coordinate tuples, not indices into a shared pool,
  so there is no vertex table or quantization to preserve verbatim; the writer rebuilds one.

## CityJSON

CityJSON is a JSON encoding of (a subset of) the CityGML data model, and the IR is modelled on it, so
a CityJSON → IR → CityJSON round-trip is **value-lossless**: the object graph, geometry and coordinate
*values* survive, but the vertex table and `transform` are rebuilt, not round-tripped verbatim (see
below). What the IR carries through:

| Carried | Where |
|---|---|
| Object hierarchy (`parents` / `children`), per-object attributes | `CityObject` |
| Per-object, multi-LOD geometries with their nested boundaries and semantics | `CityObject.geometries`, `Geometry` |
| Semantic surfaces (type, attributes, parent/child), parallel `values` | `Geometry.semantics`, `Geometry.semantic_values` |
| Per-geometry appearance (`material` / `texture` maps) | `Geometry.appearance` |
| Document appearance pool, schema `extensions`, `metadata`, CRS, `version` | `CityModel` |

### The vertex pool is rebuilt, not retained

The reader applies the `transform` and dereferences every boundary index to a coordinate tuple, so the
flat `vertices` array and the `transform` are **not** stored on `CityModel`. A writer rebuilds them from
the coordinates rather than the IR retaining the source pool. This is a deliberate choice over keeping
an optional source pool + transform on the model:

- a CityGML-sourced model has **no** source pool to retain, so a rebuild path is needed regardless;
- a retained pool would **drift** from boundaries edited in memory, silently desynchronising;
- the IR stays format-neutral and minimal (the [ADR-0003](../adr/0003-file-format-drivers.md) boundary).

The rebuild lives on the types that own the data:

```python
# An empty pool whose transform is sized to the model's coordinate extent.
pool = model.vertex_pool(precision=3)
for city_object in model.objects.values():
    for geometry in city_object.geometries:
        # Each geometry's boundaries become vertex indices; the shared pool fills.
        indexed = geometry.index_into(pool)
# pool.vertices and pool.transform are the document-wide `vertices` + `transform` to write.
```

`CityModel.vertex_pool(precision=None)` returns a lossless pool (`transform` is `None`, coordinates are
pooled unchanged). A `precision` quantizes coordinates to integers through a CityJSON `transform` —
smaller files, coordinates preserved to within the quantization step. Either way the *vertex encoding*
is not round-tripped: a document read with a `transform` and integer `vertices` re-serialises, by
default, with float `vertices` and no `transform` (coordinate values unchanged). The one *exact* claim
is the in-memory inverse — `VertexPool.dereference` is the exact inverse of `Geometry.index_into` for a
lossless pool, `dereference(index_into(pool)) == boundaries`, which is the property the round-trip tests
assert.

## CityGML

A CityGML → IR → CityGML round-trip is **lossy by design**: the reader reconstructs the LOD2 *semantic*
model (`Building → BuildingPart → boundary surface → polygon`) that GDAL's GML driver does not expose,
and deliberately does not retain the full XML. The CityGML writer's target is faithfulness to that
semantic model, not byte equality.

**Preserved:**

- the `Building` / `BuildingPart` hierarchy and parent/child links;
- each feature's `gml:id`, as the `CityObject.id` (a stable `Building_<n>` is synthesised when absent);
- boundary-surface **semantic types** (`RoofSurface`, `WallSurface`, `GroundSurface`, …);
- polygon geometry — exterior ring first, then interior rings — with `srsDimension` honoured;
- building attributes lifted from leaf elements by local name, and `gen:*Attribute` name/value pairs,
  with the known-numeric ones (`measuredHeight`, `yearOfConstruction`, `storeys*`) coerced to numbers;
- the CRS (first `srsName`, as `EPSG:<code>`) and the CityGML version.

**Intentionally lost:**

- `gml:id` on **surfaces and polygons** (only the *feature* id is kept) — shared-geometry `xlink:href`
  graphs are resolved into inline coordinates, not preserved as references;
- **non-text attribute metadata**, e.g. a `measuredHeight` `uom` unit — only element text is captured;
- **repeated** single-valued properties — a feature with several `bldg:function` keeps only the last
  (IR attributes are a flat `dict`, not a multimap);
- the **source XML structure**, element ordering, comments, and namespace prefixes;
- geometry **representations other than the reconstructed LOD2 `MultiSurface`** (e.g. a `lod2Solid`'s
  shell structure, `lodNTerrainIntersection`, implicit geometries);
- CityGML **appearances / textures** (not parsed by the reader);
- **ADE** elements beyond their flattening to text attributes by local name.

A consumer that needs any of the lost detail should read the source with GDAL's GML driver in addition
to this package, rather than expecting the semantic IR to carry it.
