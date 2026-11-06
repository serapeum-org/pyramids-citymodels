# Drivers

`pyramids-citymodels` ships three readers for the 3D city-model exchange formats. Each turns a file
into a `pyramids` [`FeatureCollection`](https://serapeum-org.github.io/pyramids/latest/) — a
GeoDataFrame subclass — using one flat row model: a `surface_type` discriminator column and a
`parent_id` foreign key, with the object's attributes passed through.

| Driver | `name` | Formats (suffixes) | Backend |
|---|---|---|---|
| `CityJsonDriver` | `cityjson` | `.city.json`, `.cityjson`, `.json` (content-sniffed) | stdlib `json` |
| `CityGmlDriver` | `citygml` | `.gml`, `.citygml`, `.xml` (namespace-sniffed) | stdlib `xml.etree` |
| `ThreeDTilesDriver` | `3dtiles` | `tileset.json` (`.json`, content-sniffed), `.b3dm`, `.glb` | stdlib `struct` + `numpy` |

None declares a separate GDAL or any heavy backend: parsing is pure standard library (plus the
`numpy`/`shapely`/`geopandas` that `pyramids-core` already provides).

## Usage

A released `pyramids-core` (≥ 0.72.0) exposes the `pyramids.vector_drivers` registry, so a
city-model file is read through `pyramids` itself — the driver is selected by content, or named
explicitly:

```python
from pyramids.feature import FeatureCollection

# selected by content sniff (.json is shared by CityJSON, GeoJSON and a 3D Tiles tileset,
# so content decides which driver claims it — or None of them, falling through to OGR)
features = FeatureCollection.read_file("building.city.json")
tiles = FeatureCollection.read_file("tileset.json")
# or named explicitly
features = FeatureCollection.read_file("building.gml", driver="citygml")
```

The drivers can also be used directly (e.g. on an older core), with an identical result:

```python
from pyramids_citymodels.drivers.cityjson import CityJsonDriver

features = CityJsonDriver().read("building.city.json", strategy="semantic_surfaces")
```

### Read options

Both drivers' `read` accept the same keyword options. The first group is a **read-time subset**
applied by the reader (non-matching objects are skipped *before* their geometry is decoded, so a
city-scale file can be sliced without paying to parse what it discards); the rest are forwarded to
the flattener.

Subset (parse-time), bundled as an
[`ObjectFilter`][pyramids_citymodels.options.ObjectFilter]:

- `ids` — keep only objects with these ids (a string, or an iterable of strings).
- `types` — keep only objects of these types (e.g. `"Building"`, `"BuildingPart"`).
- `attributes` — keep only objects whose attributes satisfy each of these criteria. A value is
  either a scalar (matched by `==`) or a **CQL2-style predicate mapping** whose operators must all
  hold: `{"ge": 10}`, `{"gt": 0, "lt": 100}`, `{"in": ["residential", "commercial"]}`,
  `{"nin": [...]}`, `{"ne": x}`, `{"like": "resid%"}` (`%`/`*` = any run, `_`/`?` = one char), or
  `{"between": [lo, hi]}`. A comparison against a missing/incomparable value is a non-match, never
  an error.
- `bbox` — `(xmin, ymin, xmax, ymax)` in the **file's own CRS** (no reprojection here). A
  *conservative prefilter*: an object is skipped only when a **declared extent the file already
  carries** — a CityJSON per-object `geographicalExtent`, or a CityGML feature's own
  `gml:Envelope` — proves it lies wholly outside. An object with **no declared extent is kept**
  (proving it outside would mean decoding the geometry this is meant to skip), so the bbox result
  is a *superset* of the exact selection. Refine it with `FeatureCollection.clip_by_rect(...)`
  downstream for an exact clip. (3D Tiles instead prunes whole tile subtrees whose bounding volume
  misses the bbox, then filters the decoded features by their exact footprint extent. Its bbox is
  in the tileset's **stored** coordinate system — commonly ECEF metres, since the 3D Tiles reader
  does not reproject — not a lon/lat or projected CRS, so a wrong-frame bbox silently selects
  nothing.)
- `exclude` — the **negation**: ids/types/attributes whose match *drops* an object (a mapping of
  `ids`/`types`/`attributes`, an `ObjectFilter`, or a lone id / iterable of ids). "Keep X but not
  Y." Metadata only — a `bbox`/`sample` inside an `exclude` is ignored.
- `sample` — keep a **random subset** of the matches: a `float` fraction (`0.1` ≈ 10%) or an `int`
  maximum count. Reproducible when `seed` is given. For CityJSON/CityGML the draw is made over the
  objects that pass the metadata and bbox tests, *before* any geometry is decoded.
- `seed` — seed for the `sample` draw, for a repeatable subset.

An object is kept only when it satisfies **every** keep criterion, is **not** matched by `exclude`,
and survives the `sample`; filtering is per object and does not pull in relatives (a kept
`BuildingPart` whose `Building` was dropped keeps a dangling `parent_id`, and in CityGML a dropped
feature's surfaces roll up to the nearest kept ancestor). Filtering a `FeatureCollection` *after*
reading is a GIS operation and stays in `pyramids-core`; these keywords are only the parse-time
skip. The same subset is available on the IR entry point as
`CityModel.from_file(path, ObjectFilter.coerce(...))`.

Projection (flatten-time):

- `strategy` — `"semantic_surfaces"` (default; one row per boundary surface) or `"footprint"`
  (one plan-view row per object).
- `lod` — the level of detail to select; `None` (default) takes the highest each object offers.
- `geometry_dimension` — `"xyz"` (default, keeps Z) or `"xy"` (drops Z). Ignored for a footprint.
- `child_inherits_parent_attributes` — when `True`, a `BuildingPart` row also carries its
  `Building`'s attributes.

```python
# Read only two buildings, as footprints, straight into a FeatureCollection:
features = CityJsonDriver().read(
    "city.city.json", ids=["B1", "B2"], strategy="footprint"
)
```

A surface geometry is a planar polygon in 3D: a **vertical wall is non-areal in the XY plane** by
design, so it is valid-but-zero-area in `semantic_surfaces` and excluded from a `footprint`. An
unsupported geometry type (a point/line aggregate) raises
[`UnsupportedGeometryError`][pyramids_citymodels.errors.UnsupportedGeometryError].

### 3D Tiles scope

3D Tiles is a renderer-oriented format, so this driver surfaces only what maps cleanly onto a
`FeatureCollection` — a building's **footprint plus its batch-table attributes**. The deliberate
limits of this first reader (all documented, none silent):

- **Footprints, not meshes.** Each feature's geometry is the 2D convex-hull footprint of its
  vertices; the full textured mesh is left to a renderer (Digital-Earth's 3-D layer).
- **No ECEF→geographic projection.** Tile and node transforms are composed and `RTC_CENTER` is
  applied, so footprints are internally consistent, but a tileset's stored (usually ECEF) coordinates
  are not re-projected — the `FeatureCollection` carries `crs=None`.
- **Uncompressed glTF only.** A Draco- or meshopt-compressed tile raises `UnsupportedGeometryError`;
  a compressed-mesh backend is a documented follow-up ([#12](https://github.com/serapeum-org/pyramids-citymodels/issues/12)).
- **`pnts` is routed out.** Point-cloud tiles belong to the `pointclouds` package and are skipped
  with a warning, not mis-read as meshes.

## Writing and conversion

The read half turns a file into a `FeatureCollection` (a deliberately lossy, flat projection). The
**write** half is different: it serializes straight from the `CityModel` intermediate representation
(IR), never from `FeatureCollection`, so the object hierarchy, multi-LOD geometries, semantics,
appearances and metadata are preserved. This is the decision recorded in
[ADR-0005](../adr/0005-city-model-writers.md); the writers live as methods on `CityModel`:

```python
from pyramids_citymodels.ir import CityModel

# read with the symmetric entry point (reader chosen by content/suffix)
model = CityModel.from_file("building.gml")

# write — the format is chosen by the output suffix
model.to_file("building.city.json")  # CityJSON 2.0
model.to_file("building.city.jsonl")  # CityJSON Text Sequence (one feature per line)
model.to_file("building.gml")  # CityGML 2.0
model.to_file("building.fg.json")  # JSON-FG (OGC Features & Geometries JSON)
model.to_file("building.obj")  # Wavefront OBJ (plain mesh)
model.to_file("building.gltf")  # glTF 2.0 (plain mesh)

# or call a format method directly; a None path returns the document instead of writing
document = model.to_cityjson()  # -> dict
```

Format conversion is therefore a single IR→IR expression that never touches the flattened form:

```python
CityModel.from_file("building.gml").to_file("building.city.json")  # CityGML → CityJSON
```

| Method | Format | Round-trips? |
|---|---|---|
| `to_cityjson` | CityJSON 2.0 | Yes — `parse → write → parse` reproduces the model (the vertex pool is rebuilt and re-indexed from the IR). Geometry, semantics, metadata, CRS, version **and appearance** round-trip: the appearance pool (`materials`, `textures` and the `vertices-texture` UV pool) is carried verbatim and the per-geometry `material`/`texture` value maps re-emitted beside it. |
| `to_cityjsonseq` | CityJSON Text Sequence (`.city.jsonl`) | Yes, via CityJSON — a header line then one `CityJSONFeature` per top-level object. |
| `to_citygml` | CityGML (2.0; a 1.0/2.0 source keeps its version) | Partially — the surfaces of `MultiSurface` and `Solid`/`MultiSolid` geometry (with their semantic types), object ids/types, the parent/child hierarchy and the CRS round-trip; a `1.0`/`2.0` source also keeps its version, while a `3.0`/unknown source is emitted as `2.0`. A `Solid`/`MultiSolid` keeps its surfaces but comes back as a `MultiSurface` (the reader rebuilds only `MultiSurface`), the `gml:LinearRing` round-trip is exact only for already-closed (spec-conformant) source rings, attribute `uom` units and surface `gml:id`s are lost, and a non-`Building`/`BuildingPart` object (a `Bridge`, `CityFurniture`, … that the reader now reads) is written with a warning under the `bldg:` namespace — **not schema-valid** for those thematic types, so a CityGML→CityGML round-trip of a non-building object is best-effort (correct per-type namespaces are a follow-up); its geometry/attributes still convert cleanly to CityJSON/JSON-FG/mesh. |
| `to_jsonfg` | JSON-FG | No — a one-way GIS-exchange projection (3D geometry in each feature's `place`); there is no JSON-FG reader. |
| `to_obj` | Wavefront OBJ | No — a one-way plain-geometry mesh export: each surface's exterior ring becomes one `f` face over a shared vertex pool, objects grouped under `o` markers. Interior rings (holes) are dropped and no appearance is written. |
| `to_gltf` | glTF 2.0 | No — a one-way plain-geometry mesh export: exterior rings are fan-triangulated into one positions-only mesh, embedded as a base64 buffer, shifted by the model's per-axis minimum (recorded as the node `translation` for float32 precision). Exterior rings only; no normals/appearance. |

Conversion coverage is set by the readers. The **CityGML reader** reads the common top-level city
objects — buildings, bridges, tunnels, city furniture, vegetation, water bodies, land use, relief
and transportation — taking each feature's geometry from its semantic boundary surfaces (LOD 2)
and/or a `lodX` container it holds directly (LOD read from the container), and reads
`app:X3DMaterial` appearances into the model's material pool; CityGML textures
(`ParameterizedTexture`) are a follow-up.

## Writing API

::: pyramids_citymodels.writers.cityjson
    options:
        show_root_heading: true
        heading_level: 3
        members: ["write_cityjson", "write_cityjsonseq"]

::: pyramids_citymodels.writers.citygml
    options:
        show_root_heading: true
        heading_level: 3
        members: ["write_citygml"]

::: pyramids_citymodels.writers.jsonfg
    options:
        show_root_heading: true
        heading_level: 3
        members: ["write_jsonfg"]

::: pyramids_citymodels.writers.obj
    options:
        show_root_heading: true
        heading_level: 3
        members: ["write_obj"]

::: pyramids_citymodels.writers.gltf
    options:
        show_root_heading: true
        heading_level: 3
        members: ["write_gltf"]

## Driver classes

::: pyramids_citymodels.drivers.cityjson
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids_citymodels.drivers.citygml
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids_citymodels.drivers.threedtiles
    options:
        show_root_heading: true
        heading_level: 3

## Reading API

The functions the drivers build on — useful when you want the intermediate representation or the
flattener directly.

::: pyramids_citymodels.readers.cityjson
    options:
        show_root_heading: true
        heading_level: 3
        members: ["parse_cityjson", "looks_like_cityjson"]

::: pyramids_citymodels.readers.citygml
    options:
        show_root_heading: true
        heading_level: 3
        members: ["parse_citygml", "looks_like_citygml"]

::: pyramids_citymodels.readers.threedtiles
    options:
        show_root_heading: true
        heading_level: 3
        members: ["parse_3dtiles", "looks_like_3dtiles"]

::: pyramids_citymodels.flatten
    options:
        show_root_heading: true
        heading_level: 3
        members: ["to_feature_collection"]

::: pyramids_citymodels.options
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids_citymodels.errors
    options:
        show_root_heading: true
        heading_level: 3
