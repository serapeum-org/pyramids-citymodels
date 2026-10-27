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

Both drivers' `read` forward the same keyword options to the flattener:

- `strategy` — `"semantic_surfaces"` (default; one row per boundary surface) or `"footprint"`
  (one plan-view row per object).
- `lod` — the level of detail to select; `None` (default) takes the highest each object offers.
- `geometry_dimension` — `"xyz"` (default, keeps Z) or `"xy"` (drops Z). Ignored for a footprint.
- `child_inherits_parent_attributes` — when `True`, a `BuildingPart` row also carries its
  `Building`'s attributes.

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
