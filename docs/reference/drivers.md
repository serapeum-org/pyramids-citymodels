# Drivers

`pyramids-citymodels` ships two readers for the 3D city-model exchange formats. Both turn a file
into a `pyramids` [`FeatureCollection`](https://serapeum-org.github.io/pyramids/latest/) — a
GeoDataFrame subclass — using one flat row model: a `surface_type` discriminator column and a
`parent_id` foreign key, with the object's attributes passed through.

| Driver | `name` | Formats (suffixes) | Backend |
|---|---|---|---|
| `CityJsonDriver` | `cityjson` | `.city.json`, `.cityjson`, `.json` (content-sniffed) | stdlib `json` |
| `CityGmlDriver` | `citygml` | `.gml`, `.citygml`, `.xml` (namespace-sniffed) | stdlib `xml.etree` |

Neither declares a separate GDAL or any heavy backend: parsing is pure standard library on top of
the shapely/geopandas that `pyramids-core` already provides.

## Usage

Once a released `pyramids-core` exposes the `pyramids.vector_drivers` registry, a city-model file is
read through `pyramids` itself — the driver is selected by content, or named explicitly:

```python
from pyramids.feature import FeatureCollection

# selected by content sniff (.json is shared with GeoJSON, so content decides)
features = FeatureCollection.read_file("building.city.json")
# or named explicitly
features = FeatureCollection.read_file("building.gml", driver="citygml")
```

Until that registry ships (tracked in
[#19](https://github.com/serapeum-org/pyramids-citymodels/issues/19)), the drivers are used
directly — the integration is otherwise identical:

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

## Driver classes

::: pyramids_citymodels.drivers.cityjson
    options:
        show_root_heading: true
        heading_level: 3

::: pyramids_citymodels.drivers.citygml
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
