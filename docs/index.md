# pyramids-citymodels

Reads and writes the file formats of semantic 3D city models — CityGML, CityJSON and more — as
[pyramids](https://github.com/serapeum-org/pyramids) objects. `pyramids-core` already ships the GIS
data model and the operations that follow once features are in hand; what it does not do is open the
city-model exchange formats. That is this package's only job.

## The layer boundary

The split is the point of the package, so it is worth stating precisely. `pyramids-core` owns the
GIS *data model* and everything that follows from features already in hand:

| Concern | Lives in |
|---|---|
| The raster `Dataset`, the vector `FeatureCollection`, the datacube | `pyramids` |
| GDAL/OGR-backed read / write of the formats GDAL already handles | `pyramids` |
| Reprojection, overlays, spatial joins, filtering, geometry predicates | `pyramids` |
| Vector ↔ raster conversion, feature extraction, spatial indexing | `pyramids` |

**pyramids-citymodels** adds the one thing that is missing from that list — **the ability to open
3D city-model exchange formats**:

| Concern | Lives in |
|---|---|
| Reading CityGML, CityJSON, and related interchange formats | `pyramids_citymodels.drivers` |
| Writing the formats that support a faithful round-trip | `pyramids_citymodels.drivers` |
| Driver discovery via the `pyramids.vector_drivers` entry points | `pyramids-core` reads, this package registers |

Anything that belongs on the left-hand side of the first table is a change to `pyramids-core`, not a
feature of this package. Report it upstream rather than duplicating it here.

## Status

`0.1.0` is the repository scaffold — packaging, CI, docs, and the import contract that pins
`import pyramids` first so the vendored osgeo bindings are active before anything else loads. The
public surface is empty; `pyramids_citymodels.__version__` is all it exports.

```python
>>> import pyramids_citymodels
>>> pyramids_citymodels.__version__
'0.1.0'
```

## Where to next

- [Installation](installation.md) — Pixi environments and what each one carries.
- [Reference](reference/index.md) — the API surface, as it lands.
- [Change log](change-log.md)
- [Architecture Decision Records](adr/index.md) — including the scope boundary above, as
  [ADR-0003](adr/0003-file-format-drivers.md).

Source: <https://github.com/serapeum-org/pyramids-citymodels>
