# Reference

The API surface of `pyramids_citymodels`, generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

What this package publishes is **drivers**, not an API you call directly. A driver is a
`VectorDriver` class registered with `pyramids-core` through the `pyramids.vector_drivers`
entry-point group (the group `FeatureCollection.read_file` consults — city models are vector
features), so installing this package is the whole integration: `pyramids` then dispatches a read
of a city-model format it does not handle itself to the driver that claims it.

Each driver has its own page listing its suffixes, options and documented limitations:

| Driver | Format | Page |
|---|---|---|
| `CityJsonDriver` | CityJSON | [Drivers](drivers.md) |
| `CityGmlDriver` | CityGML LOD2 | [Drivers](drivers.md) |

See the [Drivers](drivers.md) page for the `driver=` / `read_file` usage and the read options, and
the [example notebook](../examples/city-models.ipynb) for a runnable walkthrough.

The package root deliberately re-exports nothing (`__all__` is empty): it carries `__version__` and
the import contract that activates the vendored osgeo bindings. Driver classes are reached by their
module path — `pyramids_citymodels.drivers.<module>.<Driver>` — which is also how their entry points
name them, though normal use goes through `pyramids` rather than importing them at all.

For the GIS data model itself — the raster `Dataset`, the vector `FeatureCollection`, reprojection,
overlays and conversion — see the
[pyramids reference](https://serapeum-org.github.io/pyramids/latest/). The split between the two is
[ADR-0003](../adr/0003-file-format-drivers.md).

::: pyramids_citymodels
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2
