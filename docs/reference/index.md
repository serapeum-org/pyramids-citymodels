# Reference

The API surface of `pyramids_citymodels`, generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

What this package publishes is **drivers**, not an API you call directly. A driver is a
`VectorDriver` class registered with `pyramids-core` through the `pyramids.vector_drivers`
entry-point group (the group `FeatureCollection.read_file` consults — city models are vector
features), so installing this package is the whole integration: `pyramids` then dispatches a read
of a city-model format it does not handle itself to the driver that claims it.

The scaffold registers no drivers yet. As each one lands it is listed here with its own page — its
suffixes, capabilities and documented limitations — and the table below fills in:

| Driver | Format | Page |
|---|---|---|
| _(none yet)_ | | |

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
