"""3D city-model file-format drivers for the pyramids stack.

`pyramids-citymodels` reads and writes the file formats of semantic 3D city models —
CityGML, CityJSON and the related interchange formats — as `pyramids` objects.
`pyramids-core` owns the GIS data model (the raster `Dataset`, the vector
`FeatureCollection`, the datacube), its GDAL/OGR-backed I/O, and every operation that
follows once features are in hand — reprojection, overlays, spatial joins, conversion
and spatial indexing. What it does not do is open the city-model exchange formats;
that is this package's job, and its only job.

Drivers are discovered by `pyramids-core` through the `pyramids.citymodel_drivers`
entry-point group, so a read of a city-model file resolves to a driver here once this
package is installed. Dependency-backed drivers sit behind extras and are skipped at
discovery when their extra is absent. See ADR-0003 for the scope decision.

The package root deliberately re-exports nothing: it carries `__version__` and the
import contract that activates the vendored osgeo bindings, and that is all. Installing
the package is the whole integration, so nothing here needs importing. A driver class is
still reachable by its module path for the rare case of wanting one directly.

Examples:
    - The root carries the installed version and nothing else. `__all__` is empty
      because there is deliberately nothing to re-export:

        >>> import pyramids_citymodels
        >>> pyramids_citymodels.__version__.split(".")[0].isdigit()
        True
        >>> pyramids_citymodels.__all__
        []
"""

# Import pyramids FIRST — before anything else, and keep it first. pyramids-core
# bundles GDAL's osgeo SWIG bindings and native libs inside its platform wheel
# and activates them on import (puts `_vendor/osgeo` on sys.path, sets
# GDAL_DATA / PROJ_DATA, registers the Windows DLL directory). Any module here
# that does `from osgeo import gdal` at load time needs this line to have run
# first, or that import fails wherever no separate GDAL is installed.
import pyramids  # noqa: F401

try:
    from importlib.metadata import PackageNotFoundError  # type: ignore
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError  # type: ignore
    from importlib_metadata import version


# The distribution name, not `__name__`: the import name is `pyramids_citymodels`
# and the distribution is `pyramids-citymodels`. importlib.metadata normalizes `_`
# to `-` so either spelling resolves today, but the metadata lookup is a
# distribution lookup — spell it the way the metadata does.
try:
    __version__ = version("pyramids-citymodels")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


__all__: list[str] = []
