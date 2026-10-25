"""Shared pytest fixtures for the pyramids-citymodels test suite.

The `import pyramids` below is order-critical and must stay the first import in
this module. The comment on that line carries the rationale.
"""

# Import pyramids FIRST — before anything else, and keep it first.
# The osgeo bindings live inside the pyramids wheel; importing pyramids is what
# puts `_vendor/osgeo` on sys.path. Test modules that do `from osgeo import gdal`
# at module scope, before they import pyramids_citymodels, rely on this line to
# make those imports resolve. Nothing installs a top-level osgeo, so deleting this
# line, or sorting it below an osgeo import, breaks collection for the whole
# directory.
import pyramids  # noqa: F401  # isort:skip
