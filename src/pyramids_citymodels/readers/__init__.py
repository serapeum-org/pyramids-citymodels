"""Format parsers: one module per city-model format family, each producing the shared IR.

A reader's only job is file → :class:`~pyramids_citymodels.ir.CityModel`; it does no GIS and
builds no GeoDataFrame (that is :mod:`pyramids_citymodels.flatten`). Readers use only the
standard library (``json`` for CityJSON, ``xml.etree.ElementTree`` for CityGML), so importing
this subpackage pulls in no heavy backend.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)
