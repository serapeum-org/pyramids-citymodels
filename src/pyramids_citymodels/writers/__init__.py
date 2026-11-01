"""Format serializers: one module per city-model format family, each writing the shared IR out.

A writer's only job is :class:`~pyramids_citymodels.ir.CityModel` → file (or an in-memory
document); it does no GIS and never touches ``FeatureCollection``, which is a deliberately lossy
one-way projection (:mod:`pyramids_citymodels.flatten`) that cannot round-trip. This mirror of
:mod:`pyramids_citymodels.readers` is the write half ADR-0005 places on the IR: the public entry
is a :class:`~pyramids_citymodels.ir.CityModel` method (``to_cityjson`` / ``to_citygml`` /
``to_file`` / …), each backed by a ``write_*`` orchestrator here.

Like the readers, these modules use only the standard library (``json`` for the CityJSON family,
``xml.etree.ElementTree`` for CityGML), so importing this subpackage pulls in no heavy backend and
needs no optional ``extra``.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — activates vendored osgeo; see __init__)
