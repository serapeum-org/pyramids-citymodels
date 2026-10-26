"""Format drivers, one module per city-model format family.

Grouped by format family rather than by file suffix, because related encodings of one
standard share a parser (CityGML and its CityJSON sibling, for instance).

Each driver is a ``VectorDriver`` class registered into the ``pyramids.vector_drivers``
entry-point group. Keep this subpackage -- and each driver module -- light at import:
pyramids reads a driver's claims as class attributes without constructing it, so a driver's
heavy backend belongs inside its ``read`` method, not at module top, and the optional
dependency is named by the driver's ``extra`` attribute (the registry skips the driver
silently at discovery when that package is absent). An eager backend import here would
break the whole package import instead.
"""
