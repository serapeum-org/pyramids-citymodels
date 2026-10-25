"""Format drivers, one module per city-model format family.

Grouped by format family rather than by file suffix, because related encodings of one
standard share a parser (CityGML and its CityJSON sibling, for instance).

This module must stay free of eager driver imports. A missing extra would otherwise break
the whole package at import, instead of letting pyramids' registry skip that one driver
with a warning.
"""
