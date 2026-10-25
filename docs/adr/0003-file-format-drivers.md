# ADR-0003: pyramids-citymodels is a file-format driver package

- Status: Accepted
- Date: 2026-10-25

## Context

`pyramids-core` already ships the GIS data model — the raster `Dataset`, the vector
`FeatureCollection`, the datacube — their GDAL/OGR-backed I/O, and every operation that follows once
features are in hand: reprojection, overlays, spatial joins, filtering, geometry predicates, vector
↔ raster conversion and spatial indexing. This repository exists because the semantic 3D city-model
exchange formats are missing from what the core opens, not because the GIS model needs a second
implementation.

Without a stated boundary, the obvious failure mode is re-deriving geometry operations or re-wrapping
the vector I/O here, which gives the stack two feature models that drift apart.

## Decision

**`pyramids-citymodels` reads and writes the file formats of semantic 3D city models — CityGML,
CityJSON and the related interchange formats — as `pyramids` objects. It owns file-format drivers;
every GIS operation belongs upstream in `pyramids-core`.**

- Drivers are discovered by `pyramids-core` through the `pyramids.citymodel_drivers` **entry-point
  group**. This package depends on `pyramids-core`; `pyramids-core` never imports this package.
- Each driver is a class declaring its `name`, `long_name`, `suffixes` and a `capabilities` set, and
  implementing `can_open` plus the read/write methods the format supports. A driver whose optional
  extra is absent raises at driver-module import, and the registry skips it with a warning, so a bare
  install stays small and useful.
- Dependency-backed driver families live behind extras (one per family), added when the driver that
  needs one lands. Formats GDAL already reads (CityGML travels through GDAL's GML driver) use the
  vendored osgeo in the `pyramids-core` wheel rather than taking a new dependency.
- Test fixtures are hand-built minimal files, committed, and kept out of the built distribution
  (setuptools' sdist does not collect `tests/data/*`, and the wheel is built from `src/` only).
  Large vendor sample files are fetched in CI behind the `vendor_data` marker, never committed.

## The boundary

- `pyramids-core` remains the **single owner** of the GIS data model, its persistence, and every
  operation that follows from features already in hand. This package imports those types; it never
  redefines them.
- Public functions take and return `pyramids` types, so a result can go straight back into the rest
  of the stack and be written through pyramids without a conversion step.
- A gap or bug on the pyramids side of the boundary is reported and fixed upstream. A local
  workaround, if one is unavoidable, carries a comment naming the upstream issue and is removed when
  that lands.
- Plotting goes through cleopatra, reached via pyramids. This package declares no plotting code and
  no plotting dependency: a driver returns data, and the caller plots it.

## Consequences

- The `pyramids-core` floor is an API floor on the **driver registry**, not on the GIS model. Raising
  it is the normal way to pick up an upstream fix the drivers depend on.
- Semantic concerns a city-model format carries that `pyramids-core` does not yet model (rich LoD
  hierarchies, appearance/texture, semantic surfaces) are **upstream** work. A driver carries the
  value through without fabricating data rather than inventing a local model for it.
- Interoperability is free — anything this package produces is already a pyramids object that the
  rest of the stack reads.
