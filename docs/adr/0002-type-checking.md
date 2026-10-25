# ADR-0002: Type checking approach

- Status: Accepted
- Date: 2026-10-25

## Context
Static analysis improves reliability and developer experience, but the runtime stack underneath this
package (GDAL via the vendored osgeo bindings, and the GIS types it extends) is only partially typed,
so full strictness is not reachable from day one.

## Decision
- Adopt gradual typing with standard Python type hints; annotate public APIs first.
- mypy is installed in the `dev` group and may be run by contributors; it is not enforced in CI yet.
- Prefer Google-style docstrings with type annotations to improve mkdocstrings output.
- Geometry and feature arrays are annotated to match how pyramids annotates the same types, so
  signatures line up across the boundary.

## Consequences
- Type hints are added progressively rather than in one sweep.
- Strictness can be enabled per-module later, and a type-coverage report added to CI, without
  reworking existing annotations.
