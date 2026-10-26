# ADR-0002: Type checking approach

- Status: Accepted
- Date: 2026-10-25

## Context
Static analysis improves reliability and developer experience, but the runtime stack underneath this
package (GDAL via the vendored osgeo bindings, and the GIS types it extends) is only partially typed,
so full strictness is not reachable from day one.

## Decision
- Adopt gradual typing with standard Python type hints; annotate public APIs first.
- mypy is configured in `[tool.mypy]` (checker target 3.12, `ignore_missing_imports` for the untyped
  osgeo/pyramids-core surface, and lenient gradual-typing flags — `disallow_untyped_defs = false`,
  `check_untyped_defs = true`) and is **enforced** by the `mypy` pre-commit hook, which runs in CI
  (`lint.yml`). Contributors run it locally with `pixi run -e dev mypy`.
- Prefer Google-style docstrings with type annotations to improve mkdocstrings output.
- Geometry and feature arrays are annotated to match how pyramids annotates the same types, so
  signatures line up across the boundary.

## Consequences
- Type hints are added progressively rather than in one sweep; the lenient flags keep the gate green
  while coverage grows, and strictness can be tightened per-module later without reworking existing
  annotations.
- The checker target (3.12) is deliberately above the `requires-python` floor (3.11) so mypy can
  parse numpy's PEP 695 stubs; runtime 3.11 support is covered by the py311 test-matrix leg.
