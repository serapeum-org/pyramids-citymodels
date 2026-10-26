## Unreleased

### Feat

- CityJSON and CityGML LOD2 vector drivers: a shared city-model intermediate representation,
  parsers for both formats, an IR→FeatureCollection flattener with loading-strategy / LOD /
  3D-dimension options, and `CityJsonDriver` / `CityGmlDriver` registered into the
  `pyramids.vector_drivers` entry-point group (#4, #5, #6, #7, #8, #9, #10, #11)
- Extensibility hooks: unknown attributes (incl. CityGML ADEs) and CityJSON Extensions survive
  parsing, appearances/materials are carried through, the CityGML version is detected, and an
  unsupported geometry type raises a clear `UnsupportedGeometryError` (#13)
- A reference page for the drivers plus a runnable example notebook (#15); an end-to-end driver
  test suite against the fixtures (#14); the optional-dependency convention is documented and both
  drivers declare `extra = None` as stdlib-only readers (#12)

## 0.1.0 (2026-10-25)

### Feat

- initial repository scaffold: packaging, pixi environments, CI, docs stack, and the
  `import pyramids` import contract; no city-model API published yet
