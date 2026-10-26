## Unreleased

### Feat

- CityJSON and CityGML LOD2 vector drivers: a shared city-model intermediate representation,
  parsers for both formats, an IR→FeatureCollection flattener with loading-strategy / LOD /
  3D-dimension options, and `CityJsonDriver` / `CityGmlDriver` registered into the
  `pyramids.vector_drivers` entry-point group (#4, #5, #6, #7, #8, #9, #10, #11)

## 0.1.0 (2026-10-25)

### Feat

- initial repository scaffold: packaging, pixi environments, CI, docs stack, and the
  `import pyramids` import contract; no city-model API published yet
