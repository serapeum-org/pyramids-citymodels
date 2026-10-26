[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/serapeum-org/pyramids-citymodels/branch/main/graph/badge.svg)](https://codecov.io/gh/serapeum-org/pyramids-citymodels)

# pyramids-citymodels

**pyramids-citymodels** reads and writes the file formats of semantic 3D city models —
CityGML, CityJSON and more — as [pyramids](https://github.com/serapeum-org/pyramids) objects.
`pyramids-core` already ships the GIS data model and the vector/raster operations that follow once
features are in hand; what it does not do is open the city-model exchange formats. That is this
package's job, and its only job.

What `pyramids-core` owns today, and this package therefore does not reimplement:

- **The GIS data model** — the raster `Dataset`, the vector `FeatureCollection`, the datacube layer,
  and their GDAL/OGR-backed read/write.
- **Vector and geometry operations on features already in hand** — reprojection, spatial joins,
  overlays, filtering, geometry predicates and spatial indexing.
- **Conversion** — vector ↔ raster, feature extraction, and the GeoPackage / GeoJSON / shapefile
  round-trips GDAL/OGR provides.

What this package adds — **file-format drivers for 3D city models, and nothing else**:

- **Readers** for the native exchange formats of 3D city modelling: CityGML, CityJSON, and the
  related tiled/interchange formats as they land.
- **Writers** for the formats that support a faithful round-trip.
- **Discovery** through the `pyramids.vector_drivers` entry-point group, so a read of a
  city-model file resolves to the driver that claims it once this package is installed.

Every GIS *operation* belongs upstream in `pyramids-core`. A driver parses a file and builds
`pyramids` objects; it implements no GIS, no reprojection and no analysis.

> **Status** — `0.1.0` is the repository scaffold: packaging, CI, docs and the import contract. No
> city-model API is published yet. Follow the [change log](docs/change-log.md) for what lands.

> **Naming note** — the distribution name is `pyramids-citymodels` (with hyphen), the Python import
> name is `pyramids_citymodels` (with underscore).

## Installation

The package is not published yet. Install it from source with
[Pixi](https://pixi.sh/latest/#installation):

```bash
git clone https://github.com/serapeum-org/pyramids-citymodels.git
cd pyramids-citymodels
pixi install -e dev      # creates the dev environment
pixi shell -e dev
```

GDAL is **not** a dependency. The `pyramids-core` wheel vendors its own osgeo bindings, and
`import pyramids` is what puts them on `sys.path` — which is why `import pyramids` is the first
import in every module here.

Supported Python: **3.11–3.14**.

## Project layout

```
src/pyramids_citymodels/   — the package (import name: pyramids_citymodels)
tests/                     — pytest suite
docs/                      — MkDocs sources (MkDocs Material + mkdocstrings)
docs/examples/             — example notebooks, validated by `pixi run notebooks`
examples/data/             — data the notebooks read, via notebook-relative paths
```

## Documentation

- Live site: <https://serapeum-org.github.io/pyramids-citymodels/latest/>
- Local preview: `pixi run -e docs mkdocs serve`

## Development

This repository uses [Pixi](https://pixi.sh/) for environment management.

```bash
pixi run main          # main test suite
pixi run test-all      # everything, with a terminal coverage report
pixi run notebooks     # validate example notebooks
pre-commit run --all-files
```

Branches and commits follow [Conventional Commits](https://www.conventionalcommits.org/): branch
`<type>/<short-description>`, PR title `<type>: <imperative summary>`. Releases are driven by
commitizen (`pixi run cz-bump`) and the shared `serapeum-org/github-actions` workflows.

## License

GNU General Public License v3 — see [`LICENSE`](./LICENSE).
