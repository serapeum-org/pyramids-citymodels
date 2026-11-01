"""Regression tests for the Round-2 review findings on the city-model writers.

M1 is the one code defect: the CityGML writer kept version-specific namespaces in a mutated
module-global, which races under concurrent writes of different-version models.
"""

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import threading

from pyramids_citymodels.writers.citygml import write_citygml

from pyramids_citymodels.ir import CityModel, CityObject, Geometry, SemanticSurface


def _one_building(version: str) -> CityModel:
    """A minimal one-building model tagged with ``version`` (1.0 or 2.0)."""
    return CityModel(
        objects={
            "B": CityObject(
                id="B",
                type="Building",
                geometries=[
                    Geometry(
                        type="MultiSurface",
                        lod="2",
                        boundaries=[
                            [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
                        ],
                        semantics=[SemanticSurface(type="RoofSurface")],
                        semantic_values=[0],
                    )
                ],
            )
        },
        version=version,
    )


class TestM1NamespaceConcurrency:
    """M1 — concurrent writes of different-version models must not leak namespaces into each other."""

    def test_write_does_not_mutate_module_global_namespaces(self) -> None:
        """A non-2.0 write leaves the module-global namespace map untouched (the root-cause invariant).

        This is the deterministic form of the race: the fix removes the module-global mutation
        entirely, so a 1.0 write must not change ``citygml._NS`` (the buggy version reassigned it,
        which is what let concurrent writes corrupt each other).
        """
        from pyramids_citymodels.writers import citygml

        before = dict(citygml._NS)
        write_citygml(_one_building("1.0"))
        assert citygml._NS == before, "write_citygml mutated the module-global _NS"

    def test_concurrent_writes_keep_their_own_namespaces(self) -> None:
        """16 threads alternating 1.0/2.0 across 40 synchronized rounds each emit only their own URIs."""
        workers = 16
        rounds = 40
        barrier = threading.Barrier(workers)
        corrupted: list[str] = []

        def work(index: int) -> None:
            version = "1.0" if index % 2 == 0 else "2.0"
            other = "2.0" if version == "1.0" else "1.0"
            for _ in range(rounds):
                barrier.wait()
                try:
                    xml = write_citygml(_one_building(version))
                except Exception as error:  # noqa: BLE001 — record, don't abort the thread
                    corrupted.append(f"{version}: raised {error!r}")
                    continue
                if f"citygml/building/{version}" not in xml:
                    corrupted.append(f"{version}: missing own bldg namespace")
                if f"citygml/building/{other}" in xml:
                    corrupted.append(f"{version}: leaked {other} bldg namespace")

        threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not corrupted, (
            f"{len(corrupted)} corrupted documents, e.g. {corrupted[:3]}"
        )
