"""Scaffold-level checks: the package imports, honours its import contract, and
exposes a version and an empty public surface.

These are the only tests the scaffold ships. They are replaced/extended by each
driver's own suite as drivers land; until then they keep `pixi run main` and the
pure-wheel test meaningful rather than empty.
"""

import pyramids_citymodels


def test_version_is_a_string() -> None:
    """The package exposes a non-empty ``__version__`` string."""
    assert isinstance(pyramids_citymodels.__version__, str)
    assert pyramids_citymodels.__version__


def test_public_surface_is_empty() -> None:
    """The root re-exports nothing: ``__all__`` is deliberately empty (ADR-0003)."""
    assert pyramids_citymodels.__all__ == []


def test_pyramids_imported_first() -> None:
    """Importing the package activates the vendored osgeo via ``import pyramids``.

    The package ``__init__`` imports ``pyramids`` before anything else, so once
    ``pyramids_citymodels`` is imported, ``pyramids`` is present in ``sys.modules``.
    """
    import sys

    assert "pyramids" in sys.modules
