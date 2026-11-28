"""Atomicity of Draco multi-tile 3D Tiles writes (issue #86).

`write_3dtiles(..., compression="draco")` emits content one file per leaf and writes `tileset.json` last.
If `DracoPy.encode` fails on a *later* tile, the earlier leaves must not be left orphaned with no
`tileset.json` — the write cleans up its partial output and re-raises. These tests inject a per-tile
encode failure (no natural mesh triggers one) and assert the output directory is left with no partial
tileset, for both content formats and the explicit / implicit multi-tile modes. The module needs
`DracoPy`, so it is marked `draco`.
"""

import tempfile
from pathlib import Path

import pyramids  # noqa: F401  # isort:skip  (keep first — see conftest)

import DracoPy
import pytest

from pyramids_citymodels.errors import UnsupportedGeometryError
from tests._draco_fixtures import model as _model

pytestmark = pytest.mark.draco


def _fail_encode_after(monkeypatch, n: int) -> None:
    """Patch ``DracoPy.encode`` to succeed for the first ``n`` calls, then raise (a later-tile failure)."""
    real = DracoPy.encode
    calls = {"c": 0}

    def flaky(*args, **kwargs):
        calls["c"] += 1
        if calls["c"] > n:
            raise RuntimeError("injected Draco encode failure on a later tile")
        return real(*args, **kwargs)

    monkeypatch.setattr(DracoPy, "encode", flaky)


class TestDracoWriteAtomicity:
    """A per-tile Draco encode failure mid-write leaves no orphaned content and no tileset.json."""

    @pytest.mark.parametrize(
        "mode",
        [
            {"content": "b3dm", "max_objects_per_tile": 1},
            {"content": "gltf", "max_objects_per_tile": 1},
            {"content": "b3dm", "implicit_tiling": True, "max_objects_per_tile": 1},
        ],
        ids=["b3dm-explicit", "gltf-explicit", "b3dm-implicit"],
    )
    def test_later_tile_failure_cleans_up(self, monkeypatch, mode: dict) -> None:
        """A failure on a later tile removes every already-written leaf and writes no tileset.json.

        Args:
            monkeypatch: pytest fixture used to inject the encode failure.
            mode: the ``to_3dtiles`` keywords (content format + multi-tile mode) under test.

        Test scenario:
            With ``DracoPy.encode`` made to fail after the first two tiles, a 5-object multi-tile write
            raises ``UnsupportedGeometryError`` and leaves the output directory empty — no
            ``content_*``/``content`` leaves, no ``subtrees``, no ``tileset.json``.
        """
        _fail_encode_after(monkeypatch, 2)
        out = Path(tempfile.mkdtemp())
        with pytest.raises(UnsupportedGeometryError, match="Draco"):
            _model(5).to_3dtiles(out, compression="draco", **mode)
        leftover = sorted(p.name for p in out.rglob("*") if p.is_file())
        assert leftover == [], f"partial tileset left behind: {leftover}"

    def test_successful_draco_write_is_unaffected(self) -> None:
        """A successful Draco multi-tile write still produces tileset.json + content (no regression).

        Test scenario:
            Without any injected failure, a 4-object b3dm Draco multi-tile write produces a
            ``tileset.json`` and at least one ``.b3dm`` leaf — the cleanup path does not touch the
            success path.
        """
        out = Path(tempfile.mkdtemp())
        _model(4).to_3dtiles(
            out, content="b3dm", compression="draco", max_objects_per_tile=1
        )
        assert (out / "tileset.json").exists(), "tileset.json written on success"
        assert any(out.rglob("*.b3dm")), "at least one b3dm leaf written on success"
