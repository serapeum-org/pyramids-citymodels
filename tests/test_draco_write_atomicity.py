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
from pyramids_citymodels.writers.threedtiles import _discard_partial_tileset
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
        model = _model(5)
        with pytest.raises(UnsupportedGeometryError, match="Draco"):
            model.to_3dtiles(out, compression="draco", **mode)
        leftover = sorted(str(p.relative_to(out)) for p in out.rglob("*"))
        assert leftover == [], (
            f"partial tileset left behind (files or dirs): {leftover}"
        )

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

    def test_failed_rewrite_into_populated_dir_is_not_transactional(
        self, monkeypatch
    ) -> None:
        """Documented limitation: atomicity covers a fresh dir, not a re-write into a populated one.

        Test scenario:
            After a successful Draco write populates the directory, a failed re-write (encode fails after
            two tiles) leaves the **prior** ``tileset.json`` in place — it is written last, so the failed
            run never reaches it — and does not roll back the pre-existing content leaves (they are in the
            snapshot, so cleanup refuses to delete them). This pins the non-transactional re-write
            behaviour the docstring documents.
        """
        out = Path(tempfile.mkdtemp())
        _model(4).to_3dtiles(
            out, content="b3dm", compression="draco", max_objects_per_tile=1
        )
        original_tileset = (out / "tileset.json").read_bytes()
        preexisting_leaves = {p for p in out.rglob("*") if p.is_file()}
        _fail_encode_after(monkeypatch, 2)
        rerun = _model(4)
        with pytest.raises(UnsupportedGeometryError, match="Draco"):
            rerun.to_3dtiles(
                out, content="b3dm", compression="draco", max_objects_per_tile=1
            )
        assert (out / "tileset.json").read_bytes() == original_tileset, (
            "prior tileset.json left in place"
        )
        assert all(p.exists() for p in preexisting_leaves), (
            "every pre-existing leaf survives — a re-write rolls back nothing from the snapshot"
        )

    def test_discard_removes_added_tree_but_keeps_preexisting(self, tmp_path) -> None:
        """Rollback removes added files and their now-empty added dirs, but keeps pre-existing entries.

        Test scenario:
            A directory with a pre-existing file is snapshotted; a simulated write then adds a file in a
            **new** directory and a top-level leaf. ``_discard_partial_tileset`` removes both added files
            and ``rmdir``s the new (now-empty) directory, while the pre-existing file — and its
            directory, which is not in the added set — survive untouched.
        """
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "keep.txt").write_text("keep", encoding="utf-8")
        before = set(tmp_path.rglob("*"))
        (tmp_path / "added_dir").mkdir()
        (tmp_path / "added_dir" / "leaf.b3dm").write_bytes(b"x")
        (tmp_path / "content_0.b3dm").write_bytes(b"y")
        _discard_partial_tileset(tmp_path, before)
        assert not (tmp_path / "content_0.b3dm").exists(), (
            "added top-level leaf removed"
        )
        assert not (tmp_path / "added_dir").exists(), "added dir removed once emptied"
        assert (tmp_path / "sub" / "keep.txt").read_text(encoding="utf-8") == "keep", (
            "pre-existing kept"
        )

    def test_discard_swallows_removal_errors(self, tmp_path, monkeypatch) -> None:
        """A removal that fails (e.g. a read-only file) is swallowed so cleanup never masks the real error.

        Test scenario:
            With ``Path.unlink`` forced to raise ``OSError``, ``_discard_partial_tileset`` of an added
            leaf returns normally (the error is swallowed) rather than propagating and hiding the
            original write failure it is cleaning up after.
        """
        before = set(tmp_path.rglob("*"))
        (tmp_path / "content_0.b3dm").write_bytes(b"x")

        def boom(self):
            raise OSError("cannot remove")

        monkeypatch.setattr(Path, "unlink", boom)
        _discard_partial_tileset(tmp_path, before)  # must not raise
        assert (tmp_path / "content_0.b3dm").exists(), (
            "unlink was blocked, but cleanup did not propagate"
        )

    def test_discard_snapshot_failure_does_not_propagate(
        self, tmp_path, monkeypatch
    ) -> None:
        """A fault while walking the tree during rollback is swallowed, so it can't mask the real error.

        Test scenario:
            With ``Path.rglob`` forced to raise ``OSError`` (a traversal fault during rollback),
            ``_discard_partial_tileset`` returns normally rather than propagating — otherwise the cleanup
            error would replace the original encode failure that triggered the rollback.
        """

        def boom(self, *args, **kwargs):
            raise OSError("cannot walk the tree")

        monkeypatch.setattr(Path, "rglob", boom)
        _discard_partial_tileset(tmp_path, set())  # must not raise
