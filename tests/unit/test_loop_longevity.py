"""Longevity tests for the self-play loop's multi-hour guarantees.

CPU-only, no model weights. Covers:
  - archive rotation bounds demoted-checkpoint disk (sidecars included)
  - staging/durable path routing for promoted candidates
  - emergency SIGINT handler must not pin model+optimizer across epochs
  - honest decoding default (standard, not the no-op mtp_selfspec on Spark)
"""

import gc
import os
import signal
import sys
import time
import weakref

import pytest

from forge.self_play.infinite_loop import InfiniteSelfPlayLoop, LoopConfig


def _loop(tmp_path, **overrides) -> InfiniteSelfPlayLoop:
    cfg = LoopConfig(
        checkpoint_dir=str(tmp_path / "ckpts"),
        **overrides)
    return InfiniteSelfPlayLoop(checkpoint="base.safetensors", config=cfg)


def _fake_ckpt(path: str, size: int = 16) -> None:
    with open(path, "wb") as f:
        f.write(b"\x00" * size)
    for suffix in (".meta.json", ".train.pt"):
        with open(path + suffix, "w") as f:
            f.write("{}")


# ── archive rotation ─────────────────────────────────────────────────


class TestArchiveRotation:
    def test_rotation_keeps_newest_and_removes_sidecars(self, tmp_path):
        loop = _loop(tmp_path)
        arch = tmp_path / "archive"
        arch.mkdir()
        paths = []
        for i in range(5):
            p = arch / f"cand_{i}.safetensors"
            _fake_ckpt(str(p))
            # oldest -> newest mtimes
            ts = time.time() - (5 - i) * 60
            for suf in ("", ".meta.json", ".train.pt"):
                os.utime(str(p) + suf, (ts, ts))
            paths.append(str(p))

        loop._rotate_archive(str(arch), keep=2)

        remaining = sorted(f for f in os.listdir(arch)
                           if f.endswith(".safetensors"))
        assert remaining == ["cand_3.safetensors", "cand_4.safetensors"]
        for old in paths[:3]:
            for suf in ("", ".meta.json", ".train.pt"):
                assert not os.path.exists(old + suf)

    def test_demote_archives_and_rotates(self, tmp_path):
        loop = _loop(tmp_path, archive_keep=1)
        cdir = tmp_path / "ckpts"
        cdir.mkdir()
        for epoch in range(3):
            cand = cdir / f"ForgeLM_SP_SP{epoch}.safetensors"
            _fake_ckpt(str(cand))
            loop._maybe_promote(
                str(cand), {"passed": False, "winner": "BASE",
                            "candidate_quality": 0.4})
            # mtime ordering: make each archive successively newer
            time.sleep(0.02)
        arch = cdir / "archive"
        assert arch.is_dir()
        remaining = [f for f in os.listdir(arch) if f.endswith(".safetensors")]
        assert len(remaining) == 1  # archive_keep=1

    def test_demote_keep_zero_deletes_candidate(self, tmp_path):
        loop = _loop(tmp_path, archive_keep=0)
        cdir = tmp_path / "ckpts"
        cdir.mkdir()
        cand = cdir / "cand.safetensors"
        _fake_ckpt(str(cand))
        loop._maybe_promote(
            str(cand), {"passed": False, "winner": "BASE",
                        "candidate_quality": 0.4})
        assert not cand.exists()
        assert not (cdir / "archive").exists()


# ── staging paths ────────────────────────────────────────────────────


class TestStagingPaths:
    def test_epoch_checkpoint_uses_staging_dir(self, tmp_path):
        staging = tmp_path / "spool"
        staging.mkdir()
        loop = _loop(tmp_path, staging_dir=str(staging))
        path = loop._epoch_checkpoint_path(3)
        assert os.path.dirname(path) == os.path.normpath(str(staging))
        assert "SP3" in os.path.basename(path)

    def test_durable_path_maps_staged_to_checkpoint_dir(self, tmp_path):
        staging = tmp_path / "spool"
        staging.mkdir()
        loop = _loop(tmp_path, staging_dir=str(staging))
        cand = staging / "ForgeLM_SP_SP1.safetensors"
        cand.write_bytes(b"x")
        durable = loop._durable_path(str(cand))
        assert os.path.dirname(durable) == os.path.normpath(
            str(tmp_path / "ckpts"))
        # non-staged candidate passes through unchanged
        other = tmp_path / "ckpts" / "x.safetensors"
        assert loop._durable_path(str(other)) == str(other)

    def test_promote_moves_staged_candidate_with_sidecars(self, tmp_path):
        staging = tmp_path / "spool"
        staging.mkdir()
        cdir = tmp_path / "ckpts"
        cdir.mkdir()
        loop = _loop(tmp_path, staging_dir=str(staging))
        cand = staging / "ForgeLM_SP_SP2.safetensors"
        _fake_ckpt(str(cand))
        promoted = loop._maybe_promote(
            str(cand), {"passed": True, "winner": "CANDIDATE",
                        "candidate_quality": 0.9})
        assert promoted
        dest = cdir / "ForgeLM_SP_SP2.safetensors"
        assert dest.exists()
        assert (cdir / "ForgeLM_SP_SP2.safetensors.meta.json").exists()
        assert (cdir / "ForgeLM_SP_SP2.safetensors.train.pt").exists()
        assert not cand.exists()
        assert loop.best_checkpoint == os.path.normpath(str(dest))
        assert loop.best_score == 0.9


# ── SIGINT handler retention ─────────────────────────────────────────


class TestEmergencySigint:
    """sft_train's in-process SIGINT handler must not pin model+optimizer.

    The signal table is a GC root and every getsignal/signal pair links to
    the previously installed closure — an unbounded retention chain across
    self-play epochs (measured +4.3GB RSS / +0.2GB VRAM per epoch).
    """

    @pytest.mark.skipif(sys.platform != "win32" and
                        not hasattr(signal, "SIGINT"), reason="no SIGINT")
    def test_handler_uses_weakrefs_and_restores(self):
        from forge.training.runners.sft_train import _install_emergency_sigint

        class _Heavy:
            pass

        model = _Heavy()
        opt = _Heavy()
        prev = signal.getsignal(signal.SIGINT)
        try:
            old = _install_emergency_sigint(
                model, opt, "unused.safetensors", {"step": 0})
            assert old is not None
            # Installed handler must not prevent collection
            del model, opt
            gc.collect()
            installed = signal.getsignal(signal.SIGINT)
            cells = installed.__closure__ or ()
            heavy = [c.cell_contents for c in cells
                     if isinstance(getattr(c, "cell_contents", None), _Heavy)]
            assert heavy == []
            # Restore (what main() does on completion)
            signal.signal(signal.SIGINT, old)
            assert signal.getsignal(signal.SIGINT) is old
        finally:
            signal.signal(signal.SIGINT, prev)

    def test_dead_weakref_handler_skips_save(self):
        """A handler whose model died must no-op, not crash on save."""
        from forge.training.runners.sft_train import _install_emergency_sigint

        class _Heavy:
            pass

        prev = signal.getsignal(signal.SIGINT)
        try:
            m = _Heavy()
            old = _install_emergency_sigint(
                m, None, "x.safetensors", {"step": 1})
            del m
            gc.collect()
            handler = signal.getsignal(signal.SIGINT)
            assert handler is not prev and old is not None
            # model weakref is dead -> no emergency_save call, but the
            # handler still restores `old` and raises KeyboardInterrupt
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
            assert signal.getsignal(signal.SIGINT) is old
        finally:
            signal.signal(signal.SIGINT, prev)


# ── honest defaults ──────────────────────────────────────────────────


class TestHonestDefaults:
    def test_decoding_default_is_standard(self):
        """Spark has no MTP head — mtp_selfspec silently degraded to
        standard anyway; the default must say what it does."""
        assert LoopConfig().decoding == "standard"

    def test_promote_margin_positive(self):
        """Strict promotion needs a margin above single-question noise."""
        assert LoopConfig().promote_margin >= 0.01

    def test_archive_keep_default_bounded(self):
        assert LoopConfig().archive_keep == 8
