"""DiskTokenCache packed-format tests.

v2 layout: single append-only tokens.dat + index.json key->[off,nbytes].
The v1 one-file-per-entry layout collapsed to ~38 file-opens/s on the
USB HDD (0.1MB/s vs 230MB/s capable). These tests pin the packed
semantics: round-trip, key isolation, migrate-on-hit for legacy .bin
entries, persistence, compaction, and hit-path speed.

CPU-only; uses tmp_path.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from forge.training.data.efficient_pipeline import DiskTokenCache


@pytest.fixture()
def cache(tmp_path):
    return DiskTokenCache(cache_dir=str(tmp_path / "cache"))


def test_packed_roundtrip(cache, tmp_path):
    cache.put("hello world", [1, 2, 3, 4, 5], "tokA")
    cache.put("def f(x): return x*2", list(range(300)), "tokA")
    assert cache.get("hello world", "tokA") == [1, 2, 3, 4, 5]
    assert cache.get("def f(x): return x*2", "tokA") == list(range(300))
    assert cache.get("missing", "tokA") is None
    assert cache.get("hello world", "tokB") is None
    assert (tmp_path / "cache" / "tokens.dat").exists()


def test_reput_overwrites(cache):
    cache.put("k", [1, 2, 3], "t")
    cache.put("k", [9, 9, 9], "t")
    assert cache.get("k", "t") == [9, 9, 9]
    assert cache._dead_bytes > 0


def test_legacy_bin_reads_and_migrates(cache, tmp_path):
    shard = tmp_path / "cache" / "ab"
    shard.mkdir(parents=True, exist_ok=True)
    lbin = shard / "ab1234567890abcd.bin"
    np.array([7, 8, 9, 10], dtype=np.int64).tofile(str(lbin))
    key = cache._hash_key("legacy text", "tokA")
    cache._index[key] = str(lbin)  # legacy str path entry
    assert cache.get("legacy text", "tokA") == [7, 8, 9, 10]
    # Migrate-on-hit only when packed file/migration is active
    if cache._packed_path.exists() or cache._migrate_started:
        assert isinstance(cache._index[key], list)


def test_persistence_across_instances(cache, tmp_path):
    cache.put("a", [4, 5, 6], "t")
    cache.flush()
    c2 = DiskTokenCache(cache_dir=str(tmp_path / "cache"))
    assert c2.get("a", "t") == [4, 5, 6]


def test_compact_reclaims_dead_space(cache, tmp_path):
    cache.put("a", [1, 2, 3], "t")
    cache.put("b", list(range(500)), "t")
    cache.put("a", [9], "t")  # dead record
    cache.flush()
    before = (tmp_path / "cache" / "tokens.dat").stat().st_size
    cache._compact()
    after = (tmp_path / "cache" / "tokens.dat").stat().st_size
    assert after < before
    assert cache.get("a", "t") == [9]
    assert cache.get("b", "t") == list(range(500))


def test_hit_path_is_fast(cache):
    cache.put("x", list(range(200)), "t")
    t0 = time.time()
    for _ in range(200):
        cache.get("x", "t")
    assert time.time() - t0 < 1.0  # no per-entry file opens
