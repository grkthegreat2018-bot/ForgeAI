"""Parquet + ZSTD dataset utilities for ForgeAI training data.

Replaces JSONL as the on-disk training data format. Parquet with ZSTD
compression gives 3-5x compression and better I/O throughput thanks to
columnar layout and memory-mapped reads.

Three main components:
  1. ``convert_jsonl_to_parquet`` — one-shot converter from JSONL to Parquet/ZSTD.
  2. ``ParquetDataset`` — memory-mapped, random-access dataset reader
     (subclass of ``torch.utils.data.Dataset``).
  3. ``StreamingDataLoader`` — PyTorch-compatible DataLoader that wraps
     ``ParquetDataset`` with shuffling, multi-worker loading, and optional
     on-the-fly transformation (e.g. tokenisation).

Usage (converter)::

    from forge.training.data.parquet_dataset import convert_jsonl_to_parquet
    convert_jsonl_to_parquet("data.jsonl", "data.parquet")

Usage (training)::

    from forge.training.data.parquet_dataset import ParquetDataset, StreamingDataLoader
    ds = ParquetDataset("data.parquet")
    loader = StreamingDataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    for batch in loader:
        # batch is a dict with keys from the parquet columns
        ...
"""
from __future__ import annotations

import bisect
import json
import os
import random
from collections.abc import Callable, Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import contextlib

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

# ZSTD compression level (1-22; 3 is a good default balancing speed/ratio).
_ZSTD_LEVEL = 3


def _infer_arrow_type(values: list[Any]) -> pa.DataType:
    """Infer the narrowest Arrow type that fits all *values*.

    Falls back to ``pa.string()`` for heterogeneous / dict / list values so
    that arbitrary JSON objects round-trip through Parquet without loss.
    """
    # Collect non-None samples.
    samples = [v for v in values if v is not None]
    if not samples:
        return pa.string()

    first = samples[0]

    # Lists of ints (e.g. input_ids / labels) -> list<int64>.
    if isinstance(first, list):
        inner = [x for v in samples if isinstance(v, list) for x in v if x is not None]
        if inner and all(isinstance(x, int) for x in inner):
            return pa.list_(pa.int64())
        # List of strings or mixed -> list<string>.
        if inner and all(isinstance(x, str) for x in inner):
            return pa.list_(pa.string())
        # Fallback: store as JSON string.
        return pa.string()

    # Scalar ints.
    if all(isinstance(v, int) and not isinstance(v, bool) for v in samples):
        return pa.int64()

    # Scalar floats.
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in samples):
        return pa.float64()

    # Booleans.
    if all(isinstance(v, bool) for v in samples):
        return pa.bool_()

    # Dicts / mixed / strings -> store as JSON string for safe round-trip.
    return pa.string()


def _json_dumps_for_parquet(obj: Any) -> str:
    """Serialise *obj* to a compact JSON string for storage in a string column."""
    return json.dumps(obj, ensure_ascii=False)


def _json_loads_from_parquet(s: str) -> Any:
    """Deserialise a JSON string stored in a parquet string column."""
    return json.loads(s)


def convert_jsonl_to_parquet(jsonl_path: str, parquet_path: str) -> int:
    """Convert a JSONL file to Parquet with ZSTD compression.

    Each non-empty line in *jsonl_path* is parsed as JSON and stored as one
    row. Complex values (lists, dicts) are stored as JSON strings in string
    columns so that the schema is stable and round-trip safe. Scalar values
    (int, float, bool, str) are stored natively.

    Returns the number of rows written.
    """
    jsonl_path = str(jsonl_path)
    parquet_path = str(parquet_path)

    # ── Pass 1: read all rows and collect column names + samples ──
    rows: list[dict] = []
    column_samples: dict[str, list[Any]] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                # Wrap non-dict JSON values.
                obj = {"value": obj}
            rows.append(obj)
            for key, val in obj.items():
                column_samples.setdefault(key, []).append(val)

    if not rows:
        # Write an empty parquet file with a dummy schema.
        schema = pa.schema([pa.field("empty", pa.string())])
        with pq.ParquetWriter(parquet_path, schema,
                              compression="zstd",
                              compression_level=_ZSTD_LEVEL) as writer:
            writer.write_table(pa.table({}))
        return 0

    # ── Build schema + typed arrays ──
    # Determine which columns are "simple" (native Arrow) vs "complex" (JSON string).
    arrays: dict[str, pa.Array] = {}
    schema_fields: list[pa.Field] = []

    for col_name, samples in column_samples.items():
        arrow_type = _infer_arrow_type(samples)
        if arrow_type == pa.string() and any(
            isinstance(v, (dict, list)) for v in samples if v is not None
        ):
            # Complex column — serialise each value to JSON string.
            str_vals = [
                _json_dumps_for_parquet(r.get(col_name)) if r.get(col_name) is not None
                else None
                for r in rows
            ]
            arrays[col_name] = pa.array(str_vals, type=pa.string())
            schema_fields.append(pa.field(col_name, pa.string()))
        else:
            # Native column.
            vals = [r.get(col_name) for r in rows]
            try:
                arrays[col_name] = pa.array(vals, type=arrow_type)
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
                # Fallback to JSON string if native cast fails.
                str_vals = [
                    _json_dumps_for_parquet(v) if v is not None else None
                    for v in vals
                ]
                arrays[col_name] = pa.array(str_vals, type=pa.string())
                schema_fields.append(pa.field(col_name, pa.string()))
                continue
            schema_fields.append(pa.field(col_name, arrow_type))

    schema = pa.schema(schema_fields)
    table = pa.table(arrays, schema=schema)

    # ── Write with ZSTD compression ──
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    with pq.ParquetWriter(parquet_path, schema,
                          compression="zstd",
                          compression_level=_ZSTD_LEVEL) as writer:
        writer.write_table(table)

    return len(rows)


def convert_jsonl_to_parquet_tokenized(
    jsonl_path: str,
    parquet_path: str,
    tokenizer,
    max_seq_len: int = 1024,
    split_multi_turn_fn: Callable | None = None,
) -> int:
    """Convert a JSONL file to **pre-tokenised** Parquet with ZSTD compression.

    Each JSONL row is tokenised into one or more ``(input_ids, labels)`` pairs
    (multi-turn conversations are split per assistant turn). The resulting
    parquet file has columns ``input_ids`` (list<int64>) and ``labels``
    (list<int64>) so that ``ParquetDataset`` can feed directly into training
    without any runtime tokenisation.

    Parameters
    ----------
    jsonl_path : str
        Source JSONL file.
    parquet_path : str
        Destination ``.parquet`` file.
    tokenizer : Any
        HuggingFace-style tokenizer with ``__call__`` returning input_ids.
    max_seq_len : int
        Maximum sequence length — longer examples are dropped.
    split_multi_turn_fn : callable, optional
        If provided, called with ``messages`` to split multi-turn conversations
        into per-turn ``(prompt_text, completion_text)`` pairs. If ``None``,
        single-turn ``prompt``/``response`` rows are expected.

    Returns
    -------
    int
        Number of tokenised rows written.
    """
    from forge.training.runners.sft_train import (
        load_examples,
        tokenize_example,
    )

    examples = load_examples([jsonl_path])
    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []

    for ex in examples:
        toks = tokenize_example(ex, tokenizer, max_seq_len)
        for t in toks:
            input_ids_list.append(t["input_ids"])
            labels_list.append(t["labels"])

    if not input_ids_list:
        schema = pa.schema([
            pa.field("input_ids", pa.list_(pa.int64())),
            pa.field("labels", pa.list_(pa.int64())),
        ])
        with pq.ParquetWriter(parquet_path, schema,
                              compression="zstd",
                              compression_level=_ZSTD_LEVEL) as writer:
            writer.write_table(pa.table({}))
        return 0

    table = pa.table({
        "input_ids": pa.array(input_ids_list, type=pa.list_(pa.int64())),
        "labels": pa.array(labels_list, type=pa.list_(pa.int64())),
    })
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    pq.write_table(table, parquet_path, compression="zstd",
                   compression_level=_ZSTD_LEVEL)
    return len(input_ids_list)


# ---------------------------------------------------------------------------
# Dataset reader
# ---------------------------------------------------------------------------

class ParquetDataset:
    """Memory-mapped, random-access reader for Parquet files.

    Implements the ``torch.utils.data.Dataset`` protocol (``__len__`` +
    ``__getitem__``) so it can be used with ``torch.utils.data.DataLoader``
    or with :class:`StreamingDataLoader` below.

    Complex columns (lists, dicts) that were stored as JSON strings are
    automatically deserialised on read. Native columns (int, float, bool,
    string) are returned as-is.

    Parameters
    ----------
    parquet_path : str
        Path to the ``.parquet`` file.
    columns : list[str], optional
        Subset of columns to read (column projection for faster I/O).
        If ``None``, reads all columns.
    """

    def __init__(self, parquet_path: str, columns: list[str] | None = None):
        self.parquet_path = str(parquet_path)
        self._columns = columns
        self._pq_file = pq.ParquetFile(self.parquet_path)
        self._metadata = self._pq_file.metadata
        self._num_rows = self._metadata.num_rows
        self._schema = self._pq_file.schema_arrow

        # Build a mapping from global row index -> (row_group, row_in_group).
        self._row_group_offsets: list[int] = []
        self._rg_row_counts: list[int] = []
        offset = 0
        for rg_idx in range(self._metadata.num_row_groups):
            rg_meta = self._metadata.row_group(rg_idx)
            n = rg_meta.num_rows
            self._row_group_offsets.append(offset)
            self._rg_row_counts.append(n)
            offset += n

        # Identify which columns are JSON-encoded (string type but originally
        # complex). We detect this lazily: if a string column value parses as
        # JSON dict/list, we treat it as JSON-encoded.
        self._json_columns: set[str] | None = None

        # Cache the last-read row group to avoid re-reading millions of rows
        # for repeated accesses to the same group.
        self._cached_rg_idx: int | None = None
        self._cached_rg_table: pa.Table | None = None

    def _detect_json_columns(self, sample_row: dict) -> set[str]:
        """Detect which string columns contain JSON-encoded complex values."""
        json_cols: set[str] = set()
        for key, val in sample_row.items():
            if isinstance(val, str) and val and val[0] in "[{":
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, (dict, list)):
                        json_cols.add(key)
                except (json.JSONDecodeError, ValueError):
                    pass
        return json_cols

    def __getstate__(self) -> dict:
        """Pickle support: pyarrow ParquetFile isn't picklable, so workers
        re-open the file by path (needed for Windows spawn start method)."""
        return {"parquet_path": self.parquet_path,
                "_columns": self._columns,
                "_json_columns": self._json_columns}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["parquet_path"], columns=state["_columns"])
        self._json_columns = state.get("_json_columns")
        # Row group cache is reset on unpickle (worker re-opens file).

    @property
    def num_rows(self) -> int:
        """Total number of rows in the dataset."""
        return self._num_rows

    @property
    def schema(self) -> pa.Schema:
        """Arrow schema of the underlying parquet file."""
        return self._schema

    def __len__(self) -> int:
        return self._num_rows

    def _read_row_group(self, rg_idx: int) -> pa.Table:
        """Read a single row group as an Arrow table."""
        return self._pq_file.read_row_group(rg_idx, columns=self._columns)

    def _row_to_dict(self, row_idx: int) -> dict[str, Any]:
        """Read a single row by global index, returning a plain dict."""
        # Binary search for the right row group (offsets are sorted).
        rg_idx = bisect.bisect_right(self._row_group_offsets, row_idx) - 1
        if rg_idx < 0 or rg_idx >= len(self._row_group_offsets):
            raise IndexError(f"Row {row_idx} out of range (len={self._num_rows})")

        local_idx = row_idx - self._row_group_offsets[rg_idx]

        # Use cached row group table if available (avoids re-reading
        # millions of rows for repeated accesses to the same group).
        if self._cached_rg_idx == rg_idx and self._cached_rg_table is not None:
            table = self._cached_rg_table
        else:
            table = self._read_row_group(rg_idx)
            self._cached_rg_idx = rg_idx
            self._cached_rg_table = table

        row = {col: table.column(col)[local_idx].as_py()
               for col in table.column_names}

        # Lazily detect and decode JSON columns on first access.
        if self._json_columns is None:
            self._json_columns = self._detect_json_columns(row)

        for col in self._json_columns:
            if col in row and isinstance(row[col], str):
                try:
                    row[col] = _json_loads_from_parquet(row[col])
                except (json.JSONDecodeError, ValueError):
                    pass  # leave as string if parse fails

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self._num_rows
        if idx < 0 or idx >= self._num_rows:
            raise IndexError(f"Index {idx} out of range [0, {self._num_rows})")
        return self._row_to_dict(idx)

    def iter_batches(self, batch_size: int) -> Iterator[list[dict]]:
        """Iterate over the dataset in sequential batches.

        Yields lists of ``batch_size`` dicts (the last batch may be smaller).
        This uses pyarrow's efficient batch reading under the hood.
        """
        for batch in self._pq_file.iter_batches(batch_size=batch_size,
                                                columns=self._columns):
            rows = []
            n = batch.num_rows
            for i in range(n):
                row = {col: batch.column(col)[i].as_py()
                       for col in batch.column_names}
                # Decode JSON columns.
                if self._json_columns is None:
                    self._json_columns = self._detect_json_columns(row)
                for col in self._json_columns:
                    if col in row and isinstance(row[col], str):
                        with contextlib.suppress(json.JSONDecodeError, ValueError):
                            row[col] = _json_loads_from_parquet(row[col])
                rows.append(row)
            yield rows

    def iter_batch_dicts(self, batch_size: int) -> Iterator[dict[str, list]]:
        """Iterate over the dataset yielding column-oriented batch dicts.

        Each yielded dict maps column name -> list of values (length batch_size).
        This is more efficient than ``iter_batches`` for columnar processing.
        """
        for batch in self._pq_file.iter_batches(batch_size=batch_size,
                                                columns=self._columns):
            yield {col: [v.as_py() for v in batch.column(col)]
                   for col in batch.column_names}


# ---------------------------------------------------------------------------
# Streaming DataLoader
# ---------------------------------------------------------------------------

class _ParquetWorkerDataset:
    """Picklable worker dataset for multi-process DataLoader.

    Module-level so Windows spawn (pickle) can ship it to workers; the local
    class defined inside ``_worker_iter`` previously broke ``num_workers>0``
    on Windows (``AttributeError: Can't pickle local object``).
    """

    def __init__(self, dataset: ParquetDataset, indices: list[int],
                 transform_fn: Callable[[dict], dict] | None):
        self._dataset = dataset
        self._indices = indices
        self._transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        row = self._dataset[self._indices[idx]]
        if self._transform_fn is not None:
            row = self._transform_fn(row)
        return row


class StreamingDataLoader:
    """PyTorch-compatible streaming DataLoader for ``ParquetDataset``.

    Wraps a :class:`ParquetDataset` and provides batched iteration with
    optional shuffling, multi-worker prefetching, and on-the-fly
    transformation.

    Parameters
    ----------
    dataset : ParquetDataset
        The dataset to load from.
    batch_size : int
        Number of examples per batch.
    shuffle : bool
        Whether to shuffle examples each epoch (uses memory-mapped random
        access — no full-dataset copy into RAM).
    num_workers : int
        Number of background worker processes for parallel data loading.
        ``0`` means single-process (inline).
    transform_fn : callable, optional
        Applied to each raw row dict before batching. Useful for on-the-fly
        tokenisation or tensor conversion. Should return a dict.
    collate_fn : callable, optional
        Custom collation function that takes a list of (transformed) row
        dicts and returns a batch. If ``None``, uses :meth:`_default_collate`.
    drop_last : bool
        If ``True``, drop the last incomplete batch.
    seed : int
        Random seed for shuffling reproducibility.
    pin_memory : bool
        If ``True``, pin batch tensors in memory for faster H2D transfer.
    device : str, optional
        Device to move batch tensors to (e.g. ``"cuda"``). If ``None``,
        tensors stay on CPU.

    Yields
    ------
    dict
        Batch dict with keys from the (transformed) rows. List values are
        stacked into tensors; scalar values are kept as lists.
    """

    def __init__(
        self,
        dataset: ParquetDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        transform_fn: Callable[[dict], dict] | None = None,
        collate_fn: Callable[[list[dict]], dict] | None = None,
        drop_last: bool = False,
        seed: int = 42,
        pin_memory: bool = False,
        device: str | None = None,
        prefetch_factor: int = 2,
        # Default pad_id=0 is a fallback; callers with a tokenizer should pass
        # pad_id=tokenizer.pad_token_id (or tokenizer.eos_token_id if pad is None).
        pad_id: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform_fn = transform_fn
        self.collate_fn = collate_fn or self._default_collate
        self.drop_last = drop_last
        self.seed = seed
        self.pin_memory = pin_memory
        self.device = device
        # Batches prefetched per worker: workers decode the next N batches
        # while the GPU processes the current one (hides parquet ZSTD CPU cost).
        self.prefetch_factor = prefetch_factor
        self.pad_id = pad_id
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _get_indices(self) -> list[int]:
        """Return the index order for this epoch."""
        n = len(self.dataset)
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            indices = list(range(n))
            rng.shuffle(indices)
            return indices
        return list(range(n))

    def _default_collate(self, rows: list[dict]) -> dict:
        """Collate a list of row dicts into a batch dict.

        - List-of-int values (e.g. ``input_ids``, ``labels``) are padded and
          stacked into ``torch.Tensor``s.
        - Scalar values are collected into lists.
        - Other list values are kept as lists of lists.
        """
        try:
            import torch
        except ImportError:
            torch = None

        if not rows:
            return {}

        keys = rows[0].keys()
        batch: dict[str, Any] = {}

        for key in keys:
            values = [r.get(key) for r in rows]

            # Check if all values are lists of ints (tokenised data).
            if (torch is not None
                    and all(isinstance(v, list) for v in values if v is not None)
                    and values
                    and all(isinstance(x, int)
                            for v in values if v is not None
                            for x in v)):
                # Pad to max length and stack into tensor.
                max_len = max(len(v) for v in values if v is not None)
                tensor = torch.full((len(values), max_len), self.pad_id,
                                    dtype=torch.long)
                for i, v in enumerate(values):
                    if v is not None:
                        tensor[i, :len(v)] = torch.tensor(v, dtype=torch.long)
                if self.device and torch.cuda.is_available() and self.pin_memory:
                    # Pinned + non-blocking H2D: DMA streams batches to GPU
                    # while the compute stream is still running.
                    tensor = tensor.pin_memory().to(self.device, non_blocking=True)
                elif self.device:
                    tensor = tensor.to(self.device)
                batch[key] = tensor
            else:
                batch[key] = values

        return batch

    def _worker_iter(self, indices: list[int]) -> Iterator[dict]:
        """Multi-worker iterator using ``torch.utils.data.DataLoader``.

        Workers prefetch + decode the next ``prefetch_factor`` batches while
        the GPU processes the current one (hides parquet ZSTD CPU cost).
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(
            _ParquetWorkerDataset(self.dataset, indices, self.transform_fn),
            batch_size=self.batch_size,
            shuffle=False,  # already shuffled via indices
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
        )
        yield from loader

    def _single_iter(self, indices: list[int]) -> Iterator[dict]:
        """Single-process iterator."""
        n = len(indices)
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > n:
                break
            batch_indices = indices[start:end]
            rows = []
            for idx in batch_indices:
                row = self.dataset[idx]
                if self.transform_fn is not None:
                    row = self.transform_fn(row)
                rows.append(row)
            yield self.collate_fn(rows)

    def __iter__(self) -> Iterator[dict]:
        indices = self._get_indices()
        if self.num_workers > 0:
            yield from self._worker_iter(indices)
        else:
            yield from self._single_iter(indices)
        self._epoch += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    """Command-line interface for batch conversion of JSONL → Parquet."""
    import argparse
    p = argparse.ArgumentParser(
        description="Convert JSONL training data to Parquet (ZSTD)")
    p.add_argument("inputs", nargs="+", help="JSONL file(s) or directory")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Output directory (default: same as input)")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="Recurse into directories")
    p.add_argument("--tokenized", action="store_true",
                   help="Pre-tokenize data (requires tokenizer)")
    p.add_argument("--max-seq-len", type=int, default=1024)
    args = p.parse_args()

    import sys
    sys.path.insert(0, os.getcwd())

    tokenizer = None
    if args.tokenized:
        from research.tokenizer_cache import get_tokenizer
        tokenizer = get_tokenizer()

    files: list[str] = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            if args.recursive:
                for root, _, fns in os.walk(inp):
                    for fn in fns:
                        if fn.endswith(".jsonl"):
                            files.append(os.path.join(root, fn))
            else:
                for fn in os.listdir(inp):
                    if fn.endswith(".jsonl"):
                        files.append(os.path.join(inp, fn))
        elif inp.endswith(".jsonl"):
            files.append(inp)

    if not files:
        print("No .jsonl files found.")
        return

    out_dir = args.output_dir
    total = 0
    for jsonl_file in files:
        if out_dir:
            rel = os.path.relpath(jsonl_file, os.path.dirname(jsonl_file))
            parquet_file = os.path.join(out_dir, rel.replace(".jsonl", ".parquet"))
        else:
            parquet_file = jsonl_file.replace(".jsonl", ".parquet")

        os.makedirs(os.path.dirname(parquet_file) or ".", exist_ok=True)

        if args.tokenized and tokenizer:
            n = convert_jsonl_to_parquet_tokenized(
                jsonl_file, parquet_file, tokenizer, args.max_seq_len)
        else:
            n = convert_jsonl_to_parquet(jsonl_file, parquet_file)

        orig_size = os.path.getsize(jsonl_file)
        new_size = os.path.getsize(parquet_file)
        ratio = orig_size / new_size if new_size > 0 else float("inf")
        print(f"  {jsonl_file} -> {parquet_file}")
        print(f"    {n} rows | {orig_size/1e6:.1f} MB -> {new_size/1e6:.1f} MB "
              f"({ratio:.1f}x compression)")
        total += n

    print(f"\nTotal: {total} rows converted from {len(files)} files.")


if __name__ == "__main__":
    _cli()
