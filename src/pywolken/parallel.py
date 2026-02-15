"""Optional Dask integration for parallel point cloud processing."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.io.base import Reader


def _check_dask() -> None:
    """Check that dask is installed."""
    try:
        import dask  # noqa: F401
    except ImportError:
        raise ImportError(
            "dask required for parallel processing. "
            "Install with: pip install pywolken[dask]"
        )


def parallel_read(
    paths: list[str],
    reader: Reader | None = None,
    **options: Any,
) -> PointCloud:
    """Read multiple files in parallel using Dask and merge.

    Args:
        paths: List of file paths to read.
        reader: Reader instance. If None, auto-detects from extension.
        **options: Reader options.

    Returns:
        Merged PointCloud from all files.
    """
    _check_dask()
    import dask

    if reader is None:
        from pywolken.io.registry import _ensure_registered, io_registry
        _ensure_registered()

    @dask.delayed
    def _read_one(path: str) -> PointCloud:
        r = reader if reader is not None else io_registry.get_reader(path)
        return r.read(path, **options)

    delayed_results = [_read_one(p) for p in paths]
    clouds = dask.compute(*delayed_results)

    # Merge all
    result = clouds[0]
    for other in clouds[1:]:
        result = result.merge(other)

    return result


def parallel_filter(
    pc: PointCloud,
    filters: list[Filter],
    n_chunks: int = 4,
) -> PointCloud:
    """Apply filters to a point cloud in parallel chunks using Dask.

    Splits the point cloud into n_chunks, processes each independently,
    then merges. Only works correctly for per-point filters (range,
    expression, assign, etc.) — NOT for spatial filters (ground,
    cluster, normal).

    Args:
        pc: Input point cloud.
        filters: List of filters to apply.
        n_chunks: Number of parallel chunks.

    Returns:
        Filtered and merged PointCloud.
    """
    _check_dask()
    import dask

    chunk_size = max(1, pc.num_points // n_chunks)
    chunks = list(pc.iter_chunks(chunk_size))

    @dask.delayed
    def _process_chunk(chunk: PointCloud) -> PointCloud:
        result = chunk
        for filt in filters:
            result = filt.filter(result)
            if result.num_points == 0:
                break
        return result

    delayed_results = [_process_chunk(c) for c in chunks]
    processed = dask.compute(*delayed_results)

    # Merge non-empty results
    non_empty = [p for p in processed if p.num_points > 0]
    if not non_empty:
        return PointCloud()

    result = non_empty[0]
    for other in non_empty[1:]:
        result = result.merge(other)

    return result


def parallel_apply(
    pc: PointCloud,
    func: Callable[[PointCloud], PointCloud],
    n_chunks: int = 4,
) -> PointCloud:
    """Apply a custom function to point cloud chunks in parallel.

    Args:
        pc: Input point cloud.
        func: Function that takes a PointCloud and returns a PointCloud.
        n_chunks: Number of parallel chunks.

    Returns:
        Merged result.
    """
    _check_dask()
    import dask

    chunk_size = max(1, pc.num_points // n_chunks)
    chunks = list(pc.iter_chunks(chunk_size))

    delayed_results = [dask.delayed(func)(c) for c in chunks]
    processed = dask.compute(*delayed_results)

    non_empty = [p for p in processed if p.num_points > 0]
    if not non_empty:
        return PointCloud()

    result = non_empty[0]
    for other in non_empty[1:]:
        result = result.merge(other)

    return result
