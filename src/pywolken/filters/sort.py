"""Spatial sort filter — sort points by Morton (Z-order) curve."""

from __future__ import annotations

from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


def _interleave_bits(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute 2D Morton code (Z-order) for integer coordinate arrays."""
    # Spread bits of x and y, then interleave
    def _spread(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64)
        v = (v | (v << 16)) & np.uint64(0x0000FFFF0000FFFF)
        v = (v | (v << 8)) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v << 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v << 2)) & np.uint64(0x3333333333333333)
        v = (v | (v << 1)) & np.uint64(0x5555555555555555)
        return v

    return _spread(x) | (_spread(y) << np.uint64(1))


class SortFilter(Filter):
    """Sort points spatially using Morton (Z-order) curve.

    Improves spatial locality for subsequent processing. Points nearby
    in space become nearby in the array.

    Options:
        order: str — Sort order. "morton" (default) or "xyz".
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    def filter(self, pc: PointCloud) -> PointCloud:
        order = self.options.get("order", "morton")

        if order == "morton":
            indices = self._morton_order(pc)
        elif order == "xyz":
            # Lexicographic sort by X, then Y, then Z
            indices = np.lexsort((pc["Z"], pc["Y"], pc["X"]))
        else:
            raise ValueError(f"Unknown sort order '{order}'. Use: morton, xyz")

        result = PointCloud()
        for dim_name in pc.dimensions:
            result._arrays[dim_name] = pc[dim_name][indices]
        result._metadata = pc.metadata.copy()
        result._crs = pc.crs
        return result

    def _morton_order(self, pc: PointCloud) -> np.ndarray:
        x, y = pc["X"], pc["Y"]
        # Normalize to integer grid (16-bit precision)
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()
        x_range = max(xmax - xmin, 1e-10)
        y_range = max(ymax - ymin, 1e-10)

        xi = ((x - xmin) / x_range * 65535).astype(np.uint32)
        yi = ((y - ymin) / y_range * 65535).astype(np.uint32)

        morton = _interleave_bits(xi, yi)
        return np.argsort(morton)

    @classmethod
    def type_name(cls) -> str:
        return "filters.sort"


filter_registry.register(SortFilter)
