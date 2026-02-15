"""Outlier removal filter — statistical and radius-based.

Two methods:
    - statistical: Remove points whose mean distance to K nearest neighbors
      exceeds mean + multiplier * stddev (PDAL's filters.outlier method="statistical").
    - radius: Remove points with fewer than min_k neighbors within radius
      (PDAL's filters.outlier method="radius").
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class OutlierFilter(Filter):
    """Remove outlier points.

    Options:
        method: str — "statistical" (default) or "radius".

        Statistical options:
            mean_k: int — Number of neighbors to analyze. Default: 8.
            multiplier: float — StdDev multiplier threshold. Default: 2.0.

        Radius options:
            radius: float — Search radius. Required for radius method.
            min_k: int — Minimum neighbors within radius. Default: 2.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    def filter(self, pc: PointCloud) -> PointCloud:
        method = self.options.get("method", "statistical")

        points = np.column_stack([pc["X"], pc["Y"], pc["Z"]])
        tree = cKDTree(points)

        if method == "statistical":
            mask = self._statistical(tree, points)
        elif method == "radius":
            mask = self._radius(tree, points)
        else:
            raise ValueError(f"Unknown outlier method '{method}'. Use: statistical, radius")

        return pc.mask(mask)

    def _statistical(self, tree: cKDTree, points: np.ndarray) -> np.ndarray:
        mean_k = int(self.options.get("mean_k", 8))
        multiplier = float(self.options.get("multiplier", 2.0))

        # k+1 because query includes the point itself
        dist, _ = tree.query(points, k=mean_k + 1)
        # Skip first column (distance to self = 0)
        mean_dist = np.mean(dist[:, 1:], axis=1)

        global_mean = np.mean(mean_dist)
        global_std = np.std(mean_dist)
        threshold = global_mean + multiplier * global_std

        return mean_dist <= threshold

    def _radius(self, tree: cKDTree, points: np.ndarray) -> np.ndarray:
        radius = float(self.options.get("radius", 1.0))
        min_k = int(self.options.get("min_k", 2))

        counts = tree.query_ball_point(points, r=radius, return_length=True)
        # Subtract 1 because query includes the point itself
        return (counts - 1) >= min_k

    @classmethod
    def type_name(cls) -> str:
        return "filters.outlier"


filter_registry.register(OutlierFilter)
