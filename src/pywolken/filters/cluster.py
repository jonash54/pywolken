"""Clustering filter — DBSCAN-based point cloud clustering."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class ClusterFilter(Filter):
    """Cluster points using DBSCAN algorithm.

    Adds a 'ClusterID' dimension. Points not belonging to any cluster
    get ClusterID=0 (noise).

    Options:
        tolerance: float — Maximum distance between cluster neighbors. Default: 1.0.
        min_points: int — Minimum points per cluster. Default: 10.
        is3d: bool — Use 3D distance (True) or 2D XY only (False). Default: True.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    def filter(self, pc: PointCloud) -> PointCloud:
        tolerance = float(self.options.get("tolerance", 1.0))
        min_points = int(self.options.get("min_points", 10))
        is3d = bool(self.options.get("is3d", True))

        if is3d:
            coords = np.column_stack([pc["X"], pc["Y"], pc["Z"]])
        else:
            coords = np.column_stack([pc["X"], pc["Y"]])

        tree = cKDTree(coords)
        n = len(coords)
        labels = np.zeros(n, dtype=np.int64)  # 0 = noise/unassigned
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = tree.query_ball_point(coords[i], r=tolerance)
            if len(neighbors) < min_points:
                continue  # Noise point

            cluster_id += 1
            labels[i] = cluster_id

            # Expand cluster
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = tree.query_ball_point(coords[q], r=tolerance)
                    if len(q_neighbors) >= min_points:
                        seed_set.extend(q_neighbors)
                if labels[q] == 0:
                    labels[q] = cluster_id
                j += 1

        result = pc.copy()
        if "ClusterID" not in result:
            result.add_dimension("ClusterID", np.dtype(np.int64))
        result["ClusterID"][:] = labels

        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.cluster"


filter_registry.register(ClusterFilter)
