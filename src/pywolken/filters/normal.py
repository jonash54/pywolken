"""Surface normal estimation via KDTree + PCA."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class NormalFilter(Filter):
    """Estimate surface normals for each point using PCA on local neighborhoods.

    Adds NormalX, NormalY, NormalZ dimensions.

    Options:
        k: int — Number of neighbors for PCA. Default: 8.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    def filter(self, pc: PointCloud) -> PointCloud:
        k = int(self.options.get("k", 8))
        points = np.column_stack([pc["X"], pc["Y"], pc["Z"]])
        tree = cKDTree(points)

        # Query k+1 neighbors (includes self)
        _, idx = tree.query(points, k=min(k + 1, len(points)))

        normals = np.zeros((len(points), 3), dtype=np.float64)

        for i in range(len(points)):
            neighbors = points[idx[i]]
            # Center the neighborhood
            centered = neighbors - neighbors.mean(axis=0)
            # Covariance matrix
            cov = centered.T @ centered
            # Eigendecomposition — smallest eigenvector is the normal
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # Smallest eigenvalue

        # Orient normals upward (Z component positive)
        flip = normals[:, 2] < 0
        normals[flip] *= -1

        result = pc.copy()
        for name, col in [("NormalX", 0), ("NormalY", 1), ("NormalZ", 2)]:
            if name not in result:
                result.add_dimension(name, np.dtype(np.float64))
            result[name][:] = normals[:, col]

        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.normal"


filter_registry.register(NormalFilter)
