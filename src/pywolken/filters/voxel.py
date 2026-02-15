"""Voxel grid downsampling — reduce density by averaging within voxels."""

from __future__ import annotations

from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class VoxelFilter(Filter):
    """Downsample using a voxel grid.

    Divides space into 3D voxel cells and keeps one representative point
    per occupied voxel (centroid of all points in that voxel).

    Options:
        cell_size: float — Voxel edge length. Default: 1.0.
            Can also use 'cell' as alias.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    def filter(self, pc: PointCloud) -> PointCloud:
        cell_size = float(self.options.get("cell_size", self.options.get("cell", 1.0)))

        x, y, z = pc["X"], pc["Y"], pc["Z"]

        # Compute voxel indices
        vx = np.floor(x / cell_size).astype(np.int64)
        vy = np.floor(y / cell_size).astype(np.int64)
        vz = np.floor(z / cell_size).astype(np.int64)

        # Create unique voxel keys
        # Shift to positive indices
        vx -= vx.min()
        vy -= vy.min()
        vz -= vz.min()

        # Combine into single key: use large multipliers to avoid collisions
        ny = int(vy.max()) + 1
        nz = int(vz.max()) + 1
        voxel_key = vx * (ny * nz) + vy * nz + vz

        # Find unique voxels and compute centroids
        unique_keys, inverse = np.unique(voxel_key, return_inverse=True)
        n_voxels = len(unique_keys)

        result = PointCloud()

        # For X, Y, Z: compute centroid per voxel
        for dim in ["X", "Y", "Z"]:
            arr = pc[dim]
            sums = np.zeros(n_voxels, dtype=np.float64)
            np.add.at(sums, inverse, arr)
            counts = np.zeros(n_voxels, dtype=np.int64)
            np.add.at(counts, inverse, 1)
            result._arrays[dim] = sums / counts

        # For other dimensions: take the value from the first point in each voxel
        first_idx = np.empty(n_voxels, dtype=np.int64)
        # Walk backwards so first occurrence wins
        for i in range(len(inverse) - 1, -1, -1):
            first_idx[inverse[i]] = i

        for dim in pc.dimensions:
            if dim not in ("X", "Y", "Z"):
                result._arrays[dim] = pc[dim][first_idx]

        result._metadata = pc.metadata.copy()
        result._crs = pc.crs
        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.voxel"


filter_registry.register(VoxelFilter)
