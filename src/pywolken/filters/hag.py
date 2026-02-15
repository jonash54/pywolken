"""Height Above Ground (HAG) filter.

Computes the height of each point above the ground surface and stores
it in a new 'HeightAboveGround' dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class HagFilter(Filter):
    """Compute Height Above Ground for each point.

    Requires that ground points are already classified (Classification=2).
    Adds a 'HeightAboveGround' dimension to the point cloud.

    Options:
        neighbors: int — Number of ground neighbors to average. Default: 3.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    def filter(self, pc: PointCloud) -> PointCloud:
        neighbors = int(self.options.get("neighbors", 3))

        if "Classification" not in pc:
            raise ValueError("HAG filter requires Classification dimension")

        ground_mask = pc["Classification"] == 2
        n_ground = int(np.sum(ground_mask))
        if n_ground == 0:
            raise ValueError("No ground points (Classification=2) found")

        # Build KDTree from ground point XY positions
        ground_xy = np.column_stack([
            pc["X"][ground_mask],
            pc["Y"][ground_mask],
        ])
        ground_z = pc["Z"][ground_mask]
        tree = cKDTree(ground_xy)

        # Query nearest ground neighbors for each point
        all_xy = np.column_stack([pc["X"], pc["Y"]])
        k = min(neighbors, n_ground)
        dist, idx = tree.query(all_xy, k=k)

        # Average ground Z from nearest neighbors
        if k == 1:
            ground_elev = ground_z[idx]
        else:
            ground_elev = np.mean(ground_z[idx], axis=1)

        hag = pc["Z"] - ground_elev

        result = pc.copy()
        if "HeightAboveGround" not in result:
            result.add_dimension("HeightAboveGround", np.dtype(np.float64))
        result["HeightAboveGround"][:] = hag

        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.hag"


filter_registry.register(HagFilter)
