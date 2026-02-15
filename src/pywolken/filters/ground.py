"""Ground classification filter — Simple Morphological Filter (SMRF).

Classifies ground points (Classification=2) using a progressive
morphological approach. This is a simplified but effective implementation
of the SMRF algorithm (Pingel et al., 2013).

Algorithm:
    1. Create a minimum surface at the initial cell size
    2. Progressively increase the morphological window
    3. At each step, points below the surface + threshold are ground
    4. Points significantly above the surface are non-ground
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import minimum_filter

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class GroundFilter(Filter):
    """Classify ground points using Simple Morphological Filter (SMRF).

    Sets Classification=2 for detected ground points.
    Non-ground points are set to Classification=1 (unclassified).

    Options:
        cell_size: float — Initial grid cell size in CRS units. Default: 1.0.
        slope: float — Maximum expected terrain slope (rise/run). Default: 0.15.
        window_max: float — Maximum morphological window size. Default: 18.0.
        threshold: float — Height threshold for ground detection. Default: 0.5.
        scalar: float — Slope scaling factor. Default: 1.25.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

    def filter(self, pc: PointCloud) -> PointCloud:
        cell_size = float(self.options.get("cell_size", 1.0))
        slope = float(self.options.get("slope", 0.15))
        window_max = float(self.options.get("window_max", 18.0))
        threshold = float(self.options.get("threshold", 0.5))
        scalar = float(self.options.get("scalar", 1.25))

        x, y, z = pc["X"], pc["Y"], pc["Z"]
        n = pc.num_points

        # Step 1: Create minimum surface grid
        xmin, ymin = float(np.min(x)), float(np.min(y))
        xmax, ymax = float(np.max(x)), float(np.max(y))
        ncols = max(1, int(np.ceil((xmax - xmin) / cell_size)))
        nrows = max(1, int(np.ceil((ymax - ymin) / cell_size)))

        # Assign points to grid cells
        col_idx = np.clip(
            np.floor((x - xmin) / cell_size).astype(np.int64), 0, ncols - 1
        )
        row_idx = np.clip(
            np.floor((y - ymin) / cell_size).astype(np.int64), 0, nrows - 1
        )

        # Build minimum elevation grid
        min_grid = np.full((nrows, ncols), np.inf, dtype=np.float64)
        flat_idx = row_idx * ncols + col_idx
        np.minimum.at(min_grid.ravel(), flat_idx, z)

        # Fill empty cells with nearest neighbor from minimum grid
        empty = np.isinf(min_grid)
        if np.any(empty):
            from scipy.ndimage import distance_transform_edt
            _, nearest_idx = distance_transform_edt(empty, return_distances=True, return_indices=True)
            min_grid[empty] = min_grid[tuple(nearest_idx[:, empty])]

        # Step 2: Progressive morphological filtering
        is_ground = np.ones(n, dtype=bool)  # Start: all points are candidates

        window = 1
        while window <= window_max / cell_size:
            # Morphological opening (erosion then dilation → approximated by min filter)
            win_size = int(2 * window + 1)
            surface = minimum_filter(min_grid, size=win_size)

            # Height threshold increases with window size
            ht = threshold + slope * window * cell_size * scalar

            # Get surface elevation at each point's grid cell
            surface_z = surface[row_idx, col_idx]

            # Points above surface + threshold are not ground
            above = z - surface_z > ht
            is_ground &= ~above

            window *= 2

        # Step 3: Apply classification
        result = pc.copy()
        if "Classification" not in result:
            result.add_dimension("Classification", np.dtype(np.uint8))
        result["Classification"][:] = 1  # Default: unclassified
        result["Classification"][is_ground] = 2  # Ground

        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.ground"


filter_registry.register(GroundFilter)
