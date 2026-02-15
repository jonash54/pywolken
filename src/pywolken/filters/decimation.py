"""Decimation filter — reduce point density by thinning.

Supports multiple decimation strategies:
    - nth: Keep every Nth point (deterministic)
    - random: Random sampling to target count or fraction
    - voxel_center: Voxel grid center-of-mass (Phase 4)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class DecimationFilter(Filter):
    """Reduce point count by thinning.

    Options:
        step: int — Keep every Nth point (default strategy).
            step=10 keeps every 10th point (10% of data).

        fraction: float — Keep this fraction of points (0.0 to 1.0).
            fraction=0.1 keeps 10% of points randomly.

        count: int — Keep exactly this many points (random sample).

        seed: int — Random seed for reproducibility (default: None).
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        has_step = "step" in self.options
        has_fraction = "fraction" in self.options
        has_count = "count" in self.options
        if not (has_step or has_fraction or has_count):
            raise ValueError(
                "DecimationFilter requires one of: 'step', 'fraction', or 'count'"
            )

    def filter(self, pc: PointCloud) -> PointCloud:
        n = pc.num_points

        if "step" in self.options:
            step = int(self.options["step"])
            if step < 1:
                raise ValueError(f"step must be >= 1, got {step}")
            indices = np.arange(0, n, step)

        elif "fraction" in self.options:
            fraction = float(self.options["fraction"])
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"fraction must be in (0, 1], got {fraction}")
            target = max(1, int(n * fraction))
            rng = np.random.default_rng(self.options.get("seed"))
            indices = np.sort(rng.choice(n, size=target, replace=False))

        elif "count" in self.options:
            count = int(self.options["count"])
            if count < 1:
                raise ValueError(f"count must be >= 1, got {count}")
            count = min(count, n)
            rng = np.random.default_rng(self.options.get("seed"))
            indices = np.sort(rng.choice(n, size=count, replace=False))

        else:
            return pc

        # Apply index selection
        result = PointCloud()
        for dim_name in pc.dimensions:
            result._arrays[dim_name] = pc[dim_name][indices]
        result._metadata = pc.metadata.copy()
        result._crs = pc.crs
        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.decimation"


filter_registry.register(DecimationFilter)
