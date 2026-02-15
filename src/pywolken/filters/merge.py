"""Merge filter — combine the current point cloud with additional inputs.

In a pipeline context, the merge filter is typically used when multiple
readers feed into a single processing chain. The pipeline engine handles
merging automatically for multiple readers, but this filter can also
explicitly merge point clouds.
"""

from __future__ import annotations

from typing import Any

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class MergeFilter(Filter):
    """Merge multiple point clouds into one.

    In pipeline execution, this is handled automatically when multiple
    readers are present. This filter exists for explicit merge operations
    and API usage.

    When used via the API, pass additional clouds via the 'inputs' option:
        merge = MergeFilter(inputs=[pc2, pc3])
        result = merge.filter(pc1)
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self._inputs: list[PointCloud] = options.get("inputs", [])

    def add_input(self, pc: PointCloud) -> None:
        """Add a point cloud to be merged."""
        self._inputs.append(pc)

    def filter(self, pc: PointCloud) -> PointCloud:
        result = pc
        for other in self._inputs:
            result = result.merge(other)
        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.merge"


filter_registry.register(MergeFilter)
