"""Reprojection filter — transform coordinates between CRS."""

from __future__ import annotations

from typing import Any

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry
from pywolken.utils.crs import reproject_arrays


class ReprojectionFilter(Filter):
    """Reproject point cloud coordinates from one CRS to another.

    Options:
        in_srs: str — Source CRS (e.g., "EPSG:25832"). If omitted, uses
                       the point cloud's CRS.
        out_srs: str — Target CRS (e.g., "EPSG:4326"). Required.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        if "out_srs" not in self.options:
            raise ValueError("ReprojectionFilter requires 'out_srs' option")

    def filter(self, pc: PointCloud) -> PointCloud:
        src_crs = self.options.get("in_srs", pc.crs)
        dst_crs = self.options["out_srs"]

        if src_crs is None:
            raise ValueError(
                "No source CRS: set 'in_srs' option or ensure the point cloud has a CRS"
            )

        new_x, new_y, new_z = reproject_arrays(
            pc["X"], pc["Y"], pc["Z"], src_crs, dst_crs
        )

        result = pc.copy()
        result["X"] = new_x
        result["Y"] = new_y
        result["Z"] = new_z
        result.crs = dst_crs
        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.reprojection"


filter_registry.register(ReprojectionFilter)
