"""Colorize filter — assign RGB from a raster image overlay."""

from __future__ import annotations

from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class ColorizeFilter(Filter):
    """Assign RGB values to points from a raster file (e.g., orthophoto).

    Requires rasterio. The raster must overlap the point cloud spatially.

    Options:
        raster: str — Path to raster file (GeoTIFF, etc.). Required.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        if "raster" not in self.options:
            raise ValueError("ColorizeFilter requires 'raster' option")

    def filter(self, pc: PointCloud) -> PointCloud:
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio required for colorize filter. "
                "Install with: pip install pywolken[raster]"
            )

        raster_path = self.options["raster"]

        with rasterio.open(raster_path) as src:
            # Get pixel coordinates for each point
            rows, cols = rasterio.transform.rowcol(
                src.transform, pc["X"], pc["Y"]
            )
            rows = np.asarray(rows)
            cols = np.asarray(cols)

            # Clamp to raster bounds
            valid = (
                (rows >= 0) & (rows < src.height)
                & (cols >= 0) & (cols < src.width)
            )

            result = pc.copy()
            bands = min(src.count, 3)

            for band_idx, dim_name in enumerate(["Red", "Green", "Blue"][:bands]):
                band_data = src.read(band_idx + 1)
                values = np.zeros(pc.num_points, dtype=np.uint16)
                values[valid] = band_data[rows[valid], cols[valid]]

                # Scale to uint16 if needed (8-bit rasters → scale to 16-bit)
                if band_data.dtype == np.uint8:
                    values = (values.astype(np.uint16) * 257)

                if dim_name not in result:
                    result.add_dimension(dim_name, np.dtype(np.uint16))
                result[dim_name][:] = values

        return result

    @classmethod
    def type_name(cls) -> str:
        return "filters.colorize"


filter_registry.register(ColorizeFilter)
