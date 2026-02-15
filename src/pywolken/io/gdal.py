"""GDAL raster writer — generate DEM/hillshade GeoTIFF from point cloud.

Replaces PDAL's writers.gdal in pipelines. Generates a raster (DEM)
from point cloud data and writes it as a GeoTIFF.
"""

from __future__ import annotations

from typing import Any

from pywolken.core.pointcloud import PointCloud
from pywolken.io.base import Writer
from pywolken.raster.dem import create_dem
from pywolken.raster.export import write_geotiff


class GdalWriter(Writer):
    """Write point cloud to raster GeoTIFF (DEM generation).

    Replaces PDAL's writers.gdal. Generates a raster from point cloud data
    using interpolation, then writes it as a compressed GeoTIFF.

    Options:
        resolution: float — Grid cell size in CRS units (meters). Required.
        output_type: str — Interpolation method: "idw", "mean", "nearest", "tin".
            Default: "idw".
        window_size: int — IDW search window in cells. Default: 6.
        power: float — IDW power parameter. Default: 2.0.
        nodata: float — Nodata value. Default: -9999.0.
        gdaldriver: str — Output driver (only "GTiff" supported). Default: "GTiff".
        gdalopts: str — GDAL creation options (e.g., "COMPRESS=LZW,TILED=YES").
    """

    def write(self, pc: PointCloud, path: str, **options: Any) -> int:
        resolution = float(options.get("resolution", self.default_options.get("resolution", 1.0)))
        method = options.get("output_type", self.default_options.get("output_type", "idw"))
        window_size = int(options.get("window_size", self.default_options.get("window_size", 6)))
        power = float(options.get("power", self.default_options.get("power", 2.0)))
        nodata = float(options.get("nodata", self.default_options.get("nodata", -9999.0)))

        # Parse compression from gdalopts
        gdalopts = options.get("gdalopts", self.default_options.get("gdalopts", "COMPRESS=LZW,TILED=YES"))
        compress = "lzw"
        if gdalopts:
            for opt in gdalopts.split(","):
                if opt.strip().upper().startswith("COMPRESS="):
                    compress = opt.split("=", 1)[1].strip().lower()

        raster, transform = create_dem(
            pc,
            resolution=resolution,
            method=method,
            nodata=nodata,
            power=power,
            window_size=window_size,
        )

        write_geotiff(raster, path, transform, nodata=nodata, compress=compress)

        return pc.num_points

    def __init__(self, **options: Any) -> None:
        self.default_options = options

    @classmethod
    def extensions(cls) -> list[str]:
        return [".tif", ".tiff"]

    @classmethod
    def type_name(cls) -> str:
        return "writers.gdal"
