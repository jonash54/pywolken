"""Raster export — GeoTIFF via rasterio."""

from __future__ import annotations

from typing import Any

import numpy as np


def write_geotiff(
    array: np.ndarray,
    path: str,
    transform_info: dict,
    nodata: float | None = -9999.0,
    compress: str = "lzw",
    dtype: str | None = None,
) -> None:
    """Write a 2D array to a GeoTIFF file.

    Args:
        array: 2D numpy array to write.
        path: Output file path.
        transform_info: Dict with keys from create_dem():
            'xmin', 'ymax', 'resolution', 'nrows', 'ncols', 'crs'.
        nodata: Nodata value (None to omit).
        compress: Compression method ("lzw", "deflate", "none").
        dtype: Output dtype (default: auto from array).
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise ImportError(
            "rasterio is required for GeoTIFF export. "
            "Install it with: pip install pywolken[raster]"
        )

    res = transform_info["resolution"]
    transform = from_origin(
        transform_info["xmin"],
        transform_info["ymax"],
        res, res,
    )

    # Determine CRS
    crs_str = transform_info.get("crs")
    rasterio_crs = None
    if crs_str:
        try:
            rasterio_crs = rasterio.crs.CRS.from_user_input(crs_str)
        except Exception:
            pass

    out_dtype = dtype or array.dtype
    profile = {
        "driver": "GTiff",
        "dtype": out_dtype,
        "width": array.shape[1],
        "height": array.shape[0],
        "count": 1,
        "crs": rasterio_crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    if compress and compress.lower() != "none":
        profile["compress"] = compress

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
