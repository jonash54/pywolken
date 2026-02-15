"""Tests for GeoTIFF export."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pywolken.raster.export import write_geotiff


class TestWriteGeotiff:
    def test_basic_export(self):
        arr = np.ones((50, 50), dtype=np.float32) * 100.0
        transform = {
            "xmin": 419000.0,
            "ymax": 5622000.0,
            "resolution": 1.0,
            "nrows": 50,
            "ncols": 50,
            "crs": "EPSG:25832",
        }
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            write_geotiff(arr, path, transform)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

            # Read back with rasterio
            import rasterio
            with rasterio.open(path) as src:
                assert src.width == 50
                assert src.height == 50
                data = src.read(1)
                np.testing.assert_allclose(data, 100.0)
                assert src.crs.to_epsg() == 25832
        finally:
            Path(path).unlink(missing_ok=True)

    def test_uint8_hillshade(self):
        arr = np.full((30, 30), 128, dtype=np.uint8)
        transform = {
            "xmin": 0.0, "ymax": 30.0, "resolution": 1.0,
            "nrows": 30, "ncols": 30, "crs": None,
        }
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            write_geotiff(arr, path, transform, nodata=0, dtype="uint8")
            import rasterio
            with rasterio.open(path) as src:
                assert src.dtypes[0] == "uint8"
        finally:
            Path(path).unlink(missing_ok=True)
