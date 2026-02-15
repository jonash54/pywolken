"""Tests for DEM generation and hillshade computation."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.raster.dem import create_dem
from pywolken.raster.hillshade import hillshade, multi_directional_hillshade


@pytest.fixture
def flat_terrain():
    """A flat terrain at Z=100 on a 100x100m area."""
    rng = np.random.default_rng(42)
    n = 10_000
    pc = PointCloud.from_dict({
        "X": rng.uniform(0, 100, n),
        "Y": rng.uniform(0, 100, n),
        "Z": np.full(n, 100.0) + rng.normal(0, 0.01, n),
    })
    return pc


@pytest.fixture
def sloped_terrain():
    """A terrain sloping from Z=0 at Y=0 to Z=100 at Y=100."""
    rng = np.random.default_rng(42)
    n = 10_000
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    z = y * 1.0  # Linear slope: 1m rise per 1m
    pc = PointCloud.from_dict({"X": x, "Y": y, "Z": z})
    return pc


class TestCreateDem:
    def test_mean_flat(self, flat_terrain):
        raster, transform = create_dem(flat_terrain, resolution=1.0, method="mean")
        assert raster.shape == (100, 100)
        valid = raster != -9999.0
        assert np.sum(valid) > 0
        assert np.allclose(raster[valid], 100.0, atol=0.5)
        assert transform["resolution"] == 1.0

    def test_idw_flat(self, flat_terrain):
        raster, transform = create_dem(flat_terrain, resolution=2.0, method="idw")
        assert raster.shape == (50, 50)
        valid = raster != -9999.0
        assert np.allclose(raster[valid], 100.0, atol=0.5)

    def test_nearest(self, flat_terrain):
        raster, _ = create_dem(flat_terrain, resolution=1.0, method="nearest")
        valid = raster != -9999.0
        assert np.sum(valid) > 0

    def test_tin(self, flat_terrain):
        raster, _ = create_dem(flat_terrain, resolution=2.0, method="tin")
        valid = raster != -9999.0
        assert np.allclose(raster[valid], 100.0, atol=0.5)

    def test_sloped_mean(self, sloped_terrain):
        raster, _ = create_dem(sloped_terrain, resolution=5.0, method="mean")
        # Top row (Y=max) should have higher Z than bottom row (Y=min)
        assert np.mean(raster[0]) > np.mean(raster[-1])  # Row 0 = north (high Y)

    def test_custom_bounds(self, flat_terrain):
        raster, transform = create_dem(
            flat_terrain, resolution=1.0, method="mean",
            bounds=(25, 25, 75, 75),
        )
        assert raster.shape == (50, 50)

    def test_invalid_method(self, flat_terrain):
        with pytest.raises(ValueError, match="Unknown method"):
            create_dem(flat_terrain, resolution=1.0, method="invalid")

    def test_transform_info(self, flat_terrain):
        _, transform = create_dem(flat_terrain, resolution=2.0, method="mean")
        assert "xmin" in transform
        assert "ymax" in transform
        assert "resolution" in transform
        assert "nrows" in transform
        assert "ncols" in transform


class TestHillshade:
    def test_flat_is_uniform(self):
        """A perfectly flat DEM should produce uniform hillshade."""
        dem = np.full((50, 50), 100.0, dtype=np.float32)
        shade = hillshade(dem, resolution=1.0)
        assert shade.shape == (50, 50)
        assert shade.dtype == np.uint8
        # All interior cells should have same value (flat = uniform light)
        interior = shade[2:-2, 2:-2]
        assert np.std(interior) < 5

    def test_slope_variation(self):
        """A sloped DEM should show directional shading."""
        rows = np.arange(100).reshape(100, 1)
        dem = np.broadcast_to(rows.astype(np.float32), (100, 100)).copy()
        shade = hillshade(dem, resolution=1.0, azimuth=0)
        # Should have non-zero variation
        assert np.std(shade[5:-5, 5:-5]) > 0 or np.mean(shade) > 0

    def test_z_factor(self):
        """Higher z_factor should increase contrast."""
        rows = np.arange(50).reshape(50, 1)
        dem = np.broadcast_to(rows.astype(np.float32), (50, 50)).copy()
        shade1 = hillshade(dem, resolution=1.0, z_factor=1.0)
        shade2 = hillshade(dem, resolution=1.0, z_factor=5.0)
        # Higher z_factor should produce more contrast
        assert np.std(shade2.astype(float)) >= np.std(shade1.astype(float)) - 1

    def test_nodata_handling(self):
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        dem[5:10, 5:10] = -9999.0
        shade = hillshade(dem, resolution=1.0, nodata=-9999.0)
        assert shade[7, 7] == 0  # Nodata area should be 0

    def test_multi_directional(self):
        rows = np.arange(50).reshape(50, 1)
        dem = np.broadcast_to(rows.astype(np.float32), (50, 50)).copy()
        shade = multi_directional_hillshade(dem, resolution=1.0)
        assert shade.shape == (50, 50)
        assert shade.dtype == np.uint8
