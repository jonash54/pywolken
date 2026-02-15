"""Tests for the crop filter."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.crop import CropFilter, _parse_bounds


class TestParseBounds:
    def test_2d(self):
        pairs = _parse_bounds("([0, 100], [0, 200])")
        assert len(pairs) == 2
        assert pairs[0] == (0.0, 100.0)
        assert pairs[1] == (0.0, 200.0)

    def test_3d(self):
        pairs = _parse_bounds("([0, 100], [0, 200], [10, 50])")
        assert len(pairs) == 3
        assert pairs[2] == (10.0, 50.0)

    def test_with_spaces(self):
        pairs = _parse_bounds("( [ 0 , 100 ] , [ 0 , 200 ] )")
        assert pairs[0] == (0.0, 100.0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid bounds"):
            _parse_bounds("not valid")


class TestCropFilter:
    def test_2d_crop(self):
        pc = PointCloud.from_dict({
            "X": np.array([0.0, 50.0, 100.0, 150.0]),
            "Y": np.array([0.0, 50.0, 100.0, 150.0]),
            "Z": np.array([0.0, 0.0, 0.0, 0.0]),
        })
        f = CropFilter(bounds="([0, 100], [0, 100])")
        result = f.filter(pc)
        assert result.num_points == 3  # Points at 0, 50, 100 are inside

    def test_3d_crop(self):
        pc = PointCloud.from_dict({
            "X": np.array([50.0, 50.0, 50.0]),
            "Y": np.array([50.0, 50.0, 50.0]),
            "Z": np.array([5.0, 25.0, 45.0]),
        })
        f = CropFilter(bounds="([0, 100], [0, 100], [10, 30])")
        result = f.filter(pc)
        assert result.num_points == 1
        assert result["Z"][0] == 25.0

    def test_explicit_bounds(self):
        pc = PointCloud.from_dict({
            "X": np.array([0.0, 50.0, 100.0]),
            "Y": np.array([0.0, 50.0, 100.0]),
            "Z": np.array([0.0, 0.0, 0.0]),
        })
        f = CropFilter(minx=10, maxx=90, miny=10, maxy=90)
        result = f.filter(pc)
        assert result.num_points == 1

    def test_missing_bounds_raises(self):
        with pytest.raises(ValueError, match="bounds"):
            CropFilter()

    def test_type_name(self):
        assert CropFilter.type_name() == "filters.crop"

    def test_crop_on_sample(self, sample_pc):
        bounds = sample_pc.bounds
        mid_x = (bounds.minx + bounds.maxx) / 2
        mid_y = (bounds.miny + bounds.maxy) / 2
        f = CropFilter(bounds=f"([{bounds.minx}, {mid_x}], [{bounds.miny}, {mid_y}])")
        result = f.filter(sample_pc)
        assert 0 < result.num_points < sample_pc.num_points
