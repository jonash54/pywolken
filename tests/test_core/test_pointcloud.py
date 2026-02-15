"""Tests for PointCloud core data model."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.core.bounds import Bounds


class TestPointCloudCreation:
    def test_empty(self):
        pc = PointCloud()
        assert pc.num_points == 0
        assert pc.dimensions == []
        assert len(pc) == 0

    def test_set_get_dimension(self):
        pc = PointCloud()
        x = np.array([1.0, 2.0, 3.0])
        pc["X"] = x
        np.testing.assert_array_equal(pc["X"], x)
        assert "X" in pc
        assert len(pc) == 3

    def test_set_mismatched_length_raises(self):
        pc = PointCloud()
        pc["X"] = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="length"):
            pc["Y"] = np.array([1.0, 2.0])

    def test_get_missing_raises(self):
        pc = PointCloud()
        with pytest.raises(KeyError, match="not found"):
            _ = pc["X"]

    def test_from_dict(self):
        data = {
            "X": np.array([1.0, 2.0]),
            "Y": np.array([3.0, 4.0]),
            "Z": np.array([5.0, 6.0]),
        }
        pc = PointCloud.from_dict(data, crs="EPSG:25832")
        assert pc.num_points == 2
        assert set(pc.dimensions) == {"X", "Y", "Z"}
        assert pc.crs == "EPSG:25832"

    def test_from_dict_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            PointCloud.from_dict({"X": np.array([1.0]), "Y": np.array([1.0, 2.0])})

    def test_from_numpy(self):
        dtype = np.dtype([("X", np.float64), ("Y", np.float64), ("Z", np.float64)])
        arr = np.array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], dtype=dtype)
        pc = PointCloud.from_numpy(arr)
        assert pc.num_points == 2
        np.testing.assert_array_equal(pc["X"], [1.0, 4.0])


class TestPointCloudOperations:
    def test_bounds(self, sample_pc):
        b = sample_pc.bounds
        assert isinstance(b, Bounds)
        assert b.minx <= b.maxx
        assert b.miny <= b.maxy
        assert b.minz <= b.maxz

    def test_mask(self, sample_pc):
        mask = sample_pc["Classification"] == 2
        filtered = sample_pc.mask(mask)
        assert filtered.num_points == int(mask.sum())
        assert all(filtered["Classification"] == 2)

    def test_mask_wrong_length_raises(self, sample_pc):
        with pytest.raises(ValueError, match="Mask length"):
            sample_pc.mask(np.array([True, False]))

    def test_slice(self, sample_pc):
        sliced = sample_pc.slice(10, 20)
        assert sliced.num_points == 10
        np.testing.assert_array_equal(sliced["X"], sample_pc["X"][10:20])

    def test_iter_chunks(self, sample_pc):
        chunks = list(sample_pc.iter_chunks(30))
        assert len(chunks) == 4  # 100 / 30 = 3.33 -> 4 chunks
        assert chunks[0].num_points == 30
        assert chunks[-1].num_points == 10

    def test_copy(self, sample_pc):
        copy = sample_pc.copy()
        assert copy.num_points == sample_pc.num_points
        # Modify copy, original unchanged
        copy["X"][0] = -999
        assert sample_pc["X"][0] != -999

    def test_merge(self, sample_pc):
        merged = sample_pc.merge(sample_pc)
        assert merged.num_points == 200

    def test_add_dimension(self, sample_pc):
        sample_pc.add_dimension("NewDim", np.dtype(np.float32), fill=42.0)
        assert "NewDim" in sample_pc
        assert sample_pc["NewDim"][0] == 42.0

    def test_add_existing_raises(self, sample_pc):
        with pytest.raises(ValueError, match="already exists"):
            sample_pc.add_dimension("X")

    def test_remove_dimension(self, sample_pc):
        sample_pc.remove_dimension("Intensity")
        assert "Intensity" not in sample_pc

    def test_to_numpy(self, sample_pc):
        arr = sample_pc.to_numpy()
        assert arr.dtype.names is not None
        assert "X" in arr.dtype.names
        assert len(arr) == 100

    def test_to_dict(self, sample_pc):
        d = sample_pc.to_dict()
        assert isinstance(d, dict)
        assert "X" in d

    def test_repr(self, sample_pc):
        r = repr(sample_pc)
        assert "100" in r
        assert "PointCloud" in r


class TestBounds:
    def test_from_arrays(self):
        x = np.array([1.0, 5.0])
        y = np.array([2.0, 6.0])
        z = np.array([3.0, 7.0])
        b = Bounds.from_arrays(x, y, z)
        assert b.minx == 1.0
        assert b.maxx == 5.0
        assert b.width == 4.0
        assert b.height == 4.0
        assert b.depth == 4.0

    def test_contains_point(self):
        b = Bounds(0, 0, 0, 10, 10, 10)
        assert b.contains_point(5, 5, 5)
        assert not b.contains_point(15, 5, 5)

    def test_intersects(self):
        b1 = Bounds(0, 0, 0, 10, 10, 10)
        b2 = Bounds(5, 5, 5, 15, 15, 15)
        b3 = Bounds(20, 20, 20, 30, 30, 30)
        assert b1.intersects(b2)
        assert not b1.intersects(b3)

    def test_union(self):
        b1 = Bounds(0, 0, 0, 10, 10, 10)
        b2 = Bounds(5, 5, 5, 20, 20, 20)
        u = b1.union(b2)
        assert u.minx == 0
        assert u.maxx == 20
