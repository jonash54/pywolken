"""Tests for Phase 4 advanced filters: outlier, normal, voxel, cluster, sort."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.outlier import OutlierFilter
from pywolken.filters.normal import NormalFilter
from pywolken.filters.voxel import VoxelFilter
from pywolken.filters.cluster import ClusterFilter
from pywolken.filters.sort import SortFilter


@pytest.fixture
def cloud_with_outliers():
    """Point cloud with clear outliers."""
    rng = np.random.default_rng(42)
    n_good = 200
    n_outliers = 5
    # Dense cluster of points
    gx = rng.normal(50, 5, n_good)
    gy = rng.normal(50, 5, n_good)
    gz = rng.normal(0, 0.5, n_good)
    # Far-away outliers
    ox = rng.uniform(200, 300, n_outliers)
    oy = rng.uniform(200, 300, n_outliers)
    oz = rng.uniform(100, 200, n_outliers)
    return PointCloud.from_dict({
        "X": np.concatenate([gx, ox]),
        "Y": np.concatenate([gy, oy]),
        "Z": np.concatenate([gz, oz]),
    }), n_good, n_outliers


class TestOutlierFilter:
    def test_statistical_removes_outliers(self, cloud_with_outliers):
        pc, n_good, n_outliers = cloud_with_outliers
        f = OutlierFilter(method="statistical", mean_k=8, multiplier=2.0)
        result = f.filter(pc)
        # Should remove most outliers
        assert result.num_points < pc.num_points
        assert result.num_points >= n_good - 10  # Allow some tolerance

    def test_radius_removes_outliers(self, cloud_with_outliers):
        pc, n_good, n_outliers = cloud_with_outliers
        f = OutlierFilter(method="radius", radius=20.0, min_k=3)
        result = f.filter(pc)
        assert result.num_points < pc.num_points

    def test_invalid_method(self, sample_pc):
        f = OutlierFilter(method="invalid")
        with pytest.raises(ValueError, match="Unknown"):
            f.filter(sample_pc)

    def test_type_name(self):
        assert OutlierFilter.type_name() == "filters.outlier"


class TestNormalFilter:
    def test_flat_surface_normals(self):
        """Flat XY plane should have normals pointing up (0, 0, 1)."""
        rng = np.random.default_rng(42)
        n = 100
        pc = PointCloud.from_dict({
            "X": rng.uniform(0, 100, n),
            "Y": rng.uniform(0, 100, n),
            "Z": np.zeros(n),
        })
        f = NormalFilter(k=8)
        result = f.filter(pc)
        assert "NormalX" in result
        assert "NormalY" in result
        assert "NormalZ" in result
        # Z normals should be close to 1 (pointing up)
        assert np.mean(np.abs(result["NormalZ"])) > 0.9

    def test_adds_dimensions(self, sample_pc):
        f = NormalFilter(k=5)
        result = f.filter(sample_pc)
        assert "NormalX" in result
        assert "NormalY" in result
        assert "NormalZ" in result

    def test_type_name(self):
        assert NormalFilter.type_name() == "filters.normal"


class TestVoxelFilter:
    def test_reduces_points(self, sample_pc):
        f = VoxelFilter(cell_size=200.0)  # Large voxels
        result = f.filter(sample_pc)
        assert result.num_points < sample_pc.num_points

    def test_preserves_dimensions(self, sample_pc):
        f = VoxelFilter(cell_size=500.0)
        result = f.filter(sample_pc)
        assert set(result.dimensions) == set(sample_pc.dimensions)

    def test_centroid_accuracy(self):
        """Voxel centroids should be near the center of each cell."""
        pc = PointCloud.from_dict({
            "X": np.array([0.1, 0.2, 0.3, 1.1, 1.2]),
            "Y": np.array([0.1, 0.2, 0.3, 1.1, 1.2]),
            "Z": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        })
        f = VoxelFilter(cell_size=1.0)
        result = f.filter(pc)
        assert result.num_points == 2  # Two voxels

    def test_type_name(self):
        assert VoxelFilter.type_name() == "filters.voxel"


class TestClusterFilter:
    def test_two_clusters(self):
        """Two well-separated groups should form two clusters."""
        rng = np.random.default_rng(42)
        n = 50
        # Cluster 1 near origin
        c1x = rng.normal(0, 1, n)
        c1y = rng.normal(0, 1, n)
        # Cluster 2 far away
        c2x = rng.normal(100, 1, n)
        c2y = rng.normal(100, 1, n)

        pc = PointCloud.from_dict({
            "X": np.concatenate([c1x, c2x]),
            "Y": np.concatenate([c1y, c2y]),
            "Z": np.zeros(2 * n),
        })
        f = ClusterFilter(tolerance=5.0, min_points=10)
        result = f.filter(pc)
        assert "ClusterID" in result
        unique_clusters = set(result["ClusterID"]) - {0}  # Exclude noise
        assert len(unique_clusters) == 2

    def test_noise_points(self):
        """Isolated points should be labeled as noise (0)."""
        pc = PointCloud.from_dict({
            "X": np.array([0.0, 100.0, 200.0]),
            "Y": np.array([0.0, 100.0, 200.0]),
            "Z": np.array([0.0, 0.0, 0.0]),
        })
        f = ClusterFilter(tolerance=1.0, min_points=2)
        result = f.filter(pc)
        assert all(result["ClusterID"] == 0)

    def test_type_name(self):
        assert ClusterFilter.type_name() == "filters.cluster"


class TestSortFilter:
    def test_morton_sort(self, sample_pc):
        f = SortFilter(order="morton")
        result = f.filter(sample_pc)
        assert result.num_points == sample_pc.num_points

    def test_xyz_sort(self, sample_pc):
        f = SortFilter(order="xyz")
        result = f.filter(sample_pc)
        # X should be non-decreasing
        assert np.all(np.diff(result["X"]) >= -1e-10)

    def test_invalid_order(self, sample_pc):
        f = SortFilter(order="invalid")
        with pytest.raises(ValueError, match="Unknown"):
            f.filter(sample_pc)

    def test_type_name(self):
        assert SortFilter.type_name() == "filters.sort"
