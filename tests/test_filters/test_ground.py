"""Tests for ground classification and HAG filters."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.ground import GroundFilter
from pywolken.filters.hag import HagFilter


@pytest.fixture
def terrain_with_buildings():
    """Simulated terrain: flat ground at Z=0 with a building at Z=10."""
    rng = np.random.default_rng(42)
    n_ground = 500
    n_building = 100

    # Ground points: flat at Z=0 with small noise
    gx = rng.uniform(0, 100, n_ground)
    gy = rng.uniform(0, 100, n_ground)
    gz = rng.normal(0, 0.1, n_ground)

    # Building points: elevated block at Z=10
    bx = rng.uniform(40, 60, n_building)
    by = rng.uniform(40, 60, n_building)
    bz = np.full(n_building, 10.0) + rng.normal(0, 0.1, n_building)

    pc = PointCloud.from_dict({
        "X": np.concatenate([gx, bx]),
        "Y": np.concatenate([gy, by]),
        "Z": np.concatenate([gz, bz]),
    })
    return pc, n_ground, n_building


class TestGroundFilter:
    def test_flat_terrain_all_ground(self):
        """On flat terrain, all points should be classified as ground."""
        rng = np.random.default_rng(42)
        n = 200
        pc = PointCloud.from_dict({
            "X": rng.uniform(0, 100, n),
            "Y": rng.uniform(0, 100, n),
            "Z": rng.normal(0, 0.1, n),
        })
        f = GroundFilter(cell_size=2.0, threshold=1.0)
        result = f.filter(pc)
        ground_count = np.sum(result["Classification"] == 2)
        # Most points should be ground (>80%)
        assert ground_count / n > 0.8

    def test_building_detection(self, terrain_with_buildings):
        """Building points should not be classified as ground."""
        pc, n_ground, n_building = terrain_with_buildings
        f = GroundFilter(cell_size=2.0, threshold=1.0, slope=0.15)
        result = f.filter(pc)

        # Check ground points (first n_ground)
        ground_cls = result["Classification"][:n_ground]
        ground_correct = np.sum(ground_cls == 2)

        # Check building points (last n_building)
        building_cls = result["Classification"][n_ground:]
        building_non_ground = np.sum(building_cls != 2)

        # Most ground should be classified correctly
        assert ground_correct / n_ground > 0.7
        # Most building points should NOT be ground
        assert building_non_ground / n_building > 0.7

    def test_classification_dimension_created(self):
        """Filter should create Classification dim if not present."""
        pc = PointCloud.from_dict({
            "X": np.array([0.0, 1.0, 2.0]),
            "Y": np.array([0.0, 1.0, 2.0]),
            "Z": np.array([0.0, 0.0, 0.0]),
        })
        f = GroundFilter()
        result = f.filter(pc)
        assert "Classification" in result

    def test_type_name(self):
        assert GroundFilter.type_name() == "filters.ground"


class TestHagFilter:
    def test_ground_points_zero_hag(self):
        """Ground points should have HAG close to 0."""
        rng = np.random.default_rng(42)
        n = 100
        pc = PointCloud.from_dict({
            "X": rng.uniform(0, 100, n),
            "Y": rng.uniform(0, 100, n),
            "Z": np.zeros(n),
            "Classification": np.full(n, 2, dtype=np.uint8),
        })
        f = HagFilter()
        result = f.filter(pc)
        assert "HeightAboveGround" in result
        np.testing.assert_allclose(result["HeightAboveGround"], 0.0, atol=0.01)

    def test_elevated_points(self):
        """Points above ground should have positive HAG."""
        x = np.array([0.0, 0.0, 50.0, 50.0, 25.0])
        y = np.array([0.0, 50.0, 0.0, 50.0, 25.0])
        z = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
        cls = np.array([2, 2, 2, 2, 1], dtype=np.uint8)

        pc = PointCloud.from_dict({
            "X": x, "Y": y, "Z": z, "Classification": cls,
        })
        f = HagFilter()
        result = f.filter(pc)
        # Last point is 10m above ground
        assert result["HeightAboveGround"][4] == pytest.approx(10.0, abs=0.5)

    def test_no_ground_raises(self):
        pc = PointCloud.from_dict({
            "X": np.array([0.0]),
            "Y": np.array([0.0]),
            "Z": np.array([0.0]),
            "Classification": np.array([1], dtype=np.uint8),
        })
        f = HagFilter()
        with pytest.raises(ValueError, match="No ground"):
            f.filter(pc)

    def test_no_classification_raises(self):
        pc = PointCloud.from_dict({
            "X": np.array([0.0]),
            "Y": np.array([0.0]),
            "Z": np.array([0.0]),
        })
        f = HagFilter()
        with pytest.raises(ValueError, match="Classification"):
            f.filter(pc)

    def test_type_name(self):
        assert HagFilter.type_name() == "filters.hag"
