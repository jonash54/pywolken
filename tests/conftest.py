"""Shared test fixtures."""

from pathlib import Path

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud


@pytest.fixture
def sample_pc() -> PointCloud:
    """A small PointCloud with 100 points for testing."""
    rng = np.random.default_rng(42)
    pc = PointCloud()
    pc._arrays["X"] = rng.uniform(400000, 401000, 100)
    pc._arrays["Y"] = rng.uniform(5600000, 5601000, 100)
    pc._arrays["Z"] = rng.uniform(100, 500, 100)
    pc._arrays["Intensity"] = rng.integers(0, 65535, 100, dtype=np.uint16)
    pc._arrays["Classification"] = rng.choice(
        [1, 2, 3, 6], size=100
    ).astype(np.uint8)
    return pc


@pytest.fixture
def ground_pc() -> PointCloud:
    """A PointCloud with mixed ground (2) and non-ground points."""
    pc = PointCloud()
    n = 50
    pc._arrays["X"] = np.arange(n, dtype=np.float64)
    pc._arrays["Y"] = np.arange(n, dtype=np.float64)
    pc._arrays["Z"] = np.arange(n, dtype=np.float64)
    # First 20 are ground (class 2), rest are vegetation (class 3)
    cls = np.full(n, 3, dtype=np.uint8)
    cls[:20] = 2
    pc._arrays["Classification"] = cls
    return pc


# Path to real LAZ test data (if available)
REAL_LAZ_DIR = Path("/var/geodata")
REAL_LAZ_FILES = list(REAL_LAZ_DIR.glob("*.laz")) if REAL_LAZ_DIR.exists() else []


@pytest.fixture
def real_laz_path() -> Path | None:
    """Path to a real LAZ file for integration tests (smallest one)."""
    if not REAL_LAZ_FILES:
        pytest.skip("No real LAZ files available in /var/geodata/")
    # Use the smallest file
    return min(REAL_LAZ_FILES, key=lambda p: p.stat().st_size)
