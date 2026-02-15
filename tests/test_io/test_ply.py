"""Tests for PLY reader/writer."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.io.ply import PlyReader, PlyWriter


@pytest.fixture
def small_pc():
    """Small point cloud with XYZ + color."""
    return PointCloud.from_dict({
        "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "Y": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "Z": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "Red": np.array([255, 128, 0, 64, 200], dtype=np.uint16),
        "Green": np.array([0, 128, 255, 64, 100], dtype=np.uint16),
        "Blue": np.array([100, 50, 200, 150, 0], dtype=np.uint16),
    })


class TestPlyWriter:
    def test_write_ascii_roundtrip(self, small_pc, tmp_path):
        path = str(tmp_path / "test.ply")
        writer = PlyWriter()
        n = writer.write(small_pc, path, format="ascii")
        assert n == 5

        reader = PlyReader()
        result = reader.read(path)
        assert result.num_points == 5
        np.testing.assert_allclose(result["X"], small_pc["X"], atol=1e-4)
        np.testing.assert_allclose(result["Y"], small_pc["Y"], atol=1e-4)
        np.testing.assert_allclose(result["Z"], small_pc["Z"], atol=1e-4)

    def test_write_binary_roundtrip(self, small_pc, tmp_path):
        path = str(tmp_path / "test.ply")
        writer = PlyWriter()
        n = writer.write(small_pc, path, format="binary_little_endian")
        assert n == 5

        reader = PlyReader()
        result = reader.read(path)
        assert result.num_points == 5
        np.testing.assert_allclose(result["X"], small_pc["X"], atol=1e-4)
        np.testing.assert_allclose(result["Y"], small_pc["Y"], atol=1e-4)

    def test_write_empty_raises(self, tmp_path):
        path = str(tmp_path / "test.ply")
        writer = PlyWriter()
        with pytest.raises(ValueError, match="empty"):
            writer.write(PointCloud(), path)

    def test_preserves_dimensions(self, small_pc, tmp_path):
        path = str(tmp_path / "test.ply")
        PlyWriter().write(small_pc, path, format="ascii")
        result = PlyReader().read(path)
        # PLY maps to lowercase names, check values are preserved
        assert result.num_points == small_pc.num_points
        assert len(result.dimensions) == len(small_pc.dimensions)

    def test_metadata(self, small_pc, tmp_path):
        path = str(tmp_path / "test.ply")
        PlyWriter().write(small_pc, path, format="ascii")
        result = PlyReader().read(path)
        assert result.metadata.source_format == "ply"


class TestPlyReader:
    def test_type_name(self):
        assert PlyReader.type_name() == "readers.ply"

    def test_extensions(self):
        assert ".ply" in PlyReader.extensions()

    def test_writer_type_name(self):
        assert PlyWriter.type_name() == "writers.ply"

    def test_not_ply_raises(self, tmp_path):
        path = str(tmp_path / "bad.ply")
        with open(path, "wb") as f:
            f.write(b"not a ply file\n")
        with pytest.raises(ValueError, match="Not a PLY"):
            PlyReader().read(path)
