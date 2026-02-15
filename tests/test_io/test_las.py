"""Tests for LAS/LAZ reader and writer."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.io.las import LasReader, LasWriter
from pywolken.io.registry import read, write


class TestLasWriter:
    def test_write_and_read_roundtrip(self, sample_pc):
        """Write a PointCloud to LAS and read it back."""
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            path = f.name

        try:
            writer = LasWriter()
            count = writer.write(sample_pc, path)
            assert count == 100

            reader = LasReader()
            result = reader.read(path)
            assert result.num_points == 100
            np.testing.assert_allclose(result["X"], sample_pc["X"], atol=0.001)
            np.testing.assert_allclose(result["Y"], sample_pc["Y"], atol=0.001)
            np.testing.assert_allclose(result["Z"], sample_pc["Z"], atol=0.001)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_write_laz_roundtrip(self, sample_pc):
        """Write to LAZ (compressed) and read back."""
        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as f:
            path = f.name

        try:
            write(sample_pc, path)
            result = read(path)
            assert result.num_points == 100
            np.testing.assert_allclose(result["X"], sample_pc["X"], atol=0.001)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_write_empty_raises(self):
        pc = PointCloud()
        writer = LasWriter()
        with tempfile.NamedTemporaryFile(suffix=".las") as f:
            with pytest.raises(ValueError, match="empty"):
                writer.write(pc, f.name)

    def test_metadata_preserved(self, sample_pc):
        """Check metadata survives write→read roundtrip."""
        with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
            path = f.name

        try:
            write(sample_pc, path)
            result = read(path)
            assert result.metadata.source_format == "las"
            assert result.metadata.source_file is not None
        finally:
            Path(path).unlink(missing_ok=True)


class TestLasReaderReal:
    """Integration tests with real LAZ files (skipped if not available)."""

    def test_read_real_laz(self, real_laz_path):
        """Read a real LAZ file from /var/geodata/."""
        pc = read(str(real_laz_path))
        assert pc.num_points > 0
        assert "X" in pc
        assert "Y" in pc
        assert "Z" in pc
        assert "Classification" in pc
        print(f"\nRead {pc.num_points:,} points from {real_laz_path.name}")
        print(f"  Dimensions: {pc.dimensions}")
        print(f"  Bounds: {pc.bounds}")

    def test_read_chunked_real_laz(self, real_laz_path):
        """Read a real LAZ file in chunks."""
        reader = LasReader()
        total = 0
        chunk_count = 0
        for chunk in reader.read_chunked(str(real_laz_path), chunk_size=500_000):
            total += chunk.num_points
            chunk_count += 1
            if chunk_count >= 3:
                break  # Don't read the whole file in tests
        assert total > 0
        print(f"\nRead {total:,} points in {chunk_count} chunks")
