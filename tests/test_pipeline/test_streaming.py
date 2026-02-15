"""Tests for streaming pipeline and format interop."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.io.las import LasReader, LasWriter
from pywolken.io.ply import PlyReader, PlyWriter
from pywolken.io.csv import CsvReader, CsvWriter
from pywolken.io.registry import read, write
from pywolken.filters.range import RangeFilter
from pywolken.filters.expression import ExpressionFilter
from pywolken.pipeline.streaming import StreamingPipeline


@pytest.fixture
def medium_pc():
    """A medium-sized point cloud for streaming tests."""
    rng = np.random.default_rng(42)
    n = 5000
    return PointCloud.from_dict({
        "X": rng.uniform(400000, 401000, n),
        "Y": rng.uniform(5600000, 5601000, n),
        "Z": rng.uniform(100, 500, n),
        "Classification": rng.choice([1, 2, 3, 6], size=n).astype(np.uint8),
        "Intensity": rng.integers(0, 65535, n, dtype=np.uint16),
    })


class TestStreamingPipeline:
    def test_basic_streaming(self, medium_pc, tmp_path):
        """Streaming read → filter → write."""
        input_path = str(tmp_path / "input.las")
        output_path = str(tmp_path / "output.las")
        LasWriter().write(medium_pc, input_path)

        sp = StreamingPipeline(
            reader=LasReader(),
            input_path=input_path,
            writer=LasWriter(),
            output_path=output_path,
            filters=[RangeFilter(limits="Classification[2:2]")],
            chunk_size=1000,
        )
        total = sp.execute()
        assert total > 0
        assert total < medium_pc.num_points

        # Verify output
        result = LasReader().read(output_path)
        assert result.num_points == total
        assert all(result["Classification"] == 2)

    def test_streaming_no_filter(self, medium_pc, tmp_path):
        """Streaming without filters passes through all points."""
        input_path = str(tmp_path / "input.las")
        output_path = str(tmp_path / "output.las")
        LasWriter().write(medium_pc, input_path)

        sp = StreamingPipeline(
            reader=LasReader(),
            input_path=input_path,
            writer=LasWriter(),
            output_path=output_path,
            chunk_size=2000,
        )
        total = sp.execute()
        assert total == medium_pc.num_points

    def test_streaming_multiple_filters(self, medium_pc, tmp_path):
        """Chain multiple filters in streaming mode."""
        input_path = str(tmp_path / "input.las")
        output_path = str(tmp_path / "output.las")
        LasWriter().write(medium_pc, input_path)

        sp = StreamingPipeline(
            reader=LasReader(),
            input_path=input_path,
            writer=LasWriter(),
            output_path=output_path,
            filters=[
                RangeFilter(limits="Classification[2:2]"),
                ExpressionFilter(expression="Z > 200"),
            ],
            chunk_size=1000,
        )
        total = sp.execute()
        assert total > 0

        result = LasReader().read(output_path)
        assert all(result["Classification"] == 2)
        assert all(result["Z"] > 200)

    def test_iter_results(self, medium_pc, tmp_path):
        """Iterate over results without writing."""
        input_path = str(tmp_path / "input.las")
        LasWriter().write(medium_pc, input_path)

        sp = StreamingPipeline(
            reader=LasReader(),
            input_path=input_path,
            chunk_size=1000,
        )
        chunks = list(sp.iter_results())
        total = sum(c.num_points for c in chunks)
        assert total == medium_pc.num_points
        assert sp.total_read == medium_pc.num_points

    def test_repr(self):
        sp = StreamingPipeline(
            reader=LasReader(),
            input_path="test.laz",
            chunk_size=500_000,
        )
        r = repr(sp)
        assert "test.laz" in r
        assert "500,000" in r


class TestFormatInterop:
    """Test reading/writing across different formats."""

    def test_las_to_ply_ascii(self, medium_pc, tmp_path):
        las_path = str(tmp_path / "test.las")
        ply_path = str(tmp_path / "test.ply")

        LasWriter().write(medium_pc, las_path)
        pc = LasReader().read(las_path)
        PlyWriter().write(pc, ply_path, format="ascii")
        result = PlyReader().read(ply_path)

        assert result.num_points == medium_pc.num_points
        np.testing.assert_allclose(result["X"], pc["X"], atol=1e-3)

    def test_las_to_csv(self, tmp_path):
        pc = PointCloud.from_dict({
            "X": np.array([1.0, 2.0, 3.0]),
            "Y": np.array([4.0, 5.0, 6.0]),
            "Z": np.array([7.0, 8.0, 9.0]),
        })
        las_path = str(tmp_path / "test.las")
        csv_path = str(tmp_path / "test.csv")

        LasWriter().write(pc, las_path)
        pc2 = LasReader().read(las_path)
        CsvWriter().write(pc2, csv_path)
        result = CsvReader().read(csv_path)

        np.testing.assert_allclose(result["X"], [1.0, 2.0, 3.0], atol=0.01)

    def test_registry_auto_detect_ply(self, tmp_path):
        pc = PointCloud.from_dict({
            "X": np.array([1.0, 2.0]),
            "Y": np.array([3.0, 4.0]),
            "Z": np.array([5.0, 6.0]),
        })
        path = str(tmp_path / "test.ply")
        write(pc, path, format="ascii")
        result = read(path)
        assert result.num_points == 2

    def test_registry_auto_detect_csv(self, tmp_path):
        pc = PointCloud.from_dict({
            "X": np.array([1.0, 2.0]),
            "Y": np.array([3.0, 4.0]),
            "Z": np.array([5.0, 6.0]),
        })
        path = str(tmp_path / "test.csv")
        write(pc, path)
        result = read(path)
        assert result.num_points == 2
