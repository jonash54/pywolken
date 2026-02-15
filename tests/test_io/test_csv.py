"""Tests for CSV reader/writer."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.io.csv import CsvReader, CsvWriter


@pytest.fixture
def small_pc():
    return PointCloud.from_dict({
        "X": np.array([1.0, 2.0, 3.0]),
        "Y": np.array([4.0, 5.0, 6.0]),
        "Z": np.array([7.0, 8.0, 9.0]),
        "Classification": np.array([2, 3, 6], dtype=np.int64),
    })


class TestCsvWriter:
    def test_write_read_roundtrip(self, small_pc, tmp_path):
        path = str(tmp_path / "test.csv")
        writer = CsvWriter()
        n = writer.write(small_pc, path)
        assert n == 3

        reader = CsvReader()
        result = reader.read(path)
        assert result.num_points == 3
        np.testing.assert_allclose(result["X"], small_pc["X"])
        np.testing.assert_allclose(result["Y"], small_pc["Y"])
        np.testing.assert_allclose(result["Z"], small_pc["Z"])

    def test_semicolon_delimiter(self, small_pc, tmp_path):
        path = str(tmp_path / "test.csv")
        CsvWriter().write(small_pc, path, delimiter=";")
        result = CsvReader().read(path, delimiter=";")
        assert result.num_points == 3
        np.testing.assert_allclose(result["X"], [1.0, 2.0, 3.0])

    def test_tab_delimiter(self, small_pc, tmp_path):
        path = str(tmp_path / "test.csv")
        CsvWriter().write(small_pc, path, delimiter="\t")
        result = CsvReader().read(path, delimiter="\t")
        assert result.num_points == 3

    def test_no_header(self, small_pc, tmp_path):
        path = str(tmp_path / "test.csv")
        CsvWriter().write(small_pc, path, header=False)
        # Read with explicit header
        result = CsvReader().read(path, header="X,Y,Z,Classification")
        assert result.num_points == 3
        np.testing.assert_allclose(result["X"], [1.0, 2.0, 3.0])

    def test_auto_detect_comma(self, small_pc, tmp_path):
        path = str(tmp_path / "test.csv")
        CsvWriter().write(small_pc, path, delimiter=",")
        result = CsvReader().read(path)  # auto-detect
        assert result.num_points == 3

    def test_precision(self, tmp_path):
        pc = PointCloud.from_dict({
            "X": np.array([1.123456789]),
            "Y": np.array([2.0]),
            "Z": np.array([3.0]),
        })
        path = str(tmp_path / "test.csv")
        CsvWriter().write(pc, path, precision=3)
        result = CsvReader().read(path)
        assert abs(result["X"][0] - 1.123) < 0.001

    def test_integer_detection(self, tmp_path):
        pc = PointCloud.from_dict({
            "X": np.array([1.0, 2.0]),
            "Y": np.array([3.0, 4.0]),
            "Z": np.array([5.0, 6.0]),
            "Classification": np.array([2, 3], dtype=np.int64),
        })
        path = str(tmp_path / "test.csv")
        CsvWriter().write(pc, path)
        result = CsvReader().read(path)
        # Classification values should be preserved as integers
        assert result["Classification"][0] == 2
        assert result["Classification"][1] == 3

    def test_write_empty_raises(self, tmp_path):
        path = str(tmp_path / "test.csv")
        with pytest.raises(ValueError, match="empty"):
            CsvWriter().write(PointCloud(), path)

    def test_metadata(self, small_pc, tmp_path):
        path = str(tmp_path / "test.csv")
        CsvWriter().write(small_pc, path)
        result = CsvReader().read(path)
        assert result.metadata.source_format == "csv"

    def test_xyz_extension(self):
        assert ".xyz" in CsvReader.extensions()
        assert ".txt" in CsvReader.extensions()

    def test_type_names(self):
        assert CsvReader.type_name() == "readers.csv"
        assert CsvWriter.type_name() == "writers.csv"
