"""Tests for CLI."""

import json

import numpy as np
import pytest

from pywolken.cli import main
from pywolken.core.pointcloud import PointCloud
from pywolken.io.las import LasWriter


@pytest.fixture
def sample_las(tmp_path):
    """Create a small LAS file for CLI tests."""
    pc = PointCloud.from_dict({
        "X": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "Y": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "Z": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "Classification": np.array([2, 2, 3, 3, 6], dtype=np.uint8),
    })
    path = str(tmp_path / "test.las")
    LasWriter().write(pc, path)
    return path


class TestCliInfo:
    def test_info_command(self, sample_las, capsys):
        ret = main(["info", sample_las])
        assert ret == 0
        output = capsys.readouterr().out
        assert "Points: 5" in output
        assert "X" in output

    def test_info_missing_file(self, capsys):
        ret = main(["info", "/nonexistent/file.las"])
        assert ret == 1

    def test_info_shows_bounds(self, sample_las, capsys):
        main(["info", sample_las])
        output = capsys.readouterr().out
        assert "Bounds X:" in output
        assert "Bounds Y:" in output
        assert "Bounds Z:" in output


class TestCliConvert:
    def test_convert_las_to_csv(self, sample_las, tmp_path, capsys):
        output = str(tmp_path / "output.csv")
        ret = main(["convert", sample_las, output])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "Converted 5 points" in captured

    def test_convert_las_to_ply(self, sample_las, tmp_path, capsys):
        output = str(tmp_path / "output.ply")
        ret = main(["convert", sample_las, output])
        assert ret == 0

    def test_convert_missing_input(self, tmp_path, capsys):
        ret = main(["convert", "/nonexistent.las", str(tmp_path / "out.csv")])
        assert ret == 1


class TestCliPipeline:
    def test_pipeline_command(self, sample_las, tmp_path, capsys):
        output_path = str(tmp_path / "output.las")
        pipeline = {
            "pipeline": [
                sample_las,
                {"type": "filters.range", "limits": "Classification[2:2]"},
                output_path,
            ]
        }
        pipeline_file = str(tmp_path / "pipeline.json")
        with open(pipeline_file, "w") as f:
            json.dump(pipeline, f)

        ret = main(["pipeline", pipeline_file])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "Processed" in captured

    def test_pipeline_missing_file(self, capsys):
        ret = main(["pipeline", "/nonexistent.json"])
        assert ret == 1


class TestCliMerge:
    def test_merge_command(self, sample_las, tmp_path, capsys):
        # Create a second file
        pc = PointCloud.from_dict({
            "X": np.array([6.0, 7.0]),
            "Y": np.array([60.0, 70.0]),
            "Z": np.array([0.6, 0.7]),
            "Classification": np.array([2, 3], dtype=np.uint8),
        })
        las2 = str(tmp_path / "test2.las")
        LasWriter().write(pc, las2)

        output = str(tmp_path / "merged.las")
        ret = main(["merge", sample_las, las2, "-o", output])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "Merged 2 files" in captured
        assert "7 points" in captured


class TestCliVersion:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_no_command_shows_help(self, capsys):
        ret = main([])
        assert ret == 0
