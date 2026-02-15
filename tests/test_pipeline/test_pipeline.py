"""Tests for the pipeline engine."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.io.registry import write
from pywolken.pipeline.pipeline import Pipeline


@pytest.fixture
def las_file(sample_pc) -> str:
    """Write sample_pc to a temporary LAS file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
        path = f.name
    write(sample_pc, path)
    yield path
    Path(path).unlink(missing_ok=True)


class TestPipelineParsing:
    def test_parse_simple_pipeline(self, las_file):
        """Parse a pipeline with reader and writer."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({"pipeline": [las_file, output]})
            p = Pipeline(spec)
            assert len(p._readers) == 1
            assert len(p._writers) == 1
        finally:
            Path(output).unlink(missing_ok=True)

    def test_parse_with_dict_stages(self, las_file):
        """Parse a pipeline with explicit type definitions."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    {"type": "readers.las", "filename": las_file},
                    {"type": "writers.las", "filename": output},
                ]
            })
            p = Pipeline(spec)
            assert len(p._readers) == 1
            assert len(p._writers) == 1
        finally:
            Path(output).unlink(missing_ok=True)

    def test_validate_no_reader(self):
        """Pipeline with no reader should fail validation."""
        p = Pipeline()
        p._writers = [("fake", "out.las", {})]
        errors = p.validate()
        assert any("reader" in e.lower() for e in errors)

    def test_validate_no_writer(self, las_file):
        """Pipeline with no writer should fail validation."""
        spec = json.dumps({"pipeline": [las_file]})
        p = Pipeline(spec)
        errors = p.validate()
        assert any("writer" in e.lower() for e in errors)

    def test_to_json_roundtrip(self, las_file):
        """Pipeline can be serialized back to JSON."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({"pipeline": [las_file, output]})
            p = Pipeline(spec)
            json_out = p.to_json()
            data = json.loads(json_out)
            assert "pipeline" in data
        finally:
            Path(output).unlink(missing_ok=True)


class TestPipelineExecution:
    def test_execute_read_write(self, las_file):
        """Execute a simple read→write pipeline."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({"pipeline": [las_file, output]})
            p = Pipeline(spec)
            count = p.execute()
            assert count == 100
            assert Path(output).exists()
            assert p.result is not None
            assert p.result.num_points == 100
        finally:
            Path(output).unlink(missing_ok=True)

    def test_execute_read_write_laz(self, las_file):
        """Execute a pipeline writing to LAZ format."""
        output = tempfile.mktemp(suffix=".laz")
        try:
            spec = json.dumps({"pipeline": [las_file, output]})
            p = Pipeline(spec)
            count = p.execute()
            assert count == 100
            assert Path(output).exists()
        finally:
            Path(output).unlink(missing_ok=True)

    def test_arrays_property(self, las_file):
        """Access point arrays after execution."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({"pipeline": [las_file, output]})
            p = Pipeline(spec)
            p.execute()
            arrays = p.arrays
            assert arrays is not None
            assert "X" in arrays
            assert len(arrays["X"]) == 100
        finally:
            Path(output).unlink(missing_ok=True)

    def test_metadata_property(self, las_file):
        """Access metadata after execution."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({"pipeline": [las_file, output]})
            p = Pipeline(spec)
            p.execute()
            meta = p.metadata
            assert meta["point_count"] == 100
        finally:
            Path(output).unlink(missing_ok=True)

    def test_repr(self, las_file):
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({"pipeline": [las_file, output]})
            p = Pipeline(spec)
            r = repr(p)
            assert "1 reader" in r
            assert "1 writer" in r
        finally:
            Path(output).unlink(missing_ok=True)
