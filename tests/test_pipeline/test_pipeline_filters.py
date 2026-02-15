"""Tests for pipelines with filters."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pywolken.io.registry import write
from pywolken.pipeline.pipeline import Pipeline


@pytest.fixture
def las_file(sample_pc) -> str:
    with tempfile.NamedTemporaryFile(suffix=".las", delete=False) as f:
        path = f.name
    write(sample_pc, path)
    yield path
    Path(path).unlink(missing_ok=True)


class TestPipelineWithFilters:
    def test_range_filter(self, las_file):
        """Pipeline: read → range filter → write."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    las_file,
                    {"type": "filters.range", "limits": "Classification[2:2]"},
                    output,
                ]
            })
            p = Pipeline(spec)
            count = p.execute()
            assert 0 < count < 100
            assert all(p.result["Classification"] == 2)
        finally:
            Path(output).unlink(missing_ok=True)

    def test_crop_filter(self, las_file, sample_pc):
        """Pipeline: read → crop → write."""
        bounds = sample_pc.bounds
        mid_x = (bounds.minx + bounds.maxx) / 2
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    las_file,
                    {
                        "type": "filters.crop",
                        "bounds": f"([{bounds.minx}, {mid_x}], [{bounds.miny}, {bounds.maxy}])",
                    },
                    output,
                ]
            })
            p = Pipeline(spec)
            count = p.execute()
            assert 0 < count < 100
        finally:
            Path(output).unlink(missing_ok=True)

    def test_decimation_filter(self, las_file):
        """Pipeline: read → decimation → write."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    las_file,
                    {"type": "filters.decimation", "step": 10},
                    output,
                ]
            })
            p = Pipeline(spec)
            count = p.execute()
            assert count == 10
        finally:
            Path(output).unlink(missing_ok=True)

    def test_expression_filter(self, las_file):
        """Pipeline: read → expression → write."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    las_file,
                    {"type": "filters.expression", "expression": "Classification == 2"},
                    output,
                ]
            })
            p = Pipeline(spec)
            count = p.execute()
            assert 0 < count < 100
        finally:
            Path(output).unlink(missing_ok=True)

    def test_chained_filters(self, las_file):
        """Pipeline: read → range → decimation → write."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    las_file,
                    {"type": "filters.range", "limits": "Classification[2:2]"},
                    {"type": "filters.decimation", "step": 2},
                    output,
                ]
            })
            p = Pipeline(spec)
            count = p.execute()
            # Should be half of ground points
            assert count > 0
            assert all(p.result["Classification"] == 2)
        finally:
            Path(output).unlink(missing_ok=True)

    def test_assign_filter(self, las_file):
        """Pipeline: read → assign → write."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    las_file,
                    {"type": "filters.assign", "assignment": "Classification=2"},
                    output,
                ]
            })
            p = Pipeline(spec)
            count = p.execute()
            assert count == 100
            assert all(p.result["Classification"] == 2)
        finally:
            Path(output).unlink(missing_ok=True)

    def test_pipeline_serialization_with_filters(self, las_file):
        """Pipeline with filters can be serialized to JSON."""
        output = tempfile.mktemp(suffix=".las")
        try:
            spec = json.dumps({
                "pipeline": [
                    las_file,
                    {"type": "filters.range", "limits": "Classification[2:2]"},
                    {"type": "filters.decimation", "step": 5},
                    output,
                ]
            })
            p = Pipeline(spec)
            json_out = p.to_json()
            data = json.loads(json_out)
            assert len(data["pipeline"]) == 4
        finally:
            Path(output).unlink(missing_ok=True)
