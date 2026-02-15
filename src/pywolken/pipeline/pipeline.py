"""Pipeline — parse and execute JSON processing pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import get_filter
from pywolken.io.base import Reader, Writer
from pywolken.io.registry import _ensure_registered as _ensure_io, io_registry

logger = logging.getLogger(__name__)


class Pipeline:
    """JSON-based processing pipeline, compatible with PDAL's pipeline format.

    Parses a JSON pipeline definition, then executes it as:
      reader(s) → filter chain → writer(s)

    Examples:
        >>> import json
        >>> p = Pipeline(json.dumps({
        ...     "pipeline": [
        ...         "input.laz",
        ...         {"type": "filters.range", "limits": "Classification[2:2]"},
        ...         "output.las"
        ...     ]
        ... }))
        >>> count = p.execute()
    """

    def __init__(
        self,
        json_str: str | None = None,
        stages: list | None = None,
    ) -> None:
        self._readers: list[tuple[Reader, str, dict[str, Any]]] = []
        self._filters: list[Filter] = []
        self._writers: list[tuple[Writer, str, dict[str, Any]]] = []
        self._result: PointCloud | None = None
        self._count: int = 0

        if json_str is not None:
            self._parse_json(json_str)
        elif stages is not None:
            self._parse_stages(stages)

    def _parse_json(self, json_str: str) -> None:
        """Parse a JSON pipeline definition."""
        data = json.loads(json_str)

        if isinstance(data, dict):
            stages = data.get("pipeline", [])
        elif isinstance(data, list):
            stages = data
        else:
            raise ValueError("Pipeline JSON must be a dict with 'pipeline' key or a list")

        self._parse_stages(stages)

    def _parse_stages(self, stages: list) -> None:
        """Parse a list of stage definitions into reader/filter/writer objects."""
        _ensure_io()

        for stage in stages:
            if isinstance(stage, str):
                # Bare filename string — auto-detect reader or writer
                self._parse_filename_stage(stage)
            elif isinstance(stage, dict):
                self._parse_dict_stage(stage)
            else:
                raise ValueError(f"Invalid pipeline stage: {stage!r}")

    def _parse_filename_stage(self, filename: str) -> None:
        """Parse a bare filename string into a reader or writer."""
        # If we already have at least one reader, subsequent bare filenames
        # are treated as writers (matching PDAL behavior).
        if self._readers:
            writer = io_registry.get_writer(filename)
            self._writers.append((writer, filename, {}))
        else:
            reader = io_registry.get_reader(filename)
            self._readers.append((reader, filename, {}))

    def _parse_dict_stage(self, stage: dict[str, Any]) -> None:
        """Parse a dict stage definition."""
        stage_type = stage.get("type", "")
        options = {k: v for k, v in stage.items() if k not in ("type", "filename")}

        if stage_type.startswith("readers."):
            filename = stage.get("filename", "")
            if not filename:
                raise ValueError(f"Reader stage missing 'filename': {stage}")
            reader = io_registry.get_reader_by_type(stage_type)
            self._readers.append((reader, filename, options))

        elif stage_type.startswith("writers."):
            filename = stage.get("filename", "")
            if not filename:
                raise ValueError(f"Writer stage missing 'filename': {stage}")
            writer = io_registry.get_writer_by_type(stage_type)
            self._writers.append((writer, filename, options))

        elif stage_type.startswith("filters."):
            filt = get_filter(stage_type, **options)
            self._filters.append(filt)

        else:
            # Try to infer from filename
            filename = stage.get("filename", "")
            if filename:
                self._parse_filename_stage(filename)
            else:
                raise ValueError(f"Cannot determine stage type: {stage}")

    def validate(self) -> list[str]:
        """Validate the pipeline configuration.

        Returns:
            List of error messages (empty = valid).
        """
        errors = []
        if not self._readers:
            errors.append("Pipeline has no readers")
        if not self._writers:
            errors.append("Pipeline has no writers")
        return errors

    def execute(self) -> int:
        """Execute the pipeline: read → filter → write.

        Returns:
            Total number of points processed.
        """
        # 1. Read from all readers and merge
        clouds: list[PointCloud] = []
        for reader, path, options in self._readers:
            logger.info("Reading %s", path)
            pc = reader.read(path, **options)
            logger.info("  Read %d points", pc.num_points)
            clouds.append(pc)

        if not clouds:
            raise RuntimeError("No point clouds read")

        # Merge if multiple readers
        result = clouds[0]
        for other in clouds[1:]:
            result = result.merge(other)

        # 2. Apply filters in sequence
        for filt in self._filters:
            logger.info("Applying %s", filt)
            before = result.num_points
            result = filt.filter(result)
            logger.info("  %d → %d points", before, result.num_points)

        # 3. Write to all writers
        for writer, path, options in self._writers:
            logger.info("Writing %s", path)
            writer.write(result, path, **options)
            logger.info("  Wrote %d points", result.num_points)

        self._result = result
        self._count = result.num_points
        return self._count

    @property
    def arrays(self) -> dict[str, np.ndarray] | None:
        """Point data from the last execution (dict of arrays)."""
        if self._result is None:
            return None
        return self._result.to_dict()

    @property
    def result(self) -> PointCloud | None:
        """PointCloud result from the last execution."""
        return self._result

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata from the last execution."""
        if self._result is None:
            return {}
        m = self._result.metadata
        return {
            "source_file": m.source_file,
            "source_format": m.source_format,
            "point_count": self._count,
            "dimensions": self._result.dimensions if self._result else [],
        }

    def to_json(self) -> str:
        """Serialize the pipeline back to JSON."""
        stages: list[Any] = []

        for reader, path, options in self._readers:
            if options:
                stage = {"type": reader.type_name(), "filename": path, **options}
            else:
                stages.append(path)
                continue
            stages.append(stage)

        for filt in self._filters:
            stage = {"type": filt.type_name(), **filt.options}
            stages.append(stage)

        for writer, path, options in self._writers:
            if options:
                stage = {"type": writer.type_name(), "filename": path, **options}
            else:
                stages.append(path)
                continue
            stages.append(stage)

        return json.dumps({"pipeline": stages}, indent=2)

    def __repr__(self) -> str:
        parts = (
            f"{len(self._readers)} reader(s), "
            f"{len(self._filters)} filter(s), "
            f"{len(self._writers)} writer(s)"
        )
        return f"Pipeline({parts})"
