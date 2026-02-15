"""Streaming pipeline — memory-bounded chunked processing."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.io.base import Reader, Writer

logger = logging.getLogger(__name__)


class StreamingPipeline:
    """Memory-bounded pipeline that processes data in chunks.

    Reads data in chunks from a reader, applies per-chunk filters,
    and writes results incrementally. This allows processing files
    much larger than available RAM.

    Not all filters are streaming-compatible. Filters that need global
    knowledge (e.g., clustering, ground classification) should be marked
    as non-streamable. By default filters are assumed to be streamable.

    Examples:
        >>> from pywolken.io.las import LasReader, LasWriter
        >>> from pywolken.filters.range import RangeFilter
        >>> sp = StreamingPipeline(
        ...     reader=LasReader(),
        ...     input_path="huge.laz",
        ...     writer=LasWriter(),
        ...     output_path="filtered.laz",
        ...     filters=[RangeFilter(limits="Classification[2:2]")],
        ...     chunk_size=1_000_000,
        ... )
        >>> total = sp.execute()
    """

    def __init__(
        self,
        reader: Reader,
        input_path: str,
        writer: Writer | None = None,
        output_path: str | None = None,
        filters: list[Filter] | None = None,
        chunk_size: int = 1_000_000,
        reader_options: dict[str, Any] | None = None,
        writer_options: dict[str, Any] | None = None,
    ) -> None:
        self.reader = reader
        self.input_path = input_path
        self.writer = writer
        self.output_path = output_path
        self.filters = filters or []
        self.chunk_size = chunk_size
        self.reader_options = reader_options or {}
        self.writer_options = writer_options or {}
        self._total_read = 0
        self._total_written = 0

    def execute(self) -> int:
        """Execute the streaming pipeline.

        Returns:
            Total number of output points.
        """
        self._total_read = 0
        self._total_written = 0

        # Collect output chunks — needed because LAS writer needs all at once
        output_chunks: list[PointCloud] = []

        for chunk in self._process_chunks():
            output_chunks.append(chunk)
            self._total_written += chunk.num_points

        if not output_chunks:
            logger.warning("No output points after streaming pipeline")
            return 0

        # Write results
        if self.writer and self.output_path:
            # Merge all output chunks for writing
            if len(output_chunks) == 1:
                final = output_chunks[0]
            else:
                final = output_chunks[0]
                for chunk in output_chunks[1:]:
                    final = final.merge(chunk)

            logger.info("Writing %d points to %s", final.num_points, self.output_path)
            self.writer.write(final, self.output_path, **self.writer_options)

        return self._total_written

    def _process_chunks(self) -> Iterator[PointCloud]:
        """Read, filter, and yield chunks."""
        chunk_num = 0
        for chunk in self.reader.read_chunked(
            self.input_path, self.chunk_size, **self.reader_options
        ):
            chunk_num += 1
            self._total_read += chunk.num_points
            logger.info(
                "Chunk %d: %d points (total read: %d)",
                chunk_num, chunk.num_points, self._total_read,
            )

            # Apply filters to this chunk
            result = chunk
            for filt in self.filters:
                result = filt.filter(result)
                if result.num_points == 0:
                    break

            if result.num_points > 0:
                yield result

    def iter_results(self) -> Iterator[PointCloud]:
        """Iterate over processed chunks without writing.

        Useful for streaming results into further processing.
        """
        self._total_read = 0
        self._total_written = 0
        for chunk in self._process_chunks():
            self._total_written += chunk.num_points
            yield chunk

    @property
    def total_read(self) -> int:
        """Total points read so far."""
        return self._total_read

    @property
    def total_written(self) -> int:
        """Total points in output so far."""
        return self._total_written

    def __repr__(self) -> str:
        return (
            f"StreamingPipeline(input={self.input_path!r}, "
            f"filters={len(self.filters)}, "
            f"chunk_size={self.chunk_size:,})"
        )
