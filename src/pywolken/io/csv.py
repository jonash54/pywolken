"""CSV/text reader/writer — delimited point cloud data."""

from __future__ import annotations

import io as _io
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.core.metadata import Metadata
from pywolken.io.base import Reader, Writer


class CsvReader(Reader):
    """Read CSV/TXT point cloud files.

    Options:
        delimiter: str — Field delimiter (default: auto-detect from ',', ';', '\\t', ' ').
        header: str — Comma-separated column names if file has no header row.
            E.g., "X,Y,Z,Intensity". If not given, first row is used as header.
        skip: int — Number of header lines to skip before data (default: 0).
    """

    def read(self, path: str, **options: Any) -> PointCloud:
        delimiter = options.get("delimiter")
        header_str = options.get("header")
        skip = int(options.get("skip", 0))

        with open(path, "r") as f:
            # Skip leading lines
            for _ in range(skip):
                f.readline()

            if header_str:
                columns = [c.strip() for c in header_str.split(",")]
            else:
                first_line = f.readline().strip()
                if delimiter is None:
                    delimiter = self._detect_delimiter(first_line)
                columns = [c.strip() for c in first_line.split(delimiter)]

            # Read remaining content as a single string
            remaining = f.read()

        if delimiter is None and remaining:
            delimiter = self._detect_delimiter(remaining.split("\n", 1)[0])

        # Bulk parse with np.loadtxt — orders of magnitude faster than row-by-row
        if not remaining.strip():
            np_arrays = {col: np.array([], dtype=np.float64) for col in columns}
        else:
            data = np.loadtxt(
                _io.StringIO(remaining),
                delimiter=delimiter,
                dtype=np.float64,
                ndmin=2,
            )
            np_arrays: dict[str, np.ndarray] = {}
            for j, col in enumerate(columns):
                if j < data.shape[1]:
                    arr = data[:, j]
                    # Detect integer columns (no fractional parts in source)
                    if np.all(arr == np.floor(arr)):
                        # Check original text for decimal points in this column
                        first_line = remaining.strip().split("\n", 1)[0]
                        vals = first_line.split(delimiter)
                        if j < len(vals) and "." not in vals[j].strip():
                            arr = arr.astype(np.int64)
                    np_arrays[col] = arr
                else:
                    np_arrays[col] = np.zeros(data.shape[0], dtype=np.float64)

        pc = PointCloud.from_dict(np_arrays)
        pc._metadata = Metadata(
            source_file=str(path),
            source_format="csv",
            software="pywolken",
        )
        return pc

    def _detect_delimiter(self, line: str) -> str:
        """Auto-detect the delimiter from a sample line."""
        for delim in [",", ";", "\t"]:
            if delim in line:
                return delim
        return " "

    def read_chunked(
        self, path: str, chunk_size: int = 1_000_000, **options: Any
    ) -> Iterator[PointCloud]:
        pc = self.read(path, **options)
        yield from pc.iter_chunks(chunk_size)

    @classmethod
    def extensions(cls) -> list[str]:
        return [".csv", ".txt", ".xyz"]

    @classmethod
    def type_name(cls) -> str:
        return "readers.csv"


class CsvWriter(Writer):
    """Write CSV point cloud files.

    Options:
        delimiter: str — Field delimiter (default: ',').
        header: bool — Write header row (default: True).
        precision: int — Decimal precision for float values (default: 6).
    """

    def write(self, pc: PointCloud, path: str, **options: Any) -> int:
        if pc.num_points == 0:
            raise ValueError("Cannot write empty point cloud")

        delimiter = options.get("delimiter", ",")
        write_header = options.get("header", True)
        precision = int(options.get("precision", 6))

        dims = pc.dimensions

        # Build a 2D structured array and use np.savetxt for bulk writing
        # Collect all columns as float64 for savetxt, with int columns formatted separately
        float_mask = [np.issubdtype(pc[d].dtype, np.floating) for d in dims]
        fmt_parts = []
        for is_float in float_mask:
            if is_float:
                fmt_parts.append(f"%.{precision}f")
            else:
                fmt_parts.append("%d")

        # Stack into a 2D array (all as float64 for np.savetxt compatibility)
        data = np.column_stack([pc[d].astype(np.float64) for d in dims])

        header_line = delimiter.join(dims) if write_header else ""
        np.savetxt(
            path,
            data,
            fmt=fmt_parts,
            delimiter=delimiter,
            header=header_line,
            comments="",  # Don't prefix header with #
        )

        return pc.num_points

    @classmethod
    def extensions(cls) -> list[str]:
        return [".csv", ".txt", ".xyz"]

    @classmethod
    def type_name(cls) -> str:
        return "writers.csv"
