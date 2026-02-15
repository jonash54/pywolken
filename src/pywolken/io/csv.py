"""CSV/text reader/writer — delimited point cloud data."""

from __future__ import annotations

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
                # User-provided column names
                columns = [c.strip() for c in header_str.split(",")]
            else:
                # Read first line as header
                first_line = f.readline().strip()
                if delimiter is None:
                    delimiter = self._detect_delimiter(first_line)
                columns = [c.strip() for c in first_line.split(delimiter)]

            # Read remaining lines
            lines = f.readlines()

        if delimiter is None and lines:
            delimiter = self._detect_delimiter(lines[0])

        # Parse data
        n = len(lines)
        arrays: dict[str, list] = {col: [] for col in columns}

        for line in lines:
            line = line.strip()
            if not line:
                continue
            values = line.split(delimiter)
            for j, col in enumerate(columns):
                if j < len(values):
                    arrays[col].append(values[j].strip())
                else:
                    arrays[col].append("0")

        # Convert to numpy arrays with appropriate types
        np_arrays: dict[str, np.ndarray] = {}
        for col, vals in arrays.items():
            if not vals:
                np_arrays[col] = np.array([], dtype=np.float64)
                continue
            try:
                # Try float first (handles both int and float values)
                arr = np.array(vals, dtype=np.float64)
                # If all values are integer-valued, use int
                if np.all(arr == arr.astype(np.int64)) and not any("." in v for v in vals):
                    arr = arr.astype(np.int64)
                np_arrays[col] = arr
            except ValueError:
                # Keep as string (unusual for point clouds)
                np_arrays[col] = np.array(vals, dtype=np.float64)

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

        with open(path, "w") as f:
            dims = pc.dimensions

            if write_header:
                f.write(delimiter.join(dims) + "\n")

            for i in range(pc.num_points):
                values = []
                for dim in dims:
                    v = pc[dim][i]
                    if np.issubdtype(pc[dim].dtype, np.floating):
                        values.append(f"{v:.{precision}f}")
                    else:
                        values.append(str(int(v)))
                f.write(delimiter.join(values) + "\n")

        return pc.num_points

    @classmethod
    def extensions(cls) -> list[str]:
        return [".csv", ".txt", ".xyz"]

    @classmethod
    def type_name(cls) -> str:
        return "writers.csv"
