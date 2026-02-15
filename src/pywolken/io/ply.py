"""PLY reader/writer — Stanford PLY format (ASCII + binary)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.core.metadata import Metadata
from pywolken.io.base import Reader, Writer

# PLY type name → (numpy dtype, struct format char, byte size)
_PLY_TYPES: dict[str, tuple[np.dtype, str, int]] = {
    "char": (np.dtype("int8"), "b", 1),
    "uchar": (np.dtype("uint8"), "B", 1),
    "short": (np.dtype("int16"), "h", 2),
    "ushort": (np.dtype("uint16"), "H", 2),
    "int": (np.dtype("int32"), "i", 4),
    "uint": (np.dtype("uint32"), "I", 4),
    "float": (np.dtype("float32"), "f", 4),
    "double": (np.dtype("float64"), "d", 8),
    # Aliases
    "int8": (np.dtype("int8"), "b", 1),
    "uint8": (np.dtype("uint8"), "B", 1),
    "int16": (np.dtype("int16"), "h", 2),
    "uint16": (np.dtype("uint16"), "H", 2),
    "int32": (np.dtype("int32"), "i", 4),
    "uint32": (np.dtype("uint32"), "I", 4),
    "float32": (np.dtype("float32"), "f", 4),
    "float64": (np.dtype("float64"), "d", 8),
}

# Map common PLY property names to pywolken dimension names
_PLY_DIM_MAP: dict[str, str] = {
    "x": "X", "y": "Y", "z": "Z",
    "nx": "NormalX", "ny": "NormalY", "nz": "NormalZ",
    "red": "Red", "green": "Green", "blue": "Blue",
    "alpha": "Alpha",
    "intensity": "Intensity",
    "scalar_intensity": "Intensity",
    "classification": "Classification",
    "scalar_classification": "Classification",
    "return_number": "ReturnNumber",
    "number_of_returns": "NumberOfReturns",
    "gps_time": "GpsTime",
}

# Reverse map for writing
_DIM_PLY_MAP: dict[str, str] = {
    "X": "x", "Y": "y", "Z": "z",
    "NormalX": "nx", "NormalY": "ny", "NormalZ": "nz",
    "Red": "red", "Green": "green", "Blue": "blue",
    "Alpha": "alpha",
    "Intensity": "intensity",
    "Classification": "classification",
    "ReturnNumber": "return_number",
    "NumberOfReturns": "number_of_returns",
    "GpsTime": "gps_time",
}

# Map numpy dtypes to PLY type names for writing
_NUMPY_TO_PLY: dict[str, str] = {
    "int8": "char",
    "uint8": "uchar",
    "int16": "short",
    "uint16": "ushort",
    "int32": "int",
    "uint32": "uint",
    "int64": "int",      # downcast
    "uint64": "uint",    # downcast
    "float32": "float",
    "float64": "double",
}


def _parse_header(f) -> tuple[str, int, list[tuple[str, str]]]:
    """Parse PLY header, return (format, vertex_count, properties)."""
    magic = f.readline().strip()
    if magic != b"ply":
        raise ValueError("Not a PLY file (missing 'ply' magic)")

    fmt = "ascii"
    vertex_count = 0
    properties: list[tuple[str, str]] = []  # (name, ply_type)
    in_vertex = False

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected end of PLY header")
        line = line.strip().decode("ascii", errors="replace")

        if line == "end_header":
            break

        parts = line.split()
        if not parts:
            continue

        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and parts[1] == "vertex":
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                continue  # Skip list properties (face data etc.)
            ply_type = parts[1]
            prop_name = parts[2]
            properties.append((prop_name, ply_type))

    return fmt, vertex_count, properties


class PlyReader(Reader):
    """Read PLY files (ASCII and binary_little_endian)."""

    def read(self, path: str, **options: Any) -> PointCloud:
        with open(path, "rb") as f:
            fmt, vertex_count, properties = _parse_header(f)

            if fmt == "ascii":
                return self._read_ascii(f, vertex_count, properties, path)
            elif fmt == "binary_little_endian":
                return self._read_binary(f, vertex_count, properties, "<", path)
            elif fmt == "binary_big_endian":
                return self._read_binary(f, vertex_count, properties, ">", path)
            else:
                raise ValueError(f"Unsupported PLY format: {fmt}")

    def _read_ascii(
        self, f, vertex_count: int,
        properties: list[tuple[str, str]], path: str,
    ) -> PointCloud:
        # Pre-allocate arrays
        arrays: dict[str, np.ndarray] = {}
        prop_info: list[tuple[str, np.dtype]] = []

        for prop_name, ply_type in properties:
            if ply_type not in _PLY_TYPES:
                raise ValueError(f"Unknown PLY type: {ply_type}")
            dtype = _PLY_TYPES[ply_type][0]
            dim_name = _PLY_DIM_MAP.get(prop_name.lower(), prop_name)
            arrays[dim_name] = np.empty(vertex_count, dtype=dtype)
            prop_info.append((dim_name, dtype))

        for i in range(vertex_count):
            line = f.readline().decode("ascii", errors="replace").strip()
            values = line.split()
            for j, (dim_name, dtype) in enumerate(prop_info):
                arrays[dim_name][i] = dtype.type(values[j])

        pc = PointCloud.from_dict(arrays)
        pc._metadata = Metadata(
            source_file=str(path),
            source_format="ply",
            software="pywolken",
        )
        return pc

    def _read_binary(
        self, f, vertex_count: int,
        properties: list[tuple[str, str]],
        endian: str, path: str,
    ) -> PointCloud:
        # Build struct format and array list
        fmt_chars = []
        prop_info: list[tuple[str, np.dtype]] = []

        for prop_name, ply_type in properties:
            if ply_type not in _PLY_TYPES:
                raise ValueError(f"Unknown PLY type: {ply_type}")
            dtype, fmt_char, _ = _PLY_TYPES[ply_type]
            dim_name = _PLY_DIM_MAP.get(prop_name.lower(), prop_name)
            fmt_chars.append(fmt_char)
            prop_info.append((dim_name, dtype))

        struct_fmt = endian + "".join(fmt_chars)
        row_size = struct.calcsize(struct_fmt)

        # Read all vertex data at once
        data = f.read(row_size * vertex_count)
        if len(data) < row_size * vertex_count:
            raise ValueError(
                f"Incomplete PLY data: expected {row_size * vertex_count} bytes, "
                f"got {len(data)}"
            )

        # Pre-allocate arrays
        arrays: dict[str, np.ndarray] = {}
        for dim_name, dtype in prop_info:
            arrays[dim_name] = np.empty(vertex_count, dtype=dtype)

        # Unpack rows
        for i in range(vertex_count):
            offset = i * row_size
            values = struct.unpack_from(struct_fmt, data, offset)
            for j, (dim_name, _) in enumerate(prop_info):
                arrays[dim_name][i] = values[j]

        pc = PointCloud.from_dict(arrays)
        pc._metadata = Metadata(
            source_file=str(path),
            source_format="ply",
            software="pywolken",
        )
        return pc

    def read_chunked(
        self, path: str, chunk_size: int = 1_000_000, **options: Any
    ) -> Iterator[PointCloud]:
        # For PLY we read all and chunk in memory
        pc = self.read(path, **options)
        yield from pc.iter_chunks(chunk_size)

    @classmethod
    def extensions(cls) -> list[str]:
        return [".ply"]

    @classmethod
    def type_name(cls) -> str:
        return "readers.ply"


class PlyWriter(Writer):
    """Write PLY files (ASCII or binary_little_endian)."""

    def write(self, pc: PointCloud, path: str, **options: Any) -> int:
        if pc.num_points == 0:
            raise ValueError("Cannot write empty point cloud")

        fmt = options.get("format", "binary_little_endian")
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format: {fmt}")

        with open(path, "wb") as f:
            # Build property list
            props: list[tuple[str, str, np.dtype]] = []
            for dim_name in pc.dimensions:
                ply_name = _DIM_PLY_MAP.get(dim_name, dim_name.lower())
                dtype_name = pc[dim_name].dtype.name
                ply_type = _NUMPY_TO_PLY.get(dtype_name, "float")
                props.append((ply_name, ply_type, pc[dim_name].dtype))

            # Write header
            header_lines = [
                "ply",
                f"format {fmt} 1.0",
                f"element vertex {pc.num_points}",
            ]
            for ply_name, ply_type, _ in props:
                header_lines.append(f"property {ply_type} {ply_name}")
            header_lines.append("end_header")
            header = "\n".join(header_lines) + "\n"
            f.write(header.encode("ascii"))

            if fmt == "ascii":
                self._write_ascii(f, pc, props)
            else:
                endian = "<" if fmt == "binary_little_endian" else ">"
                self._write_binary(f, pc, props, endian)

        return pc.num_points

    def _write_ascii(self, f, pc: PointCloud, props):
        dim_names = pc.dimensions
        # Write each row as space-separated values
        for i in range(pc.num_points):
            values = []
            for dim_name in dim_names:
                v = pc[dim_name][i]
                if np.issubdtype(pc[dim_name].dtype, np.floating):
                    values.append(f"{v:.10g}")
                else:
                    values.append(str(int(v)))
            line = " ".join(values) + "\n"
            f.write(line.encode("ascii"))

    def _write_binary(self, f, pc: PointCloud, props, endian: str):
        # Build struct format
        fmt_chars = []
        for _, ply_type, _ in props:
            _, fmt_char, _ = _PLY_TYPES[ply_type]
            fmt_chars.append(fmt_char)

        struct_fmt = endian + "".join(fmt_chars)
        dim_names = pc.dimensions

        # Pack all rows
        for i in range(pc.num_points):
            values = []
            for dim_name in dim_names:
                v = pc[dim_name][i]
                values.append(v.item())  # Convert numpy scalar to Python
            f.write(struct.pack(struct_fmt, *values))

    @classmethod
    def extensions(cls) -> list[str]:
        return [".ply"]

    @classmethod
    def type_name(cls) -> str:
        return "writers.ply"
