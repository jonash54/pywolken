"""I/O registry — format auto-detection and convenience functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from pywolken.core.pointcloud import PointCloud
from pywolken.io.base import Reader, Writer


class IORegistry:
    """Registry for readers and writers, with format auto-detection."""

    def __init__(self) -> None:
        self._readers: dict[str, type[Reader]] = {}  # extension -> Reader class
        self._writers: dict[str, type[Writer]] = {}
        self._reader_types: dict[str, type[Reader]] = {}  # type_name -> Reader class
        self._writer_types: dict[str, type[Writer]] = {}

    def register_reader(self, cls: type[Reader]) -> None:
        """Register a reader class for its extensions."""
        for ext in cls.extensions():
            self._readers[ext.lower()] = cls
        self._reader_types[cls.type_name()] = cls

    def register_writer(self, cls: type[Writer]) -> None:
        """Register a writer class for its extensions."""
        for ext in cls.extensions():
            self._writers[ext.lower()] = cls
        self._writer_types[cls.type_name()] = cls

    def get_reader(self, path: str) -> Reader:
        """Get a reader instance for the given file path."""
        ext = Path(path).suffix.lower()
        if ext not in self._readers:
            raise ValueError(
                f"No reader for extension '{ext}'. "
                f"Supported: {list(self._readers.keys())}"
            )
        return self._readers[ext]()

    def get_writer(self, path: str) -> Writer:
        """Get a writer instance for the given file path."""
        ext = Path(path).suffix.lower()
        if ext not in self._writers:
            raise ValueError(
                f"No writer for extension '{ext}'. "
                f"Supported: {list(self._writers.keys())}"
            )
        return self._writers[ext]()

    def get_reader_by_type(self, type_name: str) -> Reader:
        """Get a reader by its pipeline type name (e.g., 'readers.las')."""
        if type_name not in self._reader_types:
            raise ValueError(
                f"Unknown reader type '{type_name}'. "
                f"Available: {list(self._reader_types.keys())}"
            )
        return self._reader_types[type_name]()

    def get_writer_by_type(self, type_name: str) -> Writer:
        """Get a writer by its pipeline type name (e.g., 'writers.las')."""
        if type_name not in self._writer_types:
            raise ValueError(
                f"Unknown writer type '{type_name}'. "
                f"Available: {list(self._writer_types.keys())}"
            )
        return self._writer_types[type_name]()


# Global registry instance
io_registry = IORegistry()


def _ensure_registered() -> None:
    """Register built-in readers/writers (lazy, on first use)."""
    if io_registry._readers:
        return

    from pywolken.io.las import LasReader, LasWriter
    from pywolken.io.gdal import GdalWriter
    from pywolken.io.ply import PlyReader, PlyWriter
    from pywolken.io.csv import CsvReader, CsvWriter

    io_registry.register_reader(LasReader)
    io_registry.register_writer(LasWriter)
    io_registry.register_writer(GdalWriter)
    io_registry.register_reader(PlyReader)
    io_registry.register_writer(PlyWriter)
    io_registry.register_reader(CsvReader)
    io_registry.register_writer(CsvWriter)


def read(path: str, **options: Any) -> PointCloud:
    """Read a point cloud file (auto-detects format).

    Args:
        path: File path to read.
        **options: Format-specific options.

    Returns:
        PointCloud with all dimensions from the file.
    """
    _ensure_registered()
    reader = io_registry.get_reader(path)
    return reader.read(path, **options)


def read_chunked(
    path: str, chunk_size: int = 1_000_000, **options: Any
) -> Iterator[PointCloud]:
    """Read a point cloud file in chunks (auto-detects format).

    Args:
        path: File path to read.
        chunk_size: Points per chunk.
        **options: Format-specific options.

    Yields:
        PointCloud chunks.
    """
    _ensure_registered()
    reader = io_registry.get_reader(path)
    yield from reader.read_chunked(path, chunk_size, **options)


def write(pc: PointCloud, path: str, **options: Any) -> int:
    """Write a point cloud to file (auto-detects format from extension).

    Args:
        pc: PointCloud to write.
        path: Output file path.
        **options: Format-specific options.

    Returns:
        Number of points written.
    """
    _ensure_registered()
    writer = io_registry.get_writer(path)
    return writer.write(pc, path, **options)
