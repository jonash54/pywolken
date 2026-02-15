"""Base classes for readers and writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

from pywolken.core.pointcloud import PointCloud


class Reader(ABC):
    """Base class for all point cloud readers."""

    @abstractmethod
    def read(self, path: str, **options: Any) -> PointCloud:
        """Read a point cloud file.

        Args:
            path: File path to read.
            **options: Format-specific options.

        Returns:
            PointCloud with all dimensions from the file.
        """

    def read_chunked(
        self, path: str, chunk_size: int = 1_000_000, **options: Any
    ) -> Iterator[PointCloud]:
        """Read a file in chunks for large files.

        Default implementation reads all then chunks. Subclasses should
        override this for true streaming (e.g., LAS chunked reading).

        Args:
            path: File path to read.
            chunk_size: Points per chunk.
            **options: Format-specific options.

        Yields:
            PointCloud chunks.
        """
        pc = self.read(path, **options)
        yield from pc.iter_chunks(chunk_size)

    @classmethod
    @abstractmethod
    def extensions(cls) -> list[str]:
        """File extensions this reader handles (e.g., ['.las', '.laz'])."""

    @classmethod
    @abstractmethod
    def type_name(cls) -> str:
        """Pipeline type identifier (e.g., 'readers.las')."""


class Writer(ABC):
    """Base class for all point cloud writers."""

    @abstractmethod
    def write(self, pc: PointCloud, path: str, **options: Any) -> int:
        """Write a point cloud to file.

        Args:
            pc: PointCloud to write.
            path: Output file path.
            **options: Format-specific options.

        Returns:
            Number of points written.
        """

    @classmethod
    @abstractmethod
    def extensions(cls) -> list[str]:
        """File extensions this writer handles."""

    @classmethod
    @abstractmethod
    def type_name(cls) -> str:
        """Pipeline type identifier (e.g., 'writers.las')."""
