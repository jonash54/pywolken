"""Base class for all processing filters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pywolken.core.pointcloud import PointCloud


class Filter(ABC):
    """Base class for all point cloud processing filters.

    Subclasses implement the `filter()` method which takes a PointCloud
    and returns a new (modified) PointCloud.
    """

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    def filter(self, pc: PointCloud) -> PointCloud:
        """Apply the filter to a point cloud.

        Args:
            pc: Input point cloud.

        Returns:
            Filtered/modified point cloud (may be new object or same).
        """

    @classmethod
    @abstractmethod
    def type_name(cls) -> str:
        """Pipeline type identifier (e.g., 'filters.range')."""

    def __repr__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in self.options.items())
        return f"{self.type_name()}({opts})"
