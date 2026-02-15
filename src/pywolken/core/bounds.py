"""Axis-aligned 3D bounding box."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """3D axis-aligned bounding box.

    Attributes:
        minx, miny, minz: Minimum corner coordinates.
        maxx, maxy, maxz: Maximum corner coordinates.
    """

    minx: float
    miny: float
    minz: float
    maxx: float
    maxy: float
    maxz: float

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Bounds:
        """Compute bounds from X, Y, Z arrays."""
        return cls(
            minx=float(np.min(x)),
            miny=float(np.min(y)),
            minz=float(np.min(z)),
            maxx=float(np.max(x)),
            maxy=float(np.max(y)),
            maxz=float(np.max(z)),
        )

    @classmethod
    def from_pointcloud(cls, pc: "PointCloud") -> Bounds:
        """Compute bounds from a PointCloud."""
        return cls.from_arrays(pc["X"], pc["Y"], pc["Z"])

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point is inside the bounding box."""
        return (
            self.minx <= x <= self.maxx
            and self.miny <= y <= self.maxy
            and self.minz <= z <= self.maxz
        )

    def contains_2d(self, x: float, y: float) -> bool:
        """Check if a 2D point is inside the XY bounding box."""
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy

    def intersects(self, other: Bounds) -> bool:
        """Check if two bounding boxes overlap."""
        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
            and self.minz <= other.maxz
            and self.maxz >= other.minz
        )

    def union(self, other: Bounds) -> Bounds:
        """Return the bounding box enclosing both boxes."""
        return Bounds(
            minx=min(self.minx, other.minx),
            miny=min(self.miny, other.miny),
            minz=min(self.minz, other.minz),
            maxx=max(self.maxx, other.maxx),
            maxy=max(self.maxy, other.maxy),
            maxz=max(self.maxz, other.maxz),
        )

    @property
    def width(self) -> float:
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        return self.maxy - self.miny

    @property
    def depth(self) -> float:
        return self.maxz - self.minz

    def __repr__(self) -> str:
        return (
            f"Bounds(x=[{self.minx:.2f}, {self.maxx:.2f}], "
            f"y=[{self.miny:.2f}, {self.maxy:.2f}], "
            f"z=[{self.minz:.2f}, {self.maxz:.2f}])"
        )
