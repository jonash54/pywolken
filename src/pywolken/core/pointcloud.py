"""Core PointCloud class — columnar storage backed by NumPy arrays."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from pywolken.core.bounds import Bounds
from pywolken.core.dimensions import get_dtype
from pywolken.core.metadata import Metadata


class PointCloud:
    """Point cloud container using dict-of-arrays (columnar) storage.

    Each dimension (X, Y, Z, Intensity, Classification, ...) is stored as
    a separate NumPy array. This gives excellent cache locality for
    single-dimension operations (e.g., filtering by Classification) and
    makes it easy to add/remove dimensions without reallocating.

    Examples:
        >>> pc = PointCloud()
        >>> pc["X"] = np.array([1.0, 2.0, 3.0])
        >>> pc["Y"] = np.array([4.0, 5.0, 6.0])
        >>> pc["Z"] = np.array([7.0, 8.0, 9.0])
        >>> len(pc)
        3
        >>> pc.dimensions
        ['X', 'Y', 'Z']
    """

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}
        self._metadata: Metadata = Metadata()
        self._crs: str | None = None  # WKT or EPSG string

    # ── Properties ──────────────────────────────────────────────────

    @property
    def num_points(self) -> int:
        """Number of points in the cloud."""
        if not self._arrays:
            return 0
        return len(next(iter(self._arrays.values())))

    @property
    def dimensions(self) -> list[str]:
        """List of dimension names present in this cloud."""
        return list(self._arrays.keys())

    @property
    def bounds(self) -> Bounds:
        """Compute the 3D axis-aligned bounding box."""
        if "X" not in self or "Y" not in self or "Z" not in self:
            raise ValueError("PointCloud needs X, Y, Z dimensions for bounds")
        return Bounds.from_arrays(self["X"], self["Y"], self["Z"])

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, value: Metadata) -> None:
        self._metadata = value

    @property
    def crs(self) -> str | None:
        """Coordinate reference system (WKT or EPSG string)."""
        return self._crs

    @crs.setter
    def crs(self, value: str | None) -> None:
        self._crs = value

    # ── Array Access ────────────────────────────────────────────────

    def __getitem__(self, key: str) -> np.ndarray:
        """Get a dimension array by name: pc['X']."""
        if key not in self._arrays:
            raise KeyError(f"Dimension '{key}' not found. Available: {self.dimensions}")
        return self._arrays[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Set a dimension array: pc['X'] = array."""
        value = np.asarray(value)
        if self._arrays:
            expected = self.num_points
            if len(value) != expected:
                raise ValueError(
                    f"Array length {len(value)} doesn't match "
                    f"existing point count {expected}"
                )
        self._arrays[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if dimension exists: 'X' in pc."""
        return key in self._arrays

    def __len__(self) -> int:
        """Number of points."""
        return self.num_points

    def __repr__(self) -> str:
        dims = ", ".join(self.dimensions[:6])
        if len(self.dimensions) > 6:
            dims += f", ... (+{len(self.dimensions) - 6} more)"
        return f"PointCloud({self.num_points:,} points, dims=[{dims}])"

    # ── Slicing & Filtering ─────────────────────────────────────────

    def mask(self, mask: np.ndarray) -> PointCloud:
        """Return a new PointCloud with only points where mask is True.

        Args:
            mask: Boolean array of length num_points.

        Returns:
            New PointCloud containing only the selected points.
        """
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != self.num_points:
            raise ValueError(
                f"Mask length {len(mask)} doesn't match point count {self.num_points}"
            )
        result = PointCloud()
        for name, arr in self._arrays.items():
            result._arrays[name] = arr[mask]
        result._metadata = self._metadata.copy()
        result._crs = self._crs
        return result

    def slice(self, start: int, end: int) -> PointCloud:
        """Return a new PointCloud with points in [start, end) range."""
        result = PointCloud()
        for name, arr in self._arrays.items():
            result._arrays[name] = arr[start:end]
        result._metadata = self._metadata.copy()
        result._crs = self._crs
        return result

    def iter_chunks(self, chunk_size: int) -> Iterator[PointCloud]:
        """Iterate over the point cloud in chunks.

        Args:
            chunk_size: Number of points per chunk.

        Yields:
            PointCloud objects, each containing up to chunk_size points.
        """
        for start in range(0, self.num_points, chunk_size):
            yield self.slice(start, start + chunk_size)

    # ── Dimension Management ────────────────────────────────────────

    def add_dimension(
        self, name: str, dtype: np.dtype | None = None, fill: float = 0.0
    ) -> None:
        """Add a new dimension, filled with a constant value.

        Args:
            name: Dimension name.
            dtype: NumPy dtype (auto-detected from standard dims if None).
            fill: Fill value for all points.
        """
        if name in self._arrays:
            raise ValueError(f"Dimension '{name}' already exists")
        if dtype is None:
            dtype = get_dtype(name)
        self._arrays[name] = np.full(self.num_points, fill, dtype=dtype)

    def remove_dimension(self, name: str) -> None:
        """Remove a dimension."""
        if name not in self._arrays:
            raise KeyError(f"Dimension '{name}' not found")
        del self._arrays[name]

    # ── Conversion ──────────────────────────────────────────────────

    def to_numpy(self) -> np.ndarray:
        """Convert to a NumPy structured array."""
        dtype = [(name, arr.dtype) for name, arr in self._arrays.items()]
        result = np.empty(self.num_points, dtype=dtype)
        for name, arr in self._arrays.items():
            result[name] = arr
        return result

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a dict of dimension arrays (references, not copies)."""
        return dict(self._arrays)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, crs: str | None = None) -> PointCloud:
        """Create from a NumPy structured array.

        Args:
            arr: Structured array with named fields.
            crs: Optional coordinate reference system.
        """
        pc = cls()
        for name in arr.dtype.names:
            pc._arrays[name] = arr[name].copy()
        pc._crs = crs
        return pc

    @classmethod
    def from_dict(
        cls,
        data: dict[str, np.ndarray],
        crs: str | None = None,
    ) -> PointCloud:
        """Create from a dict of arrays.

        Args:
            data: Dict mapping dimension names to arrays (all same length).
            crs: Optional coordinate reference system.
        """
        pc = cls()
        lengths = {name: len(arr) for name, arr in data.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"All arrays must have same length, got: {lengths}")
        for name, arr in data.items():
            pc._arrays[name] = np.asarray(arr)
        pc._crs = crs
        return pc

    # ── Operations ──────────────────────────────────────────────────

    def copy(self) -> PointCloud:
        """Deep copy of this point cloud."""
        result = PointCloud()
        for name, arr in self._arrays.items():
            result._arrays[name] = arr.copy()
        result._metadata = self._metadata.copy()
        result._crs = self._crs
        return result

    def merge(self, other: PointCloud) -> PointCloud:
        """Merge another point cloud into a new combined cloud.

        Only dimensions present in both clouds are kept.
        """
        common_dims = [d for d in self.dimensions if d in other]
        if not common_dims:
            raise ValueError("No common dimensions to merge")

        result = PointCloud()
        for name in common_dims:
            result._arrays[name] = np.concatenate([self[name], other[name]])
        result._metadata = self._metadata.copy()
        result._crs = self._crs
        return result
