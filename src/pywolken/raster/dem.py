"""DEM (Digital Elevation Model) generation from point clouds.

Supports multiple interpolation methods:
    - idw: Inverse Distance Weighting (default, matches PDAL's writers.gdal)
    - mean: Simple grid cell averaging
    - nearest: Nearest neighbor interpolation
    - tin: TIN-based linear interpolation via Delaunay triangulation
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import LinearNDInterpolator

from pywolken.core.pointcloud import PointCloud


def create_dem(
    pc: PointCloud,
    resolution: float,
    method: str = "idw",
    bounds: tuple[float, float, float, float] | None = None,
    nodata: float = -9999.0,
    power: float = 2.0,
    radius: float | None = None,
    window_size: int = 6,
) -> tuple[np.ndarray, dict]:
    """Generate a DEM raster from point cloud data.

    Args:
        pc: PointCloud (typically ground-classified points).
        resolution: Grid cell size in the point cloud's CRS units (meters).
        method: Interpolation method — "idw", "mean", "nearest", or "tin".
        bounds: (xmin, ymin, xmax, ymax). If None, computed from data.
        nodata: Value for cells with no data.
        power: IDW distance power parameter (default 2.0).
        radius: Search radius for IDW/nearest. If None, auto-computed.
        window_size: Window size for IDW void filling (in cells).

    Returns:
        (raster, transform) where:
            raster: 2D float32 array (rows x cols), top-left origin.
            transform: dict with keys 'xmin', 'ymax', 'resolution', 'crs',
                       'nrows', 'ncols' for GeoTIFF export.
    """
    x, y, z = pc["X"], pc["Y"], pc["Z"]

    if bounds is None:
        xmin, ymin = float(np.min(x)), float(np.min(y))
        xmax, ymax = float(np.max(x)), float(np.max(y))
    else:
        xmin, ymin, xmax, ymax = bounds

    ncols = int(np.ceil((xmax - xmin) / resolution))
    nrows = int(np.ceil((ymax - ymin) / resolution))

    if ncols <= 0 or nrows <= 0:
        raise ValueError(f"Invalid grid dimensions: {nrows}x{ncols}")

    if method == "idw":
        raster = _interpolate_idw(
            x, y, z, xmin, ymin, resolution, nrows, ncols,
            nodata=nodata, power=power, radius=radius,
            window_size=window_size,
        )
    elif method == "mean":
        raster = _interpolate_mean(
            x, y, z, xmin, ymin, resolution, nrows, ncols, nodata=nodata
        )
    elif method == "nearest":
        raster = _interpolate_nearest(
            x, y, z, xmin, ymin, resolution, nrows, ncols, nodata=nodata
        )
    elif method == "tin":
        raster = _interpolate_tin(
            x, y, z, xmin, ymin, resolution, nrows, ncols, nodata=nodata
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use: idw, mean, nearest, tin")

    transform = {
        "xmin": xmin,
        "ymax": ymin + nrows * resolution,  # Top-left Y
        "resolution": resolution,
        "nrows": nrows,
        "ncols": ncols,
        "crs": pc.crs,
    }

    return raster, transform


def _interpolate_idw(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    xmin: float, ymin: float, resolution: float,
    nrows: int, ncols: int,
    nodata: float, power: float, radius: float | None,
    window_size: int,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation (vectorized k-NN)."""
    tree = cKDTree(np.column_stack([x, y]))

    if radius is None:
        radius = resolution * window_size

    # Use fixed-k query for fully vectorized computation
    k = min(16, len(x))

    # Generate grid cell center coordinates
    col_coords = xmin + (np.arange(ncols) + 0.5) * resolution
    row_coords = ymin + (np.arange(nrows) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(col_coords, row_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    raster = np.full(nrows * ncols, nodata, dtype=np.float32)

    # Process in batches to limit memory
    batch_size = 100_000
    for start in range(0, len(grid_points), batch_size):
        end = min(start + batch_size, len(grid_points))
        batch = grid_points[start:end]

        # Fixed-k query returns (batch_size, k) arrays — fully vectorizable
        dist, idx = tree.query(batch, k=k)

        # Ensure 2D even if k=1
        if k == 1:
            dist = dist[:, np.newaxis]
            idx = idx[:, np.newaxis]

        # Mask neighbors beyond radius
        valid = dist <= radius

        # Handle zero distances (point exactly at grid center)
        zero_mask = dist == 0
        has_zero = np.any(zero_mask, axis=1)

        # Compute IDW weights (zero/invalid → weight 0)
        safe_dist = np.where((dist > 0) & valid, dist, 1.0)
        weights = np.where((dist > 0) & valid, 1.0 / np.power(safe_dist, power), 0.0)

        # Get Z values for all neighbors
        z_neighbors = z[idx]  # (batch, k)

        # Weighted average
        weighted_z = (weights * z_neighbors).sum(axis=1)
        weight_sum = weights.sum(axis=1)

        has_neighbors = weight_sum > 0
        result = np.where(has_neighbors, weighted_z / weight_sum, nodata)

        # Override with exact match for zero-distance points
        if np.any(has_zero):
            zero_z = np.where(zero_mask, z_neighbors, 0.0)
            zero_count = zero_mask.sum(axis=1)
            zero_mean = np.where(zero_count > 0, zero_z.sum(axis=1) / zero_count, 0.0)
            result = np.where(has_zero, zero_mean, result)

        raster[start:end] = result.astype(np.float32)

    # Reshape to 2D, flip so row 0 is the northern (top) row
    raster = raster.reshape(nrows, ncols)
    raster = raster[::-1]
    return raster


def _interpolate_mean(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    xmin: float, ymin: float, resolution: float,
    nrows: int, ncols: int, nodata: float,
) -> np.ndarray:
    """Simple mean gridding — average Z of all points in each cell."""
    col_idx = np.floor((x - xmin) / resolution).astype(np.int64)
    row_idx = np.floor((y - ymin) / resolution).astype(np.int64)

    # Clamp to valid range
    np.clip(col_idx, 0, ncols - 1, out=col_idx)
    np.clip(row_idx, 0, nrows - 1, out=row_idx)

    # Accumulate Z values per cell
    flat_idx = row_idx * ncols + col_idx
    z_sum = np.zeros(nrows * ncols, dtype=np.float64)
    z_count = np.zeros(nrows * ncols, dtype=np.int64)
    np.add.at(z_sum, flat_idx, z)
    np.add.at(z_count, flat_idx, 1)

    raster = np.full(nrows * ncols, nodata, dtype=np.float32)
    valid = z_count > 0
    raster[valid] = (z_sum[valid] / z_count[valid]).astype(np.float32)

    raster = raster.reshape(nrows, ncols)
    raster = raster[::-1]
    return raster


def _interpolate_nearest(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    xmin: float, ymin: float, resolution: float,
    nrows: int, ncols: int, nodata: float,
) -> np.ndarray:
    """Nearest neighbor interpolation using KDTree."""
    tree = cKDTree(np.column_stack([x, y]))

    col_coords = xmin + (np.arange(ncols) + 0.5) * resolution
    row_coords = ymin + (np.arange(nrows) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(col_coords, row_coords)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    dist, idx = tree.query(grid_points, k=1)
    raster = z[idx].astype(np.float32)

    # Mark cells too far from any point as nodata
    max_dist = resolution * 2
    raster[dist > max_dist] = nodata

    raster = raster.reshape(nrows, ncols)
    raster = raster[::-1]
    return raster


def _interpolate_tin(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    xmin: float, ymin: float, resolution: float,
    nrows: int, ncols: int, nodata: float,
) -> np.ndarray:
    """TIN-based linear interpolation via Delaunay triangulation."""
    # Build Delaunay triangulation
    points_2d = np.column_stack([x, y])
    tri = Delaunay(points_2d)
    interp = LinearNDInterpolator(tri, z, fill_value=nodata)

    col_coords = xmin + (np.arange(ncols) + 0.5) * resolution
    row_coords = ymin + (np.arange(nrows) + 0.5) * resolution
    grid_x, grid_y = np.meshgrid(col_coords, row_coords)

    raster = interp(grid_x, grid_y).astype(np.float32)
    raster = raster[::-1]
    return raster
