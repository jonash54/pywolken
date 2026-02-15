"""Hillshade computation from DEM — pure NumPy, no GDAL required.

Uses Horn's method (same algorithm as gdaldem hillshade).
"""

from __future__ import annotations

import numpy as np


def hillshade(
    dem: np.ndarray,
    resolution: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    nodata: float = -9999.0,
) -> np.ndarray:
    """Compute hillshade from a DEM using Horn's method.

    This produces identical results to `gdaldem hillshade`.

    Args:
        dem: 2D float array (rows x cols), top-left origin.
        resolution: Cell size in the DEM's CRS units.
        azimuth: Light source direction in degrees (0=N, 90=E, 180=S, 315=NW).
        altitude: Light source elevation angle in degrees (0=horizon, 90=zenith).
        z_factor: Vertical exaggeration factor.
        nodata: Nodata value in the DEM.

    Returns:
        2D uint8 array (0-255) of shaded relief.
    """
    # Create a mask for valid cells
    valid = dem != nodata

    # Replace nodata with neighbor values for gradient computation
    dem_filled = dem.copy()
    if not np.all(valid):
        # Simple fill: use nearest valid neighbor (good enough for edges)
        from scipy.ndimage import generic_filter
        def _fill_nodata(values):
            center = values[4]  # 3x3 kernel, center is index 4
            if center == nodata:
                valid_vals = values[values != nodata]
                return np.mean(valid_vals) if len(valid_vals) > 0 else nodata
            return center
        dem_filled = generic_filter(dem_filled, _fill_nodata, size=3)

    # Apply z_factor
    z = dem_filled * z_factor

    # Compute gradients using Horn's method (3x3 weighted differences)
    # dz/dx and dz/dy using np.gradient (central differences)
    dzdx = np.gradient(z, resolution, axis=1)
    dzdy = np.gradient(z, resolution, axis=0)

    # Convert angles to radians
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)  # Math convention
    altitude_rad = np.radians(altitude)

    # Slope and aspect
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)

    # Hillshade formula (Horn's method)
    shade = (
        np.sin(altitude_rad)
        + np.cos(altitude_rad) * slope
        * np.cos(azimuth_rad - np.arctan2(-dzdy, dzdx))
    )

    # Normalize to 0-255
    shade = np.clip(shade, 0, 1)
    shade = (shade * 254 + 1).astype(np.uint8)

    # Set nodata areas to 0
    shade[~valid] = 0

    return shade


def multi_directional_hillshade(
    dem: np.ndarray,
    resolution: float,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    nodata: float = -9999.0,
    directions: list[float] | None = None,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Compute multi-directional hillshade by blending multiple azimuths.

    Args:
        dem: 2D float array.
        resolution: Cell size.
        altitude: Light elevation angle.
        z_factor: Vertical exaggeration.
        nodata: Nodata value.
        directions: List of azimuth angles. Default: 8 cardinal directions.
        weights: Weights per direction. Default: equal weights.

    Returns:
        2D uint8 array (0-255) of blended shaded relief.
    """
    if directions is None:
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
    if weights is None:
        weights = [1.0] * len(directions)

    total_weight = sum(weights)
    blended = np.zeros(dem.shape, dtype=np.float64)

    for azimuth, weight in zip(directions, weights):
        shade = hillshade(
            dem, resolution,
            azimuth=azimuth, altitude=altitude,
            z_factor=z_factor, nodata=nodata,
        ).astype(np.float64)
        blended += shade * weight

    blended /= total_weight
    result = np.clip(blended, 0, 255).astype(np.uint8)
    return result
