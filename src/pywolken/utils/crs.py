"""CRS (Coordinate Reference System) utilities wrapping pyproj."""

from __future__ import annotations

from pyproj import CRS, Transformer
import numpy as np


def parse_crs(crs_input: str | CRS | None) -> CRS | None:
    """Parse a CRS from various input formats.

    Args:
        crs_input: EPSG string ("EPSG:25832"), WKT string, proj4 string,
                   or a pyproj.CRS object.

    Returns:
        pyproj.CRS object or None.
    """
    if crs_input is None:
        return None
    if isinstance(crs_input, CRS):
        return crs_input
    return CRS.from_user_input(crs_input)


def reproject_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    src_crs: str | CRS,
    dst_crs: str | CRS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject X, Y, Z arrays from one CRS to another.

    Args:
        x, y, z: Coordinate arrays.
        src_crs: Source CRS.
        dst_crs: Target CRS.

    Returns:
        Tuple of (new_x, new_y, new_z) arrays.
    """
    src = parse_crs(src_crs)
    dst = parse_crs(dst_crs)
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    new_x, new_y, new_z = transformer.transform(x, y, z)
    return (
        np.asarray(new_x, dtype=np.float64),
        np.asarray(new_y, dtype=np.float64),
        np.asarray(new_z, dtype=np.float64),
    )
