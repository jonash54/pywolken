"""Standard point cloud dimension definitions."""

from __future__ import annotations

import numpy as np

# Standard LAS dimensions with their default NumPy dtypes
STANDARD_DIMENSIONS: dict[str, np.dtype] = {
    "X": np.dtype(np.float64),
    "Y": np.dtype(np.float64),
    "Z": np.dtype(np.float64),
    "Intensity": np.dtype(np.uint16),
    "ReturnNumber": np.dtype(np.uint8),
    "NumberOfReturns": np.dtype(np.uint8),
    "Classification": np.dtype(np.uint8),
    "ScanAngleRank": np.dtype(np.float32),
    "UserData": np.dtype(np.uint8),
    "PointSourceId": np.dtype(np.uint16),
    "GpsTime": np.dtype(np.float64),
    "Red": np.dtype(np.uint16),
    "Green": np.dtype(np.uint16),
    "Blue": np.dtype(np.uint16),
    "NIR": np.dtype(np.uint16),
}

# LAS classification codes (ASPRS standard)
CLASSIFICATION_CODES: dict[int, str] = {
    0: "Created, Never Classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation",
    4: "Medium Vegetation",
    5: "High Vegetation",
    6: "Building",
    7: "Low Point (Noise)",
    8: "Reserved / Model Key-point",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved / Overlap",
    13: "Wire - Guard (Shield)",
    14: "Wire - Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-structure Connector",
    17: "Bridge Deck",
    18: "High Noise",
}


def get_dtype(name: str) -> np.dtype:
    """Get the default dtype for a dimension name, defaulting to float64."""
    return STANDARD_DIMENSIONS.get(name, np.dtype(np.float64))
