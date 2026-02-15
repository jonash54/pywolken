"""Crop filter — spatial bounding box crop.

Supports PDAL-compatible bounds syntax:
    "([xmin, xmax], [ymin, ymax])"                   — 2D crop
    "([xmin, xmax], [ymin, ymax], [zmin, zmax])"      — 3D crop
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry

# Parse bounds like "([xmin, xmax], [ymin, ymax])" or with 3 pairs
_BRACKET_PAIR = re.compile(r"\[\s*([^,\]]+)\s*,\s*([^,\]]+)\s*\]")


def _parse_bounds(bounds_str: str) -> list[tuple[float, float]]:
    """Parse a PDAL-style bounds string into a list of (min, max) pairs."""
    pairs = _BRACKET_PAIR.findall(bounds_str)
    if len(pairs) < 2:
        raise ValueError(
            f"Invalid bounds: '{bounds_str}'. "
            f"Expected: '([xmin, xmax], [ymin, ymax])' or "
            f"'([xmin, xmax], [ymin, ymax], [zmin, zmax])'"
        )
    return [(float(lo), float(hi)) for lo, hi in pairs]


class CropFilter(Filter):
    """Crop points to a spatial bounding box.

    Options:
        bounds: str — PDAL-style bounds string.
            "([xmin, xmax], [ymin, ymax])"               — 2D crop
            "([xmin, xmax], [ymin, ymax], [zmin, zmax])"  — 3D crop

    Alternative options (if bounds not provided):
        minx, maxx, miny, maxy: float — 2D crop boundaries.
        minz, maxz: float — Optional Z boundaries for 3D crop.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        if "bounds" not in self.options and "minx" not in self.options:
            raise ValueError(
                "CropFilter requires 'bounds' string or explicit minx/maxx/miny/maxy"
            )

    def filter(self, pc: PointCloud) -> PointCloud:
        if "bounds" in self.options:
            pairs = _parse_bounds(self.options["bounds"])
            xmin, xmax = pairs[0]
            ymin, ymax = pairs[1]
            zmin = pairs[2][0] if len(pairs) > 2 else None
            zmax = pairs[2][1] if len(pairs) > 2 else None
        else:
            xmin = float(self.options["minx"])
            xmax = float(self.options["maxx"])
            ymin = float(self.options["miny"])
            ymax = float(self.options["maxy"])
            zmin = float(self.options["minz"]) if "minz" in self.options else None
            zmax = float(self.options["maxz"]) if "maxz" in self.options else None

        mask = (
            (pc["X"] >= xmin)
            & (pc["X"] <= xmax)
            & (pc["Y"] >= ymin)
            & (pc["Y"] <= ymax)
        )

        if zmin is not None:
            mask &= pc["Z"] >= zmin
        if zmax is not None:
            mask &= pc["Z"] <= zmax

        return pc.mask(mask)

    @classmethod
    def type_name(cls) -> str:
        return "filters.crop"


filter_registry.register(CropFilter)
