"""Range filter — filter points by dimension value ranges.

Supports PDAL-compatible range syntax:
    "Classification[2:2]"           — exact match
    "Classification[2:5]"           — inclusive range
    "Z[100:]"                       — minimum only
    "Z[:500]"                       — maximum only
    "Classification[2:2],Z[100:]"   — multiple (AND logic)
    "Classification![7:7]"          — negation (exclude)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry

# Pattern: DimName[min:max] or DimName![min:max]
_RANGE_PATTERN = re.compile(
    r"^(?P<dim>[A-Za-z_]\w*)"
    r"(?P<negate>!)?"
    r"\[(?P<min>[^:\]]*):(?P<max>[^:\]]*)\]$"
)


def _parse_range_expr(expr: str) -> tuple[str, bool, float | None, float | None]:
    """Parse a single range expression like 'Classification[2:2]'.

    Returns:
        (dimension_name, negate, min_value, max_value)
    """
    expr = expr.strip()
    m = _RANGE_PATTERN.match(expr)
    if not m:
        raise ValueError(
            f"Invalid range expression: '{expr}'. "
            f"Expected format: DimName[min:max] (e.g., 'Classification[2:2]')"
        )

    dim = m.group("dim")
    negate = m.group("negate") == "!"
    min_str = m.group("min").strip()
    max_str = m.group("max").strip()

    min_val = float(min_str) if min_str else None
    max_val = float(max_str) if max_str else None

    return dim, negate, min_val, max_val


def _apply_single_range(
    pc: PointCloud, dim: str, negate: bool, min_val: float | None, max_val: float | None
) -> np.ndarray:
    """Apply a single range expression, returning a boolean mask."""
    if dim not in pc:
        raise KeyError(
            f"Dimension '{dim}' not found in PointCloud. "
            f"Available: {pc.dimensions}"
        )

    arr = pc[dim]
    mask = np.ones(pc.num_points, dtype=bool)

    if min_val is not None:
        mask &= arr >= min_val
    if max_val is not None:
        mask &= arr <= max_val

    if negate:
        mask = ~mask

    return mask


class RangeFilter(Filter):
    """Filter points by dimension value ranges.

    PDAL-compatible syntax for the 'limits' option:
        "Classification[2:2]"           — keep only class 2
        "Classification[2:5]"           — keep classes 2 through 5
        "Z[100:]"                       — keep Z >= 100
        "Z[:500]"                       — keep Z <= 500
        "Classification[2:2],Z[100:]"   — multiple conditions (AND)
        "Classification![7:7]"          — exclude class 7 (negation)

    Options:
        limits: str — comma-separated range expressions.
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        if "limits" not in self.options:
            raise ValueError("RangeFilter requires 'limits' option")

    def filter(self, pc: PointCloud) -> PointCloud:
        limits_str = self.options["limits"]
        expressions = [e.strip() for e in limits_str.split(",") if e.strip()]

        mask = np.ones(pc.num_points, dtype=bool)
        for expr in expressions:
            dim, negate, min_val, max_val = _parse_range_expr(expr)
            mask &= _apply_single_range(pc, dim, negate, min_val, max_val)

        return pc.mask(mask)

    @classmethod
    def type_name(cls) -> str:
        return "filters.range"


filter_registry.register(RangeFilter)
