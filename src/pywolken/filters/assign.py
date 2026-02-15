"""Assign filter — set or create dimension values.

Supports setting dimension values to constants or copying between dimensions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pywolken.core.dimensions import get_dtype
from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry


class AssignFilter(Filter):
    """Assign values to point cloud dimensions.

    Options:
        assignment: str — Assignment expression(s), comma-separated.
            "Classification=2"              — Set all points to class 2
            "Intensity=0"                   — Zero out intensity
            "NewDim=42.0"                   — Create a new dimension

        value: dict — Direct dict assignment (API usage).
            {"Classification": 2, "Intensity": 0}
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        if "assignment" not in self.options and "value" not in self.options:
            raise ValueError(
                "AssignFilter requires 'assignment' string or 'value' dict"
            )

    def filter(self, pc: PointCloud) -> PointCloud:
        result = pc.copy()

        assignments = self._parse_assignments()
        for dim_name, value in assignments.items():
            if dim_name not in result:
                dtype = get_dtype(dim_name)
                result.add_dimension(dim_name, dtype)
            result[dim_name][:] = value

        return result

    def _parse_assignments(self) -> dict[str, float]:
        """Parse assignment expressions into a dict."""
        if "value" in self.options:
            return {str(k): float(v) for k, v in self.options["value"].items()}

        assignments = {}
        for expr in self.options["assignment"].split(","):
            expr = expr.strip()
            if "=" not in expr:
                raise ValueError(f"Invalid assignment: '{expr}'. Expected 'DimName=value'")
            dim, val_str = expr.split("=", 1)
            assignments[dim.strip()] = float(val_str.strip())
        return assignments

    @classmethod
    def type_name(cls) -> str:
        return "filters.assign"


filter_registry.register(AssignFilter)
