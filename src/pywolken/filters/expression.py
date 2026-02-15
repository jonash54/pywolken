"""Expression filter — filter points by boolean expressions.

Supports simple comparison expressions on dimension values:
    "Classification == 2"
    "Z > 100"
    "Intensity >= 500 AND Z < 200"
    "Classification == 2 OR Classification == 6"
"""

from __future__ import annotations

import operator
import re
from typing import Any

import numpy as np

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.base import Filter
from pywolken.filters.registry import filter_registry

_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}

# Pattern for a single comparison: DimName op value
_COMP_PATTERN = re.compile(
    r"(?P<dim>[A-Za-z_]\w*)\s*"
    r"(?P<op>==|!=|>=|<=|>|<)\s*"
    r"(?P<val>-?[\d.]+(?:e[+-]?\d+)?)"
)


def _eval_comparison(pc: PointCloud, dim: str, op_str: str, value: float) -> np.ndarray:
    """Evaluate a single comparison, returning a boolean mask."""
    if dim not in pc:
        raise KeyError(f"Dimension '{dim}' not found. Available: {pc.dimensions}")
    op_func = _OPERATORS[op_str]
    return op_func(pc[dim], value)


def _eval_expression(pc: PointCloud, expr: str) -> np.ndarray:
    """Evaluate a boolean expression with AND/OR logic."""
    # Split on OR first (lower precedence)
    or_parts = re.split(r"\bOR\b", expr, flags=re.IGNORECASE)

    or_mask = np.zeros(pc.num_points, dtype=bool)

    for or_part in or_parts:
        # Split on AND (higher precedence)
        and_parts = re.split(r"\bAND\b", or_part, flags=re.IGNORECASE)

        and_mask = np.ones(pc.num_points, dtype=bool)
        for and_part in and_parts:
            and_part = and_part.strip()
            m = _COMP_PATTERN.match(and_part)
            if not m:
                raise ValueError(
                    f"Invalid expression part: '{and_part}'. "
                    f"Expected: 'DimName op value' (e.g., 'Z > 100')"
                )
            dim = m.group("dim")
            op_str = m.group("op")
            value = float(m.group("val"))
            and_mask &= _eval_comparison(pc, dim, op_str, value)

        or_mask |= and_mask

    return or_mask


class ExpressionFilter(Filter):
    """Filter points by boolean expression.

    Options:
        expression: str — Boolean expression.
            "Classification == 2"
            "Z > 100 AND Z < 500"
            "Classification == 2 OR Classification == 6"
            "Intensity >= 500"

        where: str — Alias for 'expression' (PDAL compatibility).
    """

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        if "expression" not in self.options and "where" not in self.options:
            raise ValueError(
                "ExpressionFilter requires 'expression' or 'where' option"
            )

    def filter(self, pc: PointCloud) -> PointCloud:
        expr = self.options.get("expression") or self.options.get("where", "")
        mask = _eval_expression(pc, expr)
        return pc.mask(mask)

    @classmethod
    def type_name(cls) -> str:
        return "filters.expression"


filter_registry.register(ExpressionFilter)
