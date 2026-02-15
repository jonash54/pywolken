"""Tests for the range filter."""

import numpy as np
import pytest

from pywolken.filters.range import RangeFilter, _parse_range_expr


class TestParseRangeExpr:
    def test_exact_match(self):
        dim, neg, lo, hi = _parse_range_expr("Classification[2:2]")
        assert dim == "Classification"
        assert neg is False
        assert lo == 2.0
        assert hi == 2.0

    def test_range(self):
        dim, neg, lo, hi = _parse_range_expr("Z[100:500]")
        assert dim == "Z"
        assert lo == 100.0
        assert hi == 500.0

    def test_min_only(self):
        dim, neg, lo, hi = _parse_range_expr("Z[100:]")
        assert lo == 100.0
        assert hi is None

    def test_max_only(self):
        dim, neg, lo, hi = _parse_range_expr("Z[:500]")
        assert lo is None
        assert hi == 500.0

    def test_negation(self):
        dim, neg, lo, hi = _parse_range_expr("Classification![7:7]")
        assert neg is True
        assert lo == 7.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid range"):
            _parse_range_expr("badformat")


class TestRangeFilter:
    def test_exact_classification(self, ground_pc):
        f = RangeFilter(limits="Classification[2:2]")
        result = f.filter(ground_pc)
        assert result.num_points == 20
        assert all(result["Classification"] == 2)

    def test_range_z(self, sample_pc):
        z_min, z_max = 200.0, 400.0
        f = RangeFilter(limits=f"Z[{z_min}:{z_max}]")
        result = f.filter(sample_pc)
        assert all(result["Z"] >= z_min)
        assert all(result["Z"] <= z_max)

    def test_min_only(self, sample_pc):
        f = RangeFilter(limits="Z[300:]")
        result = f.filter(sample_pc)
        assert all(result["Z"] >= 300.0)

    def test_max_only(self, sample_pc):
        f = RangeFilter(limits="Z[:200]")
        result = f.filter(sample_pc)
        assert all(result["Z"] <= 200.0)

    def test_negation(self, ground_pc):
        f = RangeFilter(limits="Classification![2:2]")
        result = f.filter(ground_pc)
        assert result.num_points == 30
        assert all(result["Classification"] != 2)

    def test_multiple_conditions(self, sample_pc):
        f = RangeFilter(limits="Classification[2:2],Z[200:400]")
        result = f.filter(sample_pc)
        assert all(result["Classification"] == 2)
        assert all(result["Z"] >= 200.0)
        assert all(result["Z"] <= 400.0)

    def test_missing_limits_raises(self):
        with pytest.raises(ValueError, match="limits"):
            RangeFilter()

    def test_missing_dimension_raises(self, sample_pc):
        f = RangeFilter(limits="NonExistent[0:1]")
        with pytest.raises(KeyError, match="NonExistent"):
            f.filter(sample_pc)

    def test_type_name(self):
        assert RangeFilter.type_name() == "filters.range"
