"""Tests for assign and expression filters."""

import numpy as np
import pytest

from pywolken.core.pointcloud import PointCloud
from pywolken.filters.assign import AssignFilter
from pywolken.filters.expression import ExpressionFilter


class TestAssignFilter:
    def test_set_constant(self, sample_pc):
        f = AssignFilter(assignment="Classification=2")
        result = f.filter(sample_pc)
        assert all(result["Classification"] == 2)

    def test_create_new_dimension(self, sample_pc):
        f = AssignFilter(assignment="NewDim=42")
        result = f.filter(sample_pc)
        assert "NewDim" in result
        assert all(result["NewDim"] == 42.0)

    def test_multiple_assignments(self, sample_pc):
        f = AssignFilter(assignment="Classification=2,Intensity=0")
        result = f.filter(sample_pc)
        assert all(result["Classification"] == 2)
        assert all(result["Intensity"] == 0)

    def test_dict_value(self, sample_pc):
        f = AssignFilter(value={"Classification": 6, "Intensity": 100})
        result = f.filter(sample_pc)
        assert all(result["Classification"] == 6)
        assert all(result["Intensity"] == 100)

    def test_does_not_modify_original(self, sample_pc):
        original_cls = sample_pc["Classification"].copy()
        f = AssignFilter(assignment="Classification=2")
        f.filter(sample_pc)
        np.testing.assert_array_equal(sample_pc["Classification"], original_cls)

    def test_missing_option_raises(self):
        with pytest.raises(ValueError, match="assignment.*value"):
            AssignFilter()

    def test_type_name(self):
        assert AssignFilter.type_name() == "filters.assign"


class TestExpressionFilter:
    def test_equality(self, ground_pc):
        f = ExpressionFilter(expression="Classification == 2")
        result = f.filter(ground_pc)
        assert result.num_points == 20
        assert all(result["Classification"] == 2)

    def test_greater_than(self, sample_pc):
        f = ExpressionFilter(expression="Z > 300")
        result = f.filter(sample_pc)
        assert all(result["Z"] > 300)

    def test_less_than(self, sample_pc):
        f = ExpressionFilter(expression="Z < 200")
        result = f.filter(sample_pc)
        assert all(result["Z"] < 200)

    def test_and(self, sample_pc):
        f = ExpressionFilter(expression="Z > 200 AND Z < 400")
        result = f.filter(sample_pc)
        assert all(result["Z"] > 200)
        assert all(result["Z"] < 400)

    def test_or(self, ground_pc):
        f = ExpressionFilter(expression="Classification == 2 OR Classification == 3")
        result = f.filter(ground_pc)
        assert all((result["Classification"] == 2) | (result["Classification"] == 3))
        assert result.num_points == 50  # All points are class 2 or 3

    def test_not_equal(self, ground_pc):
        f = ExpressionFilter(expression="Classification != 2")
        result = f.filter(ground_pc)
        assert result.num_points == 30
        assert all(result["Classification"] != 2)

    def test_where_alias(self, ground_pc):
        f = ExpressionFilter(where="Classification == 2")
        result = f.filter(ground_pc)
        assert result.num_points == 20

    def test_missing_option_raises(self):
        with pytest.raises(ValueError, match="expression.*where"):
            ExpressionFilter()

    def test_invalid_expression_raises(self, sample_pc):
        f = ExpressionFilter(expression="not valid at all")
        with pytest.raises(ValueError, match="Invalid expression"):
            f.filter(sample_pc)

    def test_type_name(self):
        assert ExpressionFilter.type_name() == "filters.expression"
