"""Tests for the decimation filter."""

import numpy as np
import pytest

from pywolken.filters.decimation import DecimationFilter


class TestDecimationFilter:
    def test_step(self, sample_pc):
        f = DecimationFilter(step=10)
        result = f.filter(sample_pc)
        assert result.num_points == 10  # 100 / 10

    def test_step_1_keeps_all(self, sample_pc):
        f = DecimationFilter(step=1)
        result = f.filter(sample_pc)
        assert result.num_points == 100

    def test_fraction(self, sample_pc):
        f = DecimationFilter(fraction=0.5, seed=42)
        result = f.filter(sample_pc)
        assert result.num_points == 50

    def test_fraction_small(self, sample_pc):
        f = DecimationFilter(fraction=0.01, seed=42)
        result = f.filter(sample_pc)
        assert result.num_points == 1  # max(1, 100 * 0.01)

    def test_count(self, sample_pc):
        f = DecimationFilter(count=25, seed=42)
        result = f.filter(sample_pc)
        assert result.num_points == 25

    def test_count_exceeds_total(self, sample_pc):
        f = DecimationFilter(count=500, seed=42)
        result = f.filter(sample_pc)
        assert result.num_points == 100  # Clamped to total

    def test_reproducible_with_seed(self, sample_pc):
        f1 = DecimationFilter(fraction=0.5, seed=42)
        f2 = DecimationFilter(fraction=0.5, seed=42)
        r1 = f1.filter(sample_pc)
        r2 = f2.filter(sample_pc)
        np.testing.assert_array_equal(r1["X"], r2["X"])

    def test_preserves_dimensions(self, sample_pc):
        f = DecimationFilter(step=5)
        result = f.filter(sample_pc)
        assert set(result.dimensions) == set(sample_pc.dimensions)

    def test_missing_option_raises(self):
        with pytest.raises(ValueError, match="step.*fraction.*count"):
            DecimationFilter()

    def test_invalid_step_raises(self, sample_pc):
        f = DecimationFilter(step=0)
        with pytest.raises(ValueError, match="step"):
            f.filter(sample_pc)

    def test_type_name(self):
        assert DecimationFilter.type_name() == "filters.decimation"
