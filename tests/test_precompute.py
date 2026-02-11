"""Tests for fractal.precompute: slice computation and pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fractal.data_cube import DataCubeIndex, ParamGridSpec
from fractal.precompute import (
    SliceSpec,
    _bake_brightness,
    build_slice_specs,
    compute_slice,
    precompute_cube,
)


# ---------------------------------------------------------------------------
# _bake_brightness
# ---------------------------------------------------------------------------


class TestBakeBrightness:
    """Tests for convergence-time → brightness conversion."""

    def test_uniform_times_full_brightness(self):
        times = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        result = _bake_brightness(times)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, 255)

    def test_varying_times_range(self):
        times = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        result = _bake_brightness(times)
        assert result.dtype == np.uint8
        # Fastest (0.0) should be brightest
        assert result[0] > result[2]
        # Brightness should span from ~76 (0.3*255) to 255
        assert result[0] == 255
        assert result[2] >= 70  # ~0.3 * 255 = 76

    def test_output_shape(self):
        times = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = _bake_brightness(times)
        assert result.shape == (4,)


# ---------------------------------------------------------------------------
# build_slice_specs
# ---------------------------------------------------------------------------


class TestBuildSliceSpecs:
    """Tests for slice specification generation."""

    def test_count_matches_grid(self):
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0,),
            l1_values=(1.0,),
            l2_values=(0.5,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )
        specs = build_slice_specs(grid)
        assert len(specs) == grid.n_slices

    def test_flat_indices_unique(self):
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0, 2.0),
            l1_values=(1.0,),
            l2_values=(0.5,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )
        specs = build_slice_specs(grid)
        indices = [s.flat_index for s in specs]
        assert len(set(indices)) == len(indices)

    def test_flat_indices_cover_range(self):
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0,),
            l1_values=(1.0,),
            l2_values=(0.5,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )
        specs = build_slice_specs(grid)
        indices = sorted(s.flat_index for s in specs)
        assert indices == list(range(grid.n_slices))

    def test_spec_carries_params(self):
        grid = ParamGridSpec(
            m1_values=(1.0,),
            m2_values=(2.0,),
            l1_values=(0.5,),
            l2_values=(0.3,),
            friction_values=(0.38,),
            theta_resolution=16,
        )
        specs = build_slice_specs(grid)
        assert len(specs) == 1
        spec = specs[0]
        assert spec.m1 == 1.0
        assert spec.m2 == 2.0
        assert spec.l1 == 0.5
        assert spec.l2 == 0.3
        assert spec.friction == 0.38
        assert spec.theta_resolution == 16


# ---------------------------------------------------------------------------
# compute_slice
# ---------------------------------------------------------------------------


class TestComputeSlice:
    """Tests for single-slice computation."""

    def test_output_shape(self):
        spec = SliceSpec(
            flat_index=0,
            m1=1.0,
            m2=1.0,
            l1=1.0,
            l2=0.3,
            friction=0.38,
            theta_resolution=8,
        )
        result = compute_slice(spec)
        assert result.flat_index == 0
        assert result.n1.shape == (8, 8)
        assert result.n2.shape == (8, 8)
        assert result.brightness.shape == (8, 8)

    def test_output_dtypes(self):
        spec = SliceSpec(
            flat_index=0,
            m1=1.0,
            m2=1.0,
            l1=1.0,
            l2=0.3,
            friction=0.38,
            theta_resolution=8,
        )
        result = compute_slice(spec)
        assert result.n1.dtype == np.int8
        assert result.n2.dtype == np.int8
        assert result.brightness.dtype == np.uint8

    def test_near_origin_zero_winding(self):
        """A trajectory starting near (pi, pi) with high friction settles to (0,0)."""
        spec = SliceSpec(
            flat_index=0,
            m1=1.0,
            m2=1.0,
            l1=1.0,
            l2=0.3,
            friction=2.0,  # high friction → quick settling
            theta_resolution=4,
        )
        result = compute_slice(spec)
        # Center pixel (around theta=pi) should have winding (0, 0)
        center = result.n1[2, 2]
        assert center == 0


# ---------------------------------------------------------------------------
# Full pipeline (small scale)
# ---------------------------------------------------------------------------


class TestPrecomputeCube:
    """Integration test for the full precomputation pipeline."""

    def test_small_pipeline(self):
        """Run a minimal 2x1x1x1x2 grid at 8x8 and verify round-trip."""
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0,),
            l1_values=(1.0,),
            l2_values=(0.3,),
            friction_values=(0.38, 1.0),
            theta_resolution=8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_cube.npz"
            precompute_cube(grid, output_path, n_workers=1)

            # Verify file exists and is loadable
            assert output_path.exists()
            cube = DataCubeIndex.from_npz(output_path)

            # Verify grid matches
            assert cube.grid == grid

            # Verify lookup works
            result = cube.nearest_lookup(1.0, 1.0, 1.0, 0.3, 0.38)
            assert result.n1.shape == (8, 8)
            assert result.brightness.shape == (8, 8)

            # Different params should give different slices
            r1 = cube.nearest_lookup(1.0, 1.0, 1.0, 0.3, 0.38)
            r2 = cube.nearest_lookup(2.0, 1.0, 1.0, 0.3, 1.0)
            # At least some difference expected between very different params
            assert not np.array_equal(r1.brightness, r2.brightness)
