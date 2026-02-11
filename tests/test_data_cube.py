"""Tests for fractal.data_cube: ParamGridSpec, CubeSlice, DataCubeIndex."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fractal.data_cube import (
    CubeSlice,
    DataCubeIndex,
    ParamGridSpec,
    _nearest_index,
    save_cube,
)


# ---------------------------------------------------------------------------
# ParamGridSpec
# ---------------------------------------------------------------------------


class TestParamGridSpec:
    """Tests for ParamGridSpec data structure."""

    def test_n_slices_small_grid(self):
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0,),
            l1_values=(1.0,),
            l2_values=(1.0,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )
        assert grid.n_slices == 2 * 1 * 1 * 1 * 2  # = 4

    def test_n_slices_default_grid(self):
        from fractal.data_cube import DEFAULT_GRID
        assert DEFAULT_GRID.n_slices == 5 * 5 * 5 * 5 * 8  # = 5000

    def test_shape(self):
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0, 3.0),
            m2_values=(1.0, 2.0),
            l1_values=(0.5,),
            l2_values=(0.5, 1.0),
            friction_values=(0.1, 0.2, 0.3),
            theta_resolution=16,
        )
        assert grid.shape == (3, 2, 1, 2, 3)

    def test_flat_index_first(self):
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0, 2.0),
            l1_values=(1.0,),
            l2_values=(1.0,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )
        assert grid.flat_index(0, 0, 0, 0, 0) == 0

    def test_flat_index_last(self):
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0, 2.0),
            l1_values=(1.0,),
            l2_values=(1.0,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )
        last = grid.n_slices - 1
        assert grid.flat_index(1, 1, 0, 0, 1) == last

    def test_flat_index_c_order(self):
        """Verify last axis (friction) varies fastest."""
        grid = ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0,),
            l1_values=(1.0,),
            l2_values=(1.0,),
            friction_values=(0.1, 0.5, 1.0),
            theta_resolution=8,
        )
        # m1=0: friction indices 0, 1, 2
        assert grid.flat_index(0, 0, 0, 0, 0) == 0
        assert grid.flat_index(0, 0, 0, 0, 1) == 1
        assert grid.flat_index(0, 0, 0, 0, 2) == 2
        # m1=1: friction indices 0, 1, 2
        assert grid.flat_index(1, 0, 0, 0, 0) == 3
        assert grid.flat_index(1, 0, 0, 0, 1) == 4
        assert grid.flat_index(1, 0, 0, 0, 2) == 5

    def test_frozen_dataclass(self):
        from fractal.data_cube import DEFAULT_GRID
        with pytest.raises(AttributeError):
            DEFAULT_GRID.theta_resolution = 128  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _nearest_index
# ---------------------------------------------------------------------------


class TestNearestIndex:
    """Tests for the bisect-based nearest-neighbor lookup."""

    def test_exact_match(self):
        values = (0.1, 0.5, 1.0, 2.0)
        assert _nearest_index(values, 0.5) == 1

    def test_between_values_closer_to_lower(self):
        values = (0.1, 0.5, 1.0, 2.0)
        # 0.3 is 0.2 from 0.1 and 0.2 from 0.5 — tie goes to lower
        assert _nearest_index(values, 0.3) == 0
        assert _nearest_index(values, 0.29) == 0  # closer to 0.1

    def test_between_values_closer_to_upper(self):
        values = (0.1, 0.5, 1.0, 2.0)
        assert _nearest_index(values, 0.8) == 2  # closer to 1.0

    def test_below_range(self):
        values = (0.5, 1.0, 2.0)
        assert _nearest_index(values, 0.0) == 0

    def test_above_range(self):
        values = (0.5, 1.0, 2.0)
        assert _nearest_index(values, 10.0) == 2

    def test_single_value(self):
        values = (1.0,)
        assert _nearest_index(values, 0.0) == 0
        assert _nearest_index(values, 100.0) == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _nearest_index((), 1.0)

    def test_tie_prefers_lower(self):
        values = (1.0, 3.0)
        # 2.0 is exactly between 1.0 and 3.0 — should prefer lower
        assert _nearest_index(values, 2.0) == 0


# ---------------------------------------------------------------------------
# CubeSlice
# ---------------------------------------------------------------------------


class TestCubeSlice:
    """Tests for CubeSlice named tuple."""

    def test_fields(self):
        n1 = np.zeros((8, 8), dtype=np.int8)
        n2 = np.ones((8, 8), dtype=np.int8)
        brightness = np.full((8, 8), 200, dtype=np.uint8)
        s = CubeSlice(n1=n1, n2=n2, brightness=brightness)
        assert s.n1 is n1
        assert s.n2 is n2
        assert s.brightness is brightness

    def test_tuple_unpacking(self):
        n1 = np.zeros((4, 4), dtype=np.int8)
        n2 = np.ones((4, 4), dtype=np.int8)
        brightness = np.full((4, 4), 128, dtype=np.uint8)
        a, b, c = CubeSlice(n1=n1, n2=n2, brightness=brightness)
        assert a is n1
        assert b is n2
        assert c is brightness


# ---------------------------------------------------------------------------
# DataCubeIndex
# ---------------------------------------------------------------------------


def _make_test_cube(
    grid: ParamGridSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create test data arrays for a given grid spec."""
    R = grid.theta_resolution
    n = grid.n_slices
    n1 = np.zeros((n, R, R), dtype=np.int8)
    n2 = np.zeros((n, R, R), dtype=np.int8)
    brightness = np.full((n, R, R), 200, dtype=np.uint8)

    # Fill each slice with its index for identification
    for i in range(n):
        n1[i] = np.int8(i % 5)
        n2[i] = np.int8((i // 5) % 5)
        brightness[i] = np.uint8(50 + (i % 200))

    return n1, n2, brightness


class TestDataCubeIndex:
    """Tests for DataCubeIndex lookup and construction."""

    @pytest.fixture()
    def small_grid(self):
        return ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0, 2.0),
            l1_values=(1.0,),
            l2_values=(0.5,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )

    @pytest.fixture()
    def cube(self, small_grid):
        n1, n2, brightness = _make_test_cube(small_grid)
        return DataCubeIndex(small_grid, n1, n2, brightness)

    def test_memory_bytes(self, cube, small_grid):
        R = small_grid.theta_resolution
        n = small_grid.n_slices
        expected = 3 * n * R * R  # int8 + int8 + uint8
        assert cube.memory_bytes == expected

    def test_lookup_returns_cube_slice(self, cube):
        result = cube.nearest_lookup(1.0, 1.0, 1.0, 0.5, 0.1)
        assert isinstance(result, CubeSlice)

    def test_lookup_shape(self, cube, small_grid):
        result = cube.nearest_lookup(1.0, 1.0, 1.0, 0.5, 0.1)
        R = small_grid.theta_resolution
        assert result.n1.shape == (R, R)
        assert result.n2.shape == (R, R)
        assert result.brightness.shape == (R, R)

    def test_lookup_exact_match_index_0(self, cube, small_grid):
        """First grid point should return slice 0."""
        result = cube.nearest_lookup(1.0, 1.0, 1.0, 0.5, 0.1)
        expected_idx = small_grid.flat_index(0, 0, 0, 0, 0)
        assert expected_idx == 0
        # Check that the slice has the expected fill value
        assert result.n1[0, 0] == 0 % 5
        assert result.brightness[0, 0] == 50 + (0 % 200)

    def test_lookup_last_point(self, cube, small_grid):
        """Last grid point should return the last slice."""
        result = cube.nearest_lookup(2.0, 2.0, 1.0, 0.5, 0.5)
        expected_idx = small_grid.flat_index(1, 1, 0, 0, 1)
        assert expected_idx == small_grid.n_slices - 1

    def test_lookup_nearest_snap(self, cube):
        """Values between grid points should snap to nearest."""
        # m1=1.4 is closer to 1.0 than 2.0
        result_low = cube.nearest_lookup(1.4, 1.0, 1.0, 0.5, 0.1)
        result_exact = cube.nearest_lookup(1.0, 1.0, 1.0, 0.5, 0.1)
        np.testing.assert_array_equal(result_low.n1, result_exact.n1)

        # m1=1.6 is closer to 2.0 than 1.0
        result_high = cube.nearest_lookup(1.6, 1.0, 1.0, 0.5, 0.1)
        result_exact2 = cube.nearest_lookup(2.0, 1.0, 1.0, 0.5, 0.1)
        np.testing.assert_array_equal(result_high.n1, result_exact2.n1)

    def test_mismatched_shape_raises(self, small_grid):
        R = small_grid.theta_resolution
        n = small_grid.n_slices
        n1 = np.zeros((n, R, R), dtype=np.int8)
        n2 = np.zeros((n, R, R + 1), dtype=np.int8)  # wrong shape
        brightness = np.zeros((n, R, R), dtype=np.uint8)
        with pytest.raises(ValueError, match="n2_data shape"):
            DataCubeIndex(small_grid, n1, n2, brightness)


# ---------------------------------------------------------------------------
# Round-trip save/load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save_cube and DataCubeIndex.from_npz round-trip."""

    @pytest.fixture()
    def small_grid(self):
        return ParamGridSpec(
            m1_values=(1.0, 2.0),
            m2_values=(1.0,),
            l1_values=(1.0,),
            l2_values=(0.5,),
            friction_values=(0.1, 0.5),
            theta_resolution=8,
        )

    def test_round_trip(self, small_grid):
        n1, n2, brightness = _make_test_cube(small_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test_cube.npz"
            save_cube(npz_path, small_grid, n1, n2, brightness)

            # Verify files exist
            meta_path = npz_path.with_name("test_cube_meta.json")
            assert npz_path.exists()
            assert meta_path.exists()

            # Load and verify
            loaded = DataCubeIndex.from_npz(npz_path)
            assert loaded.grid == small_grid

            # Verify data matches
            result = loaded.nearest_lookup(1.0, 1.0, 1.0, 0.5, 0.1)
            np.testing.assert_array_equal(result.n1, n1[0])
            np.testing.assert_array_equal(result.n2, n2[0])
            np.testing.assert_array_equal(result.brightness, brightness[0])

    def test_metadata_json_readable(self, small_grid):
        n1, n2, brightness = _make_test_cube(small_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test_cube.npz"
            save_cube(npz_path, small_grid, n1, n2, brightness, version="2")

            meta_path = npz_path.with_name("test_cube_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)

            assert meta["version"] == "2"
            assert meta["grid"]["theta_resolution"] == 8
            assert meta["grid"]["m1_values"] == [1.0, 2.0]

    def test_missing_npz_raises(self):
        with pytest.raises(FileNotFoundError, match="Cube data not found"):
            DataCubeIndex.from_npz("/nonexistent/path.npz")

    def test_missing_metadata_raises(self, small_grid):
        n1, n2, brightness = _make_test_cube(small_grid)

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "test_cube.npz"
            save_cube(npz_path, small_grid, n1, n2, brightness)

            # Remove metadata
            meta_path = npz_path.with_name("test_cube_meta.json")
            meta_path.unlink()

            with pytest.raises(FileNotFoundError, match="Cube metadata not found"):
                DataCubeIndex.from_npz(npz_path)
