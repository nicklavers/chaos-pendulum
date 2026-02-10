"""Tests for fractal/basin_solver.py: DOP853 basin-mode integrator."""

import math

import numpy as np
import pytest

from simulation import DoublePendulumParams
from fractal.basin_solver import simulate_basin_batch
from fractal.compute import BasinResult, saddle_energy
from fractal.winding import extract_winding_numbers


def _small_grid(resolution: int = 4, center: float = 0.5, span: float = 1.0):
    """Build a small (N, 4) IC grid for testing."""
    n = resolution * resolution
    theta1 = np.linspace(center - span / 2, center + span / 2, resolution)
    theta2 = np.linspace(center - span / 2, center + span / 2, resolution)
    t1, t2 = np.meshgrid(theta1, theta2)
    ics = np.zeros((n, 4), dtype=np.float32)
    ics[:, 0] = t1.ravel()
    ics[:, 1] = t2.ravel()
    return ics


class TestOutputShape:
    """Verify output shape and dtype."""

    def test_returns_basin_result(self):
        params = DoublePendulumParams(friction=1.0)
        ics = _small_grid(4)
        result = simulate_basin_batch(params, ics, t_end=1.0)
        assert isinstance(result, BasinResult)

    def test_final_state_shape(self):
        params = DoublePendulumParams(friction=1.0)
        ics = _small_grid(4)
        result = simulate_basin_batch(params, ics, t_end=1.0)
        assert result.final_state.shape == (16, 4)
        assert result.final_state.dtype == np.float32

    def test_single_trajectory(self):
        params = DoublePendulumParams(friction=1.0)
        ics = np.array([[0.1, 0.2, 0.0, 0.0]], dtype=np.float32)
        result = simulate_basin_batch(params, ics, t_end=1.0)
        assert result.final_state.shape == (1, 4)


class TestWindingNumberAccuracy:
    """Verify winding numbers from basin solver are correct."""

    def test_near_origin_settles_to_zero_winding(self):
        """ICs near (0, 0) with friction should settle to winding (0, 0)."""
        params = DoublePendulumParams(friction=2.0)
        # Small angles near the downward equilibrium
        ics = np.array([
            [0.1, 0.1, 0.0, 0.0],
            [-0.1, 0.1, 0.0, 0.0],
            [0.1, -0.1, 0.0, 0.0],
            [-0.1, -0.1, 0.0, 0.0],
        ], dtype=np.float32)
        result = simulate_basin_batch(params, ics, t_end=20.0)

        theta1_final = result.final_state[:, 0]
        theta2_final = result.final_state[:, 1]
        n1, n2 = extract_winding_numbers(theta1_final, theta2_final)
        np.testing.assert_array_equal(n1, 0)
        np.testing.assert_array_equal(n2, 0)

    def test_near_origin_small_final_velocity(self):
        """With friction, final velocities should be near zero."""
        params = DoublePendulumParams(friction=2.0)
        ics = np.array([[0.3, -0.2, 0.0, 0.0]], dtype=np.float32)
        result = simulate_basin_batch(params, ics, t_end=20.0)
        omega1 = abs(result.final_state[0, 2])
        omega2 = abs(result.final_state[0, 3])
        assert omega1 < 0.01
        assert omega2 < 0.01


class TestEnergyTermination:
    """Verify energy-based early termination via events."""

    def test_with_saddle_energy(self):
        """Should terminate early and still produce correct winding numbers."""
        params = DoublePendulumParams(friction=3.0)
        saddle_val = saddle_energy(params)
        ics = np.array([[0.2, 0.2, 0.0, 0.0]], dtype=np.float32)

        result = simulate_basin_batch(
            params, ics, t_end=100.0, saddle_energy_val=saddle_val,
        )

        # Should still settle to (0, 0) winding
        theta1_final = result.final_state[:, 0]
        theta2_final = result.final_state[:, 1]
        n1, n2 = extract_winding_numbers(theta1_final, theta2_final)
        assert n1[0] == 0
        assert n2[0] == 0

    def test_no_event_without_friction(self):
        """Without friction, energy event should not be used."""
        params = DoublePendulumParams(friction=0.0)
        saddle_val = saddle_energy(params)
        ics = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)

        # Should complete to t_end without error
        result = simulate_basin_batch(
            params, ics, t_end=1.0, saddle_energy_val=saddle_val,
        )
        assert result.final_state.shape == (1, 4)

    def test_no_saddle_energy_still_works(self):
        """Without saddle_energy_val, should integrate to t_end."""
        params = DoublePendulumParams(friction=1.0)
        ics = np.array([[0.5, 0.5, 0.0, 0.0]], dtype=np.float32)

        result = simulate_basin_batch(params, ics, t_end=5.0)
        assert result.final_state.shape == (1, 4)


class TestCancellation:
    """Verify cancel_check support."""

    def test_cancel_returns_partial(self):
        """Cancelling mid-batch should return a partial result."""
        params = DoublePendulumParams(friction=1.0)
        ics = _small_grid(8)  # 64 trajectories
        call_count = 0

        def cancel_after_first_check():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Cancel on second check

        result = simulate_basin_batch(
            params, ics, t_end=5.0, cancel_check=cancel_after_first_check,
        )
        # Should still return a BasinResult
        assert isinstance(result, BasinResult)
        assert result.final_state.shape == (64, 4)


class TestProgressCallback:
    """Verify progress reporting."""

    def test_progress_called(self):
        params = DoublePendulumParams(friction=1.0)
        ics = _small_grid(4)  # 16 trajectories
        calls = []

        def track_progress(done, total):
            calls.append((done, total))

        simulate_basin_batch(
            params, ics, t_end=1.0, progress_callback=track_progress,
        )

        # Should have at least the initial and final call
        assert len(calls) >= 2
        # Final call should report all done
        assert calls[-1] == (16, 16)
        # Total should always be 16
        assert all(total == 16 for _, total in calls)


class TestCrossValidation:
    """Cross-validate winding numbers against RK4 backend."""

    def test_winding_numbers_match_rk4(self):
        """Basin solver winding numbers should match RK4 backend results."""
        from fractal._numpy_backend import NumpyBackend

        params = DoublePendulumParams(friction=2.0)
        saddle_val = saddle_energy(params)
        ics = _small_grid(4, center=0.5, span=2.0)
        t_end = 20.0

        # Basin solver (DOP853)
        basin_result = simulate_basin_batch(
            params, ics, t_end=t_end, saddle_energy_val=saddle_val,
        )
        basin_n1, basin_n2 = extract_winding_numbers(
            basin_result.final_state[:, 0],
            basin_result.final_state[:, 1],
        )

        # RK4 backend
        backend = NumpyBackend()
        rk4_result = backend.simulate_batch(
            params=params,
            initial_conditions=ics,
            t_end=t_end,
            dt=0.01,
            n_samples=96,
            saddle_energy_val=saddle_val,
        )
        rk4_n1, rk4_n2 = extract_winding_numbers(
            rk4_result.snapshots[:, 0, -1],
            rk4_result.snapshots[:, 1, -1],
        )

        np.testing.assert_array_equal(basin_n1, rk4_n1)
        np.testing.assert_array_equal(basin_n2, rk4_n2)
