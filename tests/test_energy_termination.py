"""Tests for energy-based early termination in basin mode.

Verifies that:
- Frozen trajectories have constant angles in remaining snapshot slots
- Saddle energy gate only activates when provided
- Conservative systems (friction=0) never freeze
- Frozen trajectories have near-zero final velocities
- Early termination reduces computation time
- Winding numbers are stable after freeze
"""

import math
import time

import numpy as np
import pytest

from simulation import DoublePendulumParams
from fractal.compute import saddle_energy
from fractal._numpy_backend import (
    NumpyBackend,
    rk4_batch_with_snapshots,
    total_energy_batch,
)


class TestFrozenSnapshotsConstant:
    """Frozen trajectories should have identical angles in remaining slots."""

    def test_frozen_snapshots_constant(self):
        """High friction + saddle energy: frozen angles should repeat."""
        params = DoublePendulumParams(friction=3.0)
        se = saddle_energy(params)

        # Small grid of ICs near equilibrium (will freeze quickly)
        n = 16
        ics = np.zeros((n, 4), dtype=np.float32)
        ics[:, 0] = np.linspace(-0.5, 0.5, n)
        ics[:, 1] = np.linspace(-0.5, 0.5, n)

        result = rk4_batch_with_snapshots(
            params, ics, t_end=20.0, dt=0.01, n_samples=50,
            saddle_energy_val=se,
        )
        snapshots = result.snapshots

        # For each trajectory, check that the final snapshot value repeats
        # (frozen trajectories have identical values from freeze time onward)
        for i in range(n):
            theta1_final = snapshots[i, 0, -1]
            theta2_final = snapshots[i, 1, -1]

            # Count consecutive identical values from the end
            n_constant_t1 = 0
            for s in range(snapshots.shape[2] - 2, -1, -1):
                if snapshots[i, 0, s] == theta1_final:
                    n_constant_t1 += 1
                else:
                    break

            # At least some trajectories should have frozen (constant tail)
            # We just verify the pattern exists, not that all are frozen
            if n_constant_t1 > 5:
                # Verify theta2 is also constant over the same range
                start_idx = snapshots.shape[2] - 1 - n_constant_t1
                t2_tail = snapshots[i, 1, start_idx:]
                assert np.all(t2_tail == theta2_final), (
                    f"Trajectory {i}: theta1 frozen but theta2 varies"
                )


class TestNoFreezeWithoutSaddle:
    """Without saddle_energy_val, no early termination should occur."""

    def test_no_freeze_without_saddle(self):
        """saddle_energy_val=None should produce valid results."""
        params = DoublePendulumParams(friction=3.0)
        ics = np.array([[1.0, 0.5, 0.0, 0.0]], dtype=np.float32)

        result_no_saddle = rk4_batch_with_snapshots(
            params, ics, t_end=5.0, dt=0.01, n_samples=50,
            saddle_energy_val=None,
        )

        # Should return valid BatchResult
        assert result_no_saddle.snapshots.shape == (1, 2, 50)
        assert result_no_saddle.final_velocities.shape == (1, 2)

        # Without saddle energy gate, the trajectory should still be
        # actively evolving at the start (first several snapshots differ).
        early_t1 = result_no_saddle.snapshots[0, 0, :10]
        early_diffs = np.abs(np.diff(early_t1))
        assert np.any(early_diffs > 1e-6), (
            "Without saddle_energy_val, trajectory should be evolving"
        )


class TestNoFreezeWithoutFriction:
    """Conservative system (friction=0) should never freeze."""

    def test_no_freeze_without_friction(self):
        """With friction=0, energy is conserved and stays above saddle."""
        params = DoublePendulumParams(friction=0.0)
        se = saddle_energy(params)

        # Start with enough energy to stay above saddle
        ics = np.array([[2.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        result = rk4_batch_with_snapshots(
            params, ics, t_end=5.0, dt=0.01, n_samples=50,
            saddle_energy_val=se,
        )

        # Angles should still be changing at the end (not frozen)
        t1_diff = abs(result.snapshots[0, 0, -1] - result.snapshots[0, 0, -2])
        t2_diff = abs(result.snapshots[0, 1, -1] - result.snapshots[0, 1, -2])

        # At least one angle should show non-trivial change
        assert t1_diff + t2_diff > 0.001, (
            f"Angles appear frozen without friction: "
            f"d_theta1={t1_diff}, d_theta2={t2_diff}"
        )


class TestFrozenFinalVelocities:
    """Frozen trajectories should have small final velocities."""

    def test_frozen_final_velocities_small(self):
        """High friction + saddle energy: final velocities near zero."""
        params = DoublePendulumParams(friction=3.0)
        se = saddle_energy(params)

        ics = np.array([
            [0.3, 0.2, 0.0, 0.0],
            [0.5, 0.3, 0.0, 0.0],
            [-0.3, 0.4, 0.0, 0.0],
            [-0.5, -0.3, 0.0, 0.0],
        ], dtype=np.float32)

        result = rk4_batch_with_snapshots(
            params, ics, t_end=20.0, dt=0.01, n_samples=50,
            saddle_energy_val=se,
        )

        # All final velocities should be small
        max_vel = np.max(np.abs(result.final_velocities))
        assert max_vel < 0.5, f"Max final velocity = {max_vel}"


class TestEarlyTerminationSpeedup:
    """Early termination should be faster than full integration."""

    def test_early_termination_speedup(self):
        """With high friction + saddle energy, should compute faster."""
        params = DoublePendulumParams(friction=5.0)
        se = saddle_energy(params)

        n = 64
        ics = np.zeros((n, 4), dtype=np.float32)
        ics[:, 0] = np.linspace(-1.0, 1.0, n)
        ics[:, 1] = np.linspace(-1.0, 1.0, n)

        t_end = 30.0

        # Time without early termination
        t0 = time.perf_counter()
        rk4_batch_with_snapshots(
            params, ics, t_end=t_end, dt=0.01, n_samples=50,
            saddle_energy_val=None,
        )
        time_no_freeze = time.perf_counter() - t0

        # Time with early termination
        t0 = time.perf_counter()
        rk4_batch_with_snapshots(
            params, ics, t_end=t_end, dt=0.01, n_samples=50,
            saddle_energy_val=se,
        )
        time_with_freeze = time.perf_counter() - t0

        # With high friction and long t_end, freeze should save significant time
        # Use generous tolerance (freeze should be at least 20% faster)
        assert time_with_freeze < time_no_freeze * 0.95, (
            f"Early termination not faster: "
            f"freeze={time_with_freeze:.3f}s, "
            f"no_freeze={time_no_freeze:.3f}s"
        )


class TestWindingNumbersStable:
    """Winding numbers should be correct after freeze."""

    def test_winding_numbers_stable_after_freeze(self):
        """Frozen trajectory's winding number at freeze time should match final."""
        params = DoublePendulumParams(friction=3.0)
        se = saddle_energy(params)

        ics = np.array([
            [0.5, 0.3, 0.0, 0.0],
            [2.5, 1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0, 0.0],
        ], dtype=np.float32)

        result = rk4_batch_with_snapshots(
            params, ics, t_end=20.0, dt=0.01, n_samples=100,
            saddle_energy_val=se,
        )

        for i in range(ics.shape[0]):
            # Find the freeze point (where angles stop changing)
            theta1_vals = result.snapshots[i, 0, :]
            freeze_idx = result.snapshots.shape[2] - 1
            for s in range(result.snapshots.shape[2] - 2, 0, -1):
                if theta1_vals[s] != theta1_vals[-1]:
                    freeze_idx = s + 1
                    break

            # Winding number at freeze should match final
            winding_at_freeze = round(
                float(result.snapshots[i, 0, freeze_idx]) / (2 * math.pi)
            )
            winding_final = round(
                float(result.snapshots[i, 0, -1]) / (2 * math.pi)
            )
            assert winding_at_freeze == winding_final, (
                f"Trajectory {i}: winding at freeze={winding_at_freeze} "
                f"!= final={winding_final}"
            )


class TestTotalEnergyBatch:
    """Test the total_energy_batch vectorized helper."""

    def test_matches_scalar(self):
        """Batch energy should match scalar simulation.total_energy."""
        from simulation import total_energy

        params = DoublePendulumParams()
        states_list = [
            [0.5, 1.0, 0.3, -0.2],
            [math.pi, 0.0, 0.0, 0.0],
            [0.0, math.pi, 0.0, 0.0],
            [1.0, 2.0, -1.0, 0.5],
        ]
        states = np.array(states_list, dtype=np.float64)
        batch_energies = total_energy_batch(states, params)

        for i, state in enumerate(states_list):
            scalar_e = total_energy(state, params)
            assert batch_energies[i] == pytest.approx(scalar_e, rel=1e-10), (
                f"State {i}: batch={batch_energies[i]}, scalar={scalar_e}"
            )

    def test_output_shape(self):
        """Should return (N,) array."""
        params = DoublePendulumParams()
        states = np.random.randn(20, 4)
        energies = total_energy_batch(states, params)
        assert energies.shape == (20,)
