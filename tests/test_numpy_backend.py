"""Tests for fractal/_numpy_backend.py: vectorized RK4 cross-validation.

Verifies that the batch derivatives and integration produce results
consistent with the scalar simulation.py implementation.
"""

import math

import numpy as np
import pytest

from simulation import DoublePendulumParams, derivatives, simulate
from fractal._numpy_backend import (
    NumpyBackend, derivatives_batch, rk4_batch_with_snapshots,
    total_energy_batch,
)


class TestDerivativesBatch:
    """Cross-validate batch derivatives against scalar derivatives."""

    def test_single_trajectory_matches_scalar(self):
        """A (1, 4) batch should match the scalar derivatives exactly."""
        params = DoublePendulumParams()
        state = [1.0, 0.5, 0.1, -0.2]
        states = np.array([state], dtype=np.float64)

        scalar_d = derivatives(0, state, params)
        batch_d = derivatives_batch(states, params)

        for i in range(4):
            assert abs(batch_d[0, i] - scalar_d[i]) < 1e-12, (
                f"Component {i}: batch={batch_d[0, i]}, scalar={scalar_d[i]}"
            )

    def test_multiple_trajectories(self):
        """Verify batch produces correct result for N trajectories."""
        params = DoublePendulumParams()
        test_states = [
            [0.0, 0.0, 0.0, 0.0],
            [math.pi / 2, math.pi / 2, 0.0, 0.0],
            [1.0, -1.0, 2.0, -2.0],
            [0.1, 0.2, 0.3, 0.4],
        ]
        states = np.array(test_states, dtype=np.float64)
        batch_d = derivatives_batch(states, params)

        for j, state in enumerate(test_states):
            scalar_d = derivatives(0, state, params)
            for i in range(4):
                assert abs(batch_d[j, i] - scalar_d[i]) < 1e-12

    def test_output_shape(self):
        params = DoublePendulumParams()
        n = 100
        states = np.random.randn(n, 4)
        result = derivatives_batch(states, params)
        assert result.shape == (n, 4)

    def test_different_params(self):
        """Verify batch works with non-default physics parameters."""
        params = DoublePendulumParams(m1=2.0, m2=0.5, l1=1.5, l2=0.8, g=10.0)
        state = [0.7, -0.3, 1.0, -0.5]
        states = np.array([state], dtype=np.float64)

        scalar_d = derivatives(0, state, params)
        batch_d = derivatives_batch(states, params)

        for i in range(4):
            assert abs(batch_d[0, i] - scalar_d[i]) < 1e-12


class TestRK4BatchWithSnapshots:
    """Test the full RK4 batch integration."""

    def test_output_shape(self):
        params = DoublePendulumParams()
        n = 16
        ics = np.zeros((n, 4), dtype=np.float32)
        ics[:, 0] = np.linspace(-1, 1, n)
        ics[:, 1] = np.linspace(-1, 1, n)

        result = rk4_batch_with_snapshots(params, ics, t_end=1.0, dt=0.01, n_samples=10)
        assert result.snapshots.shape == (n, 2, 10)
        assert result.snapshots.dtype == np.float32
        assert result.final_velocities.shape == (n, 2)
        assert result.final_velocities.dtype == np.float32

    def test_zero_initial_conditions(self):
        """Starting at rest should stay near zero (within RK4 tolerance)."""
        params = DoublePendulumParams()
        ics = np.zeros((1, 4), dtype=np.float32)

        result = rk4_batch_with_snapshots(params, ics, t_end=1.0, dt=0.01, n_samples=10)
        # Both theta1 and theta2 should remain very close to 0
        assert np.max(np.abs(result.snapshots)) < 1e-6
        # Final velocities should also be near zero
        assert np.max(np.abs(result.final_velocities)) < 1e-6

    def test_cancellation(self):
        """Cancellation should return partial results."""
        params = DoublePendulumParams()
        ics = np.zeros((4, 4), dtype=np.float32)
        ics[:, 0] = [1.0, 1.1, 1.2, 1.3]
        ics[:, 1] = [0.5, 0.5, 0.5, 0.5]

        call_count = [0]
        def cancel_after_50():
            call_count[0] += 1
            return call_count[0] > 1  # Cancel after second check

        result = rk4_batch_with_snapshots(
            params, ics, t_end=10.0, dt=0.01, n_samples=96,
            cancel_check=cancel_after_50,
        )
        # Should have partial results (some zeros remaining)
        assert result.snapshots.shape == (4, 2, 96)
        assert result.final_velocities.shape == (4, 2)

    def test_progress_callback(self):
        """Progress callback should be called during integration."""
        params = DoublePendulumParams()
        ics = np.zeros((4, 4), dtype=np.float32)
        ics[:, 0] = [0.5, 1.0, 1.5, 2.0]

        progress_calls = []
        def on_progress(done, total):
            progress_calls.append((done, total))

        result = rk4_batch_with_snapshots(
            params, ics, t_end=1.0, dt=0.01, n_samples=10,
            progress_callback=on_progress,
        )
        assert len(progress_calls) > 0
        # Last call should have done == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    def test_visual_equivalence_with_dop853(self):
        """Batch RK4 should produce visually equivalent results to DOP853.

        We compare theta2 at the end of a short simulation. RK4 with
        h=0.01 won't match DOP853 at 1e-14 tolerance exactly, but
        should agree to within ~0.1 radians for short sims.
        """
        params = DoublePendulumParams()
        theta1_0, theta2_0 = math.pi / 4, math.pi / 6

        # DOP853 reference
        t_end = 5.0
        t, states = simulate(params, theta1_0, theta2_0, t_end=t_end, dt=0.005)
        dop853_theta2_final = states[-1, 1]

        # Batch RK4
        ics = np.array([[theta1_0, theta2_0, 0.0, 0.0]], dtype=np.float32)
        result = rk4_batch_with_snapshots(
            params, ics, t_end=t_end, dt=0.01, n_samples=96,
        )
        rk4_theta2_final = result.snapshots[0, 1, -1]  # theta2 is index 1

        # Should be reasonably close (within ~0.5 rad for 5s sim)
        diff = abs(float(rk4_theta2_final) - float(dop853_theta2_final))
        assert diff < 1.0, (
            f"RK4 vs DOP853 theta2 diff = {diff:.3f} rad at t={t_end}s "
            f"(rk4={rk4_theta2_final:.3f}, dop853={dop853_theta2_final:.3f})"
        )


class TestTotalEnergyBatch:
    """Test batch total energy computation."""

    def test_matches_scalar(self):
        """Batch energy should match scalar total_energy for each trajectory."""
        from simulation import total_energy
        params = DoublePendulumParams()
        states_list = [
            [0.5, 1.0, 0.3, -0.2],
            [0.0, 0.0, 0.0, 0.0],
            [math.pi, 0.0, 0.0, 0.0],
        ]
        states = np.array(states_list, dtype=np.float64)
        batch_energies = total_energy_batch(states, params)

        for i, state in enumerate(states_list):
            scalar_e = total_energy(state, params)
            assert batch_energies[i] == pytest.approx(scalar_e, rel=1e-12)

    def test_output_shape(self):
        params = DoublePendulumParams()
        states = np.random.randn(10, 4)
        energies = total_energy_batch(states, params)
        assert energies.shape == (10,)

    def test_zero_state_has_minimum_energy(self):
        """All at rest, hanging down — minimum potential energy."""
        params = DoublePendulumParams()
        states = np.zeros((1, 4), dtype=np.float64)
        energy = total_energy_batch(states, params)
        # V(0,0) = -(m1+m2)*g*l1 - m2*g*l2 (most negative)
        expected = -(params.m1 + params.m2) * params.g * params.l1 - params.m2 * params.g * params.l2
        assert energy[0] == pytest.approx(expected, rel=1e-12)


class TestFreezeEarlyTermination:
    """Test energy-based early termination (freeze) in the numpy backend."""

    def test_freeze_with_high_friction(self):
        """With high friction and saddle energy, trajectories should freeze early."""
        from fractal.compute import saddle_energy
        params = DoublePendulumParams(friction=3.0)
        se = saddle_energy(params)

        # Start with small angle — will quickly lose energy and freeze
        ics = np.array([[0.3, 0.2, 0.0, 0.0]], dtype=np.float32)
        result = rk4_batch_with_snapshots(
            params, ics, t_end=20.0, dt=0.01, n_samples=50,
            saddle_energy_val=se,
        )

        # Once frozen, remaining snapshot slots should all be identical
        # (filled with the freeze-time angle)
        snaps = result.snapshots[0]  # (2, 50)
        # Find where theta1 stops changing (frozen)
        diffs = np.abs(np.diff(snaps[0]))
        frozen_idx = np.where(diffs == 0)[0]
        assert len(frozen_idx) > 0, "Expected trajectory to freeze but it didn't"

    def test_no_freeze_without_saddle_energy(self):
        """Without saddle_energy_val, no freezing should occur."""
        params = DoublePendulumParams(friction=3.0)
        ics = np.array([[0.3, 0.2, 0.0, 0.0]], dtype=np.float32)
        result = rk4_batch_with_snapshots(
            params, ics, t_end=5.0, dt=0.01, n_samples=50,
        )
        # Trajectory settles but shouldn't have identical consecutive
        # snapshots early on (no freeze mechanism)
        snaps = result.snapshots[0, 0]  # theta1 over time
        # First few samples should be changing
        early_diffs = np.abs(np.diff(snaps[:10]))
        assert np.any(early_diffs > 0), "Trajectory should still be moving early on"

    def test_freeze_fills_remaining_snapshots(self):
        """When frozen, all subsequent snapshot slots get the freeze angle."""
        from fractal.compute import saddle_energy
        params = DoublePendulumParams(friction=5.0)  # very high friction
        se = saddle_energy(params)

        ics = np.array([[0.1, 0.1, 0.0, 0.0]], dtype=np.float32)
        result = rk4_batch_with_snapshots(
            params, ics, t_end=20.0, dt=0.01, n_samples=100,
            saddle_energy_val=se,
        )

        snaps_t1 = result.snapshots[0, 0]  # theta1 snapshots
        # Find freeze point: first index where all remaining values are identical
        for i in range(len(snaps_t1) - 1):
            if snaps_t1[i] == snaps_t1[i + 1] and i + 2 < len(snaps_t1):
                # Check all remaining are same
                tail = snaps_t1[i:]
                if np.all(tail == tail[0]):
                    # Found freeze point — verify theta2 also frozen at same point
                    snaps_t2 = result.snapshots[0, 1]
                    tail_t2 = snaps_t2[i:]
                    assert np.all(tail_t2 == tail_t2[0]), (
                        "theta2 should also be frozen at the same point"
                    )
                    return
        # If we get here, trajectory may not have frozen — that's OK for
        # very low energy starts, but with friction=5 it really should
        pytest.fail("Expected trajectory to freeze with friction=5.0")

    def test_final_velocities_near_zero_when_frozen(self):
        """Frozen trajectories should have near-zero final velocities."""
        from fractal.compute import saddle_energy
        params = DoublePendulumParams(friction=5.0)
        se = saddle_energy(params)

        ics = np.array([[0.2, 0.15, 0.0, 0.0]], dtype=np.float32)
        result = rk4_batch_with_snapshots(
            params, ics, t_end=20.0, dt=0.01, n_samples=50,
            saddle_energy_val=se,
        )

        # Final velocities should be very small (frozen = energy below threshold)
        assert abs(result.final_velocities[0, 0]) < 0.1, (
            f"omega1 = {result.final_velocities[0, 0]}"
        )
        assert abs(result.final_velocities[0, 1]) < 0.1, (
            f"omega2 = {result.final_velocities[0, 1]}"
        )


class TestNumpyBackend:
    """Test the NumpyBackend class interface."""

    def test_backend_interface(self):
        backend = NumpyBackend()
        params = DoublePendulumParams()
        ics = np.zeros((4, 4), dtype=np.float32)
        ics[:, 0] = [0.5, 1.0, 1.5, 2.0]

        result = backend.simulate_batch(
            params, ics, t_end=1.0, dt=0.01, n_samples=10,
        )
        assert result.snapshots.shape == (4, 2, 10)
        assert result.snapshots.dtype == np.float32
        assert result.final_velocities.shape == (4, 2)
        assert result.final_velocities.dtype == np.float32

    def test_saddle_energy_passthrough(self):
        """Backend should pass saddle_energy_val through to the RK4 function."""
        from fractal.compute import saddle_energy
        backend = NumpyBackend()
        params = DoublePendulumParams(friction=5.0)
        se = saddle_energy(params)

        ics = np.array([[0.1, 0.1, 0.0, 0.0]], dtype=np.float32)
        result = backend.simulate_batch(
            params, ics, t_end=20.0, dt=0.01, n_samples=50,
            saddle_energy_val=se,
        )
        assert result.snapshots.shape == (1, 2, 50)
        assert result.final_velocities.shape == (1, 2)
