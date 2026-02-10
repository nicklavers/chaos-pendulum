"""Tests for friction/damping across simulation and backends."""

import math

import numpy as np
import pytest

from simulation import DoublePendulumParams, derivatives, simulate, total_energy
from fractal._numpy_backend import NumpyBackend, derivatives_batch


class TestZeroFrictionUnchanged:
    """With friction=0, behavior should match original undamped physics."""

    def test_scalar_derivatives_zero_friction(self):
        """Zero friction should not alter accelerations."""
        params = DoublePendulumParams(friction=0.0)
        state = [0.5, 1.0, 0.3, -0.2]
        result = derivatives(0.0, state, params)
        # Re-derive without friction term: alpha = alpha - 0*omega = alpha
        assert len(result) == 4
        # d_theta1 = omega1, d_theta2 = omega2
        assert result[0] == pytest.approx(0.3)
        assert result[1] == pytest.approx(-0.2)

    def test_batch_derivatives_zero_friction(self):
        """Batch backend with friction=0 should match frictionless result."""
        params = DoublePendulumParams(friction=0.0)
        states = np.array([[0.5, 1.0, 0.3, -0.2]], dtype=np.float64)
        result = derivatives_batch(states, params)
        assert result.shape == (1, 4)
        assert result[0, 0] == pytest.approx(0.3)
        assert result[0, 1] == pytest.approx(-0.2)


class TestFrictionReducesAcceleration:
    """Positive friction should reduce angular acceleration magnitude."""

    def test_positive_omega_friction_reduces_alpha(self):
        """With positive omega and friction, alpha should decrease."""
        state = [0.5, 1.0, 2.0, 1.5]  # positive omegas

        params_no_fric = DoublePendulumParams(friction=0.0)
        derivs_no_fric = derivatives(0.0, state, params_no_fric)

        params_fric = DoublePendulumParams(friction=1.0)
        derivs_fric = derivatives(0.0, state, params_fric)

        # alpha1 should decrease by friction * omega1 = 1.0 * 2.0 = 2.0
        assert derivs_fric[2] == pytest.approx(derivs_no_fric[2] - 2.0)
        # alpha2 should decrease by friction * omega2 = 1.0 * 1.5 = 1.5
        assert derivs_fric[3] == pytest.approx(derivs_no_fric[3] - 1.5)


class TestBatchMatchesScalarWithFriction:
    """Batch numpy backend should match scalar derivatives with friction."""

    def test_single_trajectory_matches(self):
        """Batch with N=1 should match scalar derivatives."""
        params = DoublePendulumParams(friction=0.5)
        state = [0.7, 1.2, -0.5, 0.8]

        scalar_result = derivatives(0.0, state, params)

        batch_states = np.array([state], dtype=np.float64)
        batch_result = derivatives_batch(batch_states, params)

        for i in range(4):
            assert batch_result[0, i] == pytest.approx(scalar_result[i], rel=1e-10)

    def test_multiple_trajectories(self):
        """Batch with N=4 should match scalar for each trajectory."""
        params = DoublePendulumParams(friction=0.8, m1=1.5, l2=0.8)
        states_list = [
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 2.0, -1.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [math.pi, math.pi, 0.1, -0.1],
        ]

        batch_states = np.array(states_list, dtype=np.float64)
        batch_result = derivatives_batch(batch_states, params)

        for i, state in enumerate(states_list):
            scalar = derivatives(0.0, state, params)
            for j in range(4):
                assert batch_result[i, j] == pytest.approx(
                    scalar[j], rel=1e-10
                ), f"Mismatch at trajectory {i}, component {j}"


class TestEnergyDecreases:
    """With friction > 0, total energy should monotonically decrease."""

    def test_energy_decreases_over_time(self):
        """Energy should decrease when friction is applied."""
        params = DoublePendulumParams(friction=0.5)
        t_arr, states = simulate(
            params, theta1_0=1.0, theta2_0=0.5,
            omega1_0=1.0, omega2_0=-0.5,
            t_end=5.0, dt=0.01,
        )

        energies = [total_energy(states[i], params) for i in range(len(t_arr))]

        # Energy at end should be less than at start
        assert energies[-1] < energies[0]

        # Check that energy is non-increasing (allowing small numerical noise)
        for i in range(1, len(energies)):
            assert energies[i] <= energies[i - 1] + 1e-6, (
                f"Energy increased at step {i}: "
                f"{energies[i - 1]:.8f} -> {energies[i]:.8f}"
            )


class TestConvergesToRest:
    """High friction should cause velocities to approach zero."""

    def test_high_friction_settles(self):
        """Strong damping should drive omegas near zero."""
        params = DoublePendulumParams(friction=3.0)
        t_arr, states = simulate(
            params, theta1_0=1.0, theta2_0=0.5,
            omega1_0=2.0, omega2_0=-1.0,
            t_end=10.0, dt=0.005,
        )

        final_state = states[-1]
        omega1_final = final_state[2]
        omega2_final = final_state[3]

        assert abs(omega1_final) < 0.01, f"omega1 = {omega1_final}"
        assert abs(omega2_final) < 0.01, f"omega2 = {omega2_final}"

    def test_numpy_backend_settles(self):
        """NumPy batch backend should also settle with high friction."""
        params = DoublePendulumParams(friction=3.0)
        backend = NumpyBackend()

        ics = np.array([[1.0, 0.5, 2.0, -1.0]], dtype=np.float32)
        result = backend.simulate_batch(
            params, ics, t_end=10.0, dt=0.01, n_samples=100,
        )

        # Angles should have converged (small difference between last two samples)
        theta1_diff = abs(result.snapshots[0, 0, -1] - result.snapshots[0, 0, -2])
        theta2_diff = abs(result.snapshots[0, 1, -1] - result.snapshots[0, 1, -2])

        assert theta1_diff < 0.05, f"theta1 still changing: {theta1_diff}"
        assert theta2_diff < 0.05, f"theta2 still changing: {theta2_diff}"

    def test_final_velocities_near_zero(self):
        """With high friction and long sim, final velocities should be near zero."""
        params = DoublePendulumParams(friction=3.0)
        backend = NumpyBackend()

        ics = np.array([[1.0, 0.5, 2.0, -1.0]], dtype=np.float32)
        result = backend.simulate_batch(
            params, ics, t_end=10.0, dt=0.01, n_samples=100,
        )

        assert abs(result.final_velocities[0, 0]) < 0.01, (
            f"omega1 = {result.final_velocities[0, 0]}"
        )
        assert abs(result.final_velocities[0, 1]) < 0.01, (
            f"omega2 = {result.final_velocities[0, 1]}"
        )
