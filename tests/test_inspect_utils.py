"""Tests for inspect column utilities: rk4_single_trajectory and get_single_winding_color."""

import math

import numpy as np
import pytest

from simulation import DoublePendulumParams
from fractal._numpy_backend import rk4_single_trajectory
from fractal.winding import (
    get_single_winding_color,
    winding_modular_grid,
    winding_basin_hash,
)


class TestRk4SingleTrajectory:
    """Tests for the single-trajectory RK4 integrator."""

    def test_output_shape(self):
        """Return shape should be (n_steps + 1, 4)."""
        params = DoublePendulumParams(friction=1.0)
        result = rk4_single_trajectory(params, 0.1, 0.2, t_end=1.0, dt=0.01)
        n_steps = int(1.0 / 0.01)
        assert result.shape == (n_steps + 1, 4)

    def test_output_dtype(self):
        """Result should be float32."""
        params = DoublePendulumParams(friction=1.0)
        result = rk4_single_trajectory(params, 0.1, 0.2, t_end=0.5, dt=0.01)
        assert result.dtype == np.float32

    def test_initial_state_preserved(self):
        """First row should match the provided initial conditions."""
        params = DoublePendulumParams()
        theta1, theta2 = 0.3, -0.5
        result = rk4_single_trajectory(params, theta1, theta2, t_end=0.5, dt=0.01)
        np.testing.assert_allclose(result[0, 0], theta1, atol=1e-6)
        np.testing.assert_allclose(result[0, 1], theta2, atol=1e-6)
        np.testing.assert_allclose(result[0, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 3], 0.0, atol=1e-6)

    def test_zero_initial_stays_zero(self):
        """Starting at the downward equilibrium should remain near zero."""
        params = DoublePendulumParams(friction=2.0)
        result = rk4_single_trajectory(params, 0.0, 0.0, t_end=1.0, dt=0.01)
        # All states should remain at zero
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_with_friction_decays(self):
        """With friction, a small perturbation should decay over time."""
        params = DoublePendulumParams(friction=2.0)
        result = rk4_single_trajectory(params, 0.3, 0.2, t_end=20.0, dt=0.01)
        # Final velocities should be near zero
        omega1_final = abs(result[-1, 2])
        omega2_final = abs(result[-1, 3])
        assert omega1_final < 0.01
        assert omega2_final < 0.01

    def test_without_friction_conserves_energy(self):
        """Without friction, total energy should be approximately conserved."""
        params = DoublePendulumParams(friction=0.0)
        result = rk4_single_trajectory(params, 0.5, 0.3, t_end=2.0, dt=0.001)

        def energy(state):
            th1, th2, om1, om2 = state
            m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g
            kinetic = (
                0.5 * (m1 + m2) * l1**2 * om1**2
                + 0.5 * m2 * l2**2 * om2**2
                + m2 * l1 * l2 * om1 * om2 * np.cos(th1 - th2)
            )
            potential = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
            return kinetic + potential

        e_initial = energy(result[0].astype(np.float64))
        e_final = energy(result[-1].astype(np.float64))
        # Energy should be conserved to within ~1% for this short integration
        np.testing.assert_allclose(e_final, e_initial, rtol=0.01)

    def test_matches_batch_backend(self):
        """Single trajectory should match batch simulation at final step."""
        from fractal._numpy_backend import NumpyBackend

        params = DoublePendulumParams(friction=1.0)
        theta1, theta2 = 0.5, -0.3
        t_end = 5.0
        dt = 0.01

        # Single trajectory
        single = rk4_single_trajectory(params, theta1, theta2, t_end, dt)

        # Batch with N=1 (basin mode, final state only)
        backend = NumpyBackend()
        ics = np.array([[theta1, theta2, 0.0, 0.0]], dtype=np.float32)
        basin_result = backend.simulate_basin_batch(
            params, ics, t_end=t_end, dt=dt,
        )

        # Final states should match
        np.testing.assert_allclose(
            single[-1], basin_result.final_state[0], atol=1e-4,
        )


class TestGetSingleWindingColor:
    """Tests for get_single_winding_color utility."""

    def test_returns_tuple_of_four_ints(self):
        """Should return (b, g, r, a) tuple."""
        result = get_single_winding_color(0, 0, winding_modular_grid)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(x, int) for x in result)

    def test_values_in_range(self):
        """All values should be in [0, 255]."""
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                b, g, r, a = get_single_winding_color(
                    n1, n2, winding_modular_grid,
                )
                assert 0 <= b <= 255
                assert 0 <= g <= 255
                assert 0 <= r <= 255
                assert 0 <= a <= 255

    def test_alpha_is_255(self):
        """All winding colormaps should return full alpha."""
        for colormap_fn in [
            winding_modular_grid,
            winding_basin_hash,
        ]:
            _, _, _, a = get_single_winding_color(1, 0, colormap_fn)
            assert a == 255

    def test_matches_batch_colormap(self):
        """Single-point output should match batch colormap at same index."""
        n1, n2 = 2, -1
        n1_arr = np.array([n1], dtype=np.int32)
        n2_arr = np.array([n2], dtype=np.int32)

        for colormap_fn in [
            winding_modular_grid,
            winding_basin_hash,
        ]:
            single_bgra = get_single_winding_color(n1, n2, colormap_fn)
            batch_bgra = colormap_fn(n1_arr, n2_arr)
            expected = tuple(int(x) for x in batch_bgra[0])
            assert single_bgra == expected, (
                f"{colormap_fn.__name__}: {single_bgra} != {expected}"
            )

    def test_different_pairs_different_colors(self):
        """Different winding pairs should produce different colors (usually)."""
        c1 = get_single_winding_color(0, 0, winding_basin_hash)
        c2 = get_single_winding_color(1, 0, winding_basin_hash)
        c3 = get_single_winding_color(0, 1, winding_basin_hash)
        # At least some should differ
        assert c1 != c2 or c1 != c3, "Expected some color variation"
