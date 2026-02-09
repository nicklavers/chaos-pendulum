"""Tests for simulation.py: energy conservation, reversibility, known cases."""

import math

import numpy as np
import pytest

from simulation import (
    DoublePendulumParams, derivatives, simulate, positions, total_energy,
)


class TestDerivatives:
    """Test the derivatives function for known states."""

    def test_zero_state_zero_derivatives(self):
        """At rest hanging straight down, angular accelerations should be zero."""
        params = DoublePendulumParams()
        state = [0.0, 0.0, 0.0, 0.0]
        d = derivatives(0, state, params)
        # omega1=0, omega2=0 -> d_theta1=0, d_theta2=0
        assert d[0] == 0.0
        assert d[1] == 0.0
        # At theta=0 (hanging down), sin(0)=0 so alpha=0
        assert abs(d[2]) < 1e-10
        assert abs(d[3]) < 1e-10

    def test_horizontal_initial_has_acceleration(self):
        """A pendulum starting horizontal should have nonzero alpha."""
        params = DoublePendulumParams()
        state = [math.pi / 2, math.pi / 2, 0.0, 0.0]
        d = derivatives(0, state, params)
        # Both bobs are horizontal -> gravity should pull them down
        assert d[2] != 0 or d[3] != 0

    def test_returns_four_values(self):
        params = DoublePendulumParams()
        state = [0.5, 0.5, 1.0, -1.0]
        d = derivatives(0, state, params)
        assert len(d) == 4


class TestSimulate:
    """Test the simulate function."""

    def test_returns_correct_shapes(self):
        params = DoublePendulumParams()
        t, states = simulate(params, 1.0, 1.0, t_end=1.0, dt=0.01)
        assert t.ndim == 1
        assert states.ndim == 2
        assert states.shape[1] == 4
        assert len(t) == len(states)

    def test_initial_conditions_preserved(self):
        params = DoublePendulumParams()
        theta1_0, theta2_0, omega1_0, omega2_0 = 1.0, 0.5, 0.1, -0.2
        t, states = simulate(params, theta1_0, theta2_0, omega1_0, omega2_0,
                             t_end=1.0)
        assert abs(states[0, 0] - theta1_0) < 1e-10
        assert abs(states[0, 1] - theta2_0) < 1e-10
        assert abs(states[0, 2] - omega1_0) < 1e-10
        assert abs(states[0, 3] - omega2_0) < 1e-10


class TestEnergyConservation:
    """Test energy conservation over simulation."""

    def test_energy_drift_within_tolerance(self):
        """Energy should be conserved to high precision with DOP853."""
        params = DoublePendulumParams()
        t, states = simulate(params, math.pi / 2, math.pi / 2,
                             t_end=10.0, dt=0.005)
        energies = np.array([total_energy(s, params) for s in states])
        drift = np.max(np.abs(energies - energies[0]))
        # DOP853 at 1e-14 tolerance should conserve to ~1e-10
        assert drift < 1e-6, f"Energy drift {drift} exceeds tolerance"


class TestPositions:
    """Test Cartesian coordinate conversion."""

    def test_straight_down(self):
        """Both bobs hanging straight down."""
        params = DoublePendulumParams(l1=1.0, l2=1.0)
        state = [0.0, 0.0, 0.0, 0.0]
        x1, y1, x2, y2 = positions(state, params)
        assert abs(x1) < 1e-10
        assert abs(y1 - (-1.0)) < 1e-10
        assert abs(x2) < 1e-10
        assert abs(y2 - (-2.0)) < 1e-10

    def test_horizontal(self):
        """Both bobs pointing right (pi/2)."""
        params = DoublePendulumParams(l1=1.0, l2=1.0)
        state = [math.pi / 2, math.pi / 2, 0.0, 0.0]
        x1, y1, x2, y2 = positions(state, params)
        assert abs(x1 - 1.0) < 1e-10
        assert abs(y1) < 1e-10
        assert abs(x2 - 2.0) < 1e-10
        assert abs(y2) < 1e-10
