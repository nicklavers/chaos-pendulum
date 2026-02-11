"""Tests for fractal/compute.py: IC grid builder, viewport, backend selection."""

import math

import numpy as np
import pytest

from simulation import DoublePendulumParams
from fractal.compute import (
    FractalViewport, FractalTask, DEFAULT_N_SAMPLES, BatchResult, BasinResult,
    build_initial_conditions, get_default_backend, get_progressive_levels,
    saddle_energy,
)


class TestFractalViewport:
    """Test FractalViewport frozen dataclass."""

    def test_immutable(self):
        vp = FractalViewport(0.0, 0.0, 6.28, 6.28, 64)
        with pytest.raises(AttributeError):
            vp.resolution = 128

    def test_fields(self):
        vp = FractalViewport(1.0, 2.0, 3.0, 4.0, 256)
        assert vp.center_theta1 == 1.0
        assert vp.center_theta2 == 2.0
        assert vp.span_theta1 == 3.0
        assert vp.span_theta2 == 4.0
        assert vp.resolution == 256


class TestBuildInitialConditions:
    """Test IC grid construction."""

    def test_output_shape(self):
        vp = FractalViewport(0.0, 0.0, 2 * math.pi, 2 * math.pi, 8)
        ics = build_initial_conditions(vp)
        assert ics.shape == (64, 4)
        assert ics.dtype == np.float32

    def test_omega_zero(self):
        """All initial angular velocities should be zero."""
        vp = FractalViewport(0.0, 0.0, 2 * math.pi, 2 * math.pi, 4)
        ics = build_initial_conditions(vp)
        assert np.all(ics[:, 2] == 0.0)
        assert np.all(ics[:, 3] == 0.0)

    def test_theta_range(self):
        """Theta values should span the viewport range."""
        center = 1.0
        span = 2.0
        vp = FractalViewport(center, center, span, span, 10)
        ics = build_initial_conditions(vp)

        theta1_min = ics[:, 0].min()
        theta1_max = ics[:, 0].max()
        expected_min = center - span / 2
        expected_max = center + span / 2

        assert abs(theta1_min - expected_min) < 0.01
        assert abs(theta1_max - expected_max) < 0.01

    def test_resolution_1(self):
        """Resolution 1 should produce exactly 1 IC."""
        vp = FractalViewport(1.5, 2.5, 1.0, 1.0, 1)
        ics = build_initial_conditions(vp)
        assert ics.shape == (1, 4)
        # np.linspace(start, stop, 1) returns [start]
        assert abs(ics[0, 0] - 1.0) < 1e-5  # center - span/2
        assert abs(ics[0, 1] - 2.0) < 1e-5  # center - span/2


class TestGetDefaultBackend:
    """Test backend auto-selection."""

    def test_returns_a_backend(self):
        backend = get_default_backend()
        assert hasattr(backend, 'simulate_batch')

    def test_numpy_always_available(self):
        """NumPy backend should always be available as fallback."""
        from fractal._numpy_backend import NumpyBackend
        backend = get_default_backend()
        # Should at least be NumpyBackend (might be Numba if installed)
        assert hasattr(backend, 'simulate_batch')


class TestGetProgressiveLevels:
    """Test progressive level selection per backend."""

    def test_numpy_has_three_levels(self):
        from fractal._numpy_backend import NumpyBackend
        levels = get_progressive_levels(NumpyBackend())
        assert len(levels) >= 2
        assert 64 in levels
        assert 256 in levels


class TestFractalTask:
    """Test FractalTask immutable specification."""

    def test_immutable(self):
        params = DoublePendulumParams()
        vp = FractalViewport(0.0, 0.0, 6.28, 6.28, 64)
        task = FractalTask(params, vp, t_end=30.0, dt=0.01, n_samples=96)
        with pytest.raises(AttributeError):
            task.dt = 0.02

    def test_fields(self):
        params = DoublePendulumParams()
        vp = FractalViewport(0.0, 0.0, 6.28, 6.28, 64)
        task = FractalTask(params, vp, 30.0, 0.01, 96)
        assert task.t_end == 30.0
        assert task.dt == 0.01
        assert task.n_samples == 96

    def test_basin_default_false(self):
        """Basin field should default to False."""
        params = DoublePendulumParams()
        vp = FractalViewport(0.0, 0.0, 6.28, 6.28, 64)
        task = FractalTask(params, vp, 30.0, 0.01, 96)
        assert task.basin is False

    def test_basin_true(self):
        """Basin field can be set to True."""
        params = DoublePendulumParams()
        vp = FractalViewport(0.0, 0.0, 6.28, 6.28, 64)
        task = FractalTask(params, vp, 30.0, 0.01, 96, basin=True)
        assert task.basin is True

    def test_basin_immutable(self):
        """Basin field should be immutable."""
        params = DoublePendulumParams()
        vp = FractalViewport(0.0, 0.0, 6.28, 6.28, 64)
        task = FractalTask(params, vp, 30.0, 0.01, 96, basin=True)
        with pytest.raises(AttributeError):
            task.basin = False


class TestBatchResult:
    """Test BatchResult NamedTuple."""

    def test_field_access(self):
        """Should provide named field access."""
        snaps = np.zeros((4, 2, 10), dtype=np.float32)
        vels = np.zeros((4, 2), dtype=np.float32)
        result = BatchResult(snaps, vels)
        assert result.snapshots is snaps
        assert result.final_velocities is vels

    def test_tuple_unpacking(self):
        """Should support tuple unpacking."""
        snaps = np.ones((4, 2, 10), dtype=np.float32)
        vels = np.ones((4, 2), dtype=np.float32)
        result = BatchResult(snaps, vels)

        s, v = result
        assert np.array_equal(s, snaps)
        assert np.array_equal(v, vels)

    def test_immutable(self):
        """NamedTuple fields should be read-only."""
        snaps = np.zeros((4, 2, 10), dtype=np.float32)
        vels = np.zeros((4, 2), dtype=np.float32)
        result = BatchResult(snaps, vels)
        with pytest.raises(AttributeError):
            result.snapshots = np.zeros((4, 2, 10), dtype=np.float32)


class TestBasinResult:
    """Test BasinResult NamedTuple."""

    def test_field_access(self):
        """Should provide named field access."""
        state = np.zeros((16, 4), dtype=np.float32)
        conv = np.zeros(16, dtype=np.float32)
        result = BasinResult(state, conv)
        assert result.final_state is state
        assert result.convergence_times is conv

    def test_tuple_unpacking(self):
        """Should support tuple unpacking."""
        state = np.ones((16, 4), dtype=np.float32)
        conv = np.full(16, 5.0, dtype=np.float32)
        result = BasinResult(state, conv)
        s, c = result
        assert np.array_equal(s, state)
        assert np.array_equal(c, conv)

    def test_immutable(self):
        """NamedTuple fields should be read-only."""
        state = np.zeros((16, 4), dtype=np.float32)
        conv = np.zeros(16, dtype=np.float32)
        result = BasinResult(state, conv)
        with pytest.raises(AttributeError):
            result.final_state = np.zeros((16, 4), dtype=np.float32)


class TestSaddleEnergy:
    """Test saddle_energy() computation."""

    def test_default_params(self):
        """Default params: min(V(pi,0), V(0,pi)) should be -9.81."""
        params = DoublePendulumParams()
        se = saddle_energy(params)
        # V(pi,0) = (1+1)*9.81*1 - 1*9.81*1 = 9.81
        # V(0,pi) = -(1+1)*9.81*1 + 1*9.81*1 = -9.81
        assert se == pytest.approx(-9.81)

    def test_asymmetric_masses(self):
        """With m1=2, m2=1: V(pi,0)=(2+1)*g*1 - 1*g*1 = 2g, V(0,pi)=-2g."""
        params = DoublePendulumParams(m1=2.0, m2=1.0, g=10.0)
        se = saddle_energy(params)
        # V(pi,0) = 3*10*1 - 1*10*1 = 20
        # V(0,pi) = -3*10*1 + 1*10*1 = -20
        assert se == pytest.approx(-20.0)

    def test_pure_function(self):
        """Calling saddle_energy should not modify params."""
        params = DoublePendulumParams(m1=1.5, m2=0.8, l1=1.2, l2=0.9, g=9.81)
        se1 = saddle_energy(params)
        se2 = saddle_energy(params)
        assert se1 == se2
        assert params.m1 == 1.5
        assert params.m2 == 0.8

    def test_symmetric_params_zero(self):
        """When (m1+m2)*g*l1 == m2*g*l2, one saddle is at zero."""
        # (m1+m2)*g*l1 = m2*g*l2 => 2*g*1 = 1*g*2 => 2g = 2g
        params = DoublePendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=2.0, g=9.81)
        se = saddle_energy(params)
        # V(pi,0) = 2*9.81*1 - 1*9.81*2 = 0
        # V(0,pi) = -2*9.81*1 + 1*9.81*2 = 0
        assert se == pytest.approx(0.0)
