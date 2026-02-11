"""Numba JIT-compiled backend for fractal computation.

Uses @njit(parallel=True) with prange for 10-50x speedup over NumPy.
This module is optional: if numba is not installed, get_default_backend()
falls back to the NumPy backend automatically.

IMPORTANT: The JIT-compiled functions use explicit loops (not NumPy
vectorization) since Numba compiles them to native machine code.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from fractal.compute import BasinResult, BatchResult
from simulation import DoublePendulumParams


@njit(cache=True)
def _derivatives_single(theta1, theta2, omega1, omega2,
                         m1, m2, l1, l2, g, friction):
    """Compute derivatives for a single trajectory (Numba-compiled).

    Returns (d_theta1, d_theta2, d_omega1, d_omega2).
    """
    delta = theta1 - theta2
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    denom = m1 + m2 - m2 * cos_delta * cos_delta

    alpha1 = (
        -m2 * l1 * omega1 * omega1 * sin_delta * cos_delta
        - m2 * l2 * omega2 * omega2 * sin_delta
        - (m1 + m2) * g * np.sin(theta1)
        + m2 * g * np.sin(theta2) * cos_delta
    ) / (l1 * denom)

    alpha2 = (
        (m1 + m2) * l1 * omega1 * omega1 * sin_delta
        + (m1 + m2) * g * np.sin(theta1) * cos_delta
        + m2 * l2 * omega2 * omega2 * sin_delta * cos_delta
        - (m1 + m2) * g * np.sin(theta2)
    ) / (l2 * denom)

    # Linear viscous damping
    alpha1 = alpha1 - friction * omega1
    alpha2 = alpha2 - friction * omega2

    return omega1, omega2, alpha1, alpha2


@njit(cache=True)
def _total_energy_single(theta1, theta2, omega1, omega2,
                          m1, m2, l1, l2, g):
    """Compute total energy for a single trajectory (Numba-compiled).

    Returns scalar T + V.
    """
    kinetic = (
        0.5 * (m1 + m2) * l1 * l1 * omega1 * omega1
        + 0.5 * m2 * l2 * l2 * omega2 * omega2
        + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )
    potential = (
        -(m1 + m2) * g * l1 * np.cos(theta1)
        - m2 * g * l2 * np.cos(theta2)
    )
    return kinetic + potential


@njit(parallel=True, cache=True)
def _rk4_batch_numba(
    initial_conditions,  # (N, 4) float64
    m1, m2, l1, l2, g, friction,
    n_steps,
    dt,
    sample_every,
    n_samples,
    saddle_energy_val,  # np.inf means disabled
):
    """Numba-compiled parallel RK4 batch integration.

    Each trajectory is computed independently in parallel via prange.
    Returns (snapshots, final_velocities) where:
      - snapshots: (N, 2, n_samples) float32 of unwrapped [theta1, theta2]
      - final_velocities: (N, 2) float32 of [omega1, omega2]

    When saddle_energy_val < inf, trajectories whose total energy drops
    below this threshold are frozen (remaining snapshots filled, loop exits).
    """
    n_traj = initial_conditions.shape[0]
    snapshots = np.zeros((n_traj, 2, n_samples), dtype=np.float32)
    final_velocities = np.zeros((n_traj, 2), dtype=np.float32)
    check_energy = saddle_energy_val < np.inf

    for i in prange(n_traj):
        theta1 = initial_conditions[i, 0]
        theta2 = initial_conditions[i, 1]
        omega1 = initial_conditions[i, 2]
        omega2 = initial_conditions[i, 3]

        sample_idx = 0

        for step in range(n_steps):
            # Capture snapshot (both theta1 and theta2)
            if step % sample_every == 0 and sample_idx < n_samples:
                snapshots[i, 0, sample_idx] = np.float32(theta1)
                snapshots[i, 1, sample_idx] = np.float32(theta2)
                sample_idx += 1

            # RK4 step
            k1_t1, k1_t2, k1_o1, k1_o2 = _derivatives_single(
                theta1, theta2, omega1, omega2,
                m1, m2, l1, l2, g, friction,
            )

            ht = 0.5 * dt
            k2_t1, k2_t2, k2_o1, k2_o2 = _derivatives_single(
                theta1 + ht * k1_t1, theta2 + ht * k1_t2,
                omega1 + ht * k1_o1, omega2 + ht * k1_o2,
                m1, m2, l1, l2, g, friction,
            )

            k3_t1, k3_t2, k3_o1, k3_o2 = _derivatives_single(
                theta1 + ht * k2_t1, theta2 + ht * k2_t2,
                omega1 + ht * k2_o1, omega2 + ht * k2_o2,
                m1, m2, l1, l2, g, friction,
            )

            k4_t1, k4_t2, k4_o1, k4_o2 = _derivatives_single(
                theta1 + dt * k3_t1, theta2 + dt * k3_t2,
                omega1 + dt * k3_o1, omega2 + dt * k3_o2,
                m1, m2, l1, l2, g, friction,
            )

            dt6 = dt / 6.0
            theta1 = theta1 + dt6 * (k1_t1 + 2 * k2_t1 + 2 * k3_t1 + k4_t1)
            theta2 = theta2 + dt6 * (k1_t2 + 2 * k2_t2 + 2 * k3_t2 + k4_t2)
            omega1 = omega1 + dt6 * (k1_o1 + 2 * k2_o1 + 2 * k3_o1 + k4_o1)
            omega2 = omega2 + dt6 * (k1_o2 + 2 * k2_o2 + 2 * k3_o2 + k4_o2)

            # Energy-based freeze check
            if check_energy:
                energy = _total_energy_single(
                    theta1, theta2, omega1, omega2,
                    m1, m2, l1, l2, g,
                )
                if energy < saddle_energy_val:
                    # Fill remaining snapshot slots and exit
                    for s in range(sample_idx, n_samples):
                        snapshots[i, 0, s] = np.float32(theta1)
                        snapshots[i, 1, s] = np.float32(theta2)
                    break

        # Final snapshot (normal completion, not frozen)
        if sample_idx < n_samples:
            snapshots[i, 0, sample_idx] = np.float32(theta1)
            snapshots[i, 1, sample_idx] = np.float32(theta2)

        # Record final velocities (both frozen and normal exit)
        final_velocities[i, 0] = np.float32(omega1)
        final_velocities[i, 1] = np.float32(omega2)

    return snapshots, final_velocities


@njit(parallel=True, cache=True)
def _rk4_basin_numba(
    initial_conditions,  # (N, 4) float64
    m1, m2, l1, l2, g, friction,
    n_steps,
    dt,
    saddle_energy_val,  # np.inf means disabled
):
    """Numba-compiled parallel RK4 for basin mode (final state only).

    Returns final_state (N, 4) float32.
    No intermediate snapshots are stored.
    """
    n_traj = initial_conditions.shape[0]
    final_state = np.zeros((n_traj, 4), dtype=np.float32)
    check_energy = saddle_energy_val < np.inf
    energy_check_interval = 50

    for i in prange(n_traj):
        theta1 = initial_conditions[i, 0]
        theta2 = initial_conditions[i, 1]
        omega1 = initial_conditions[i, 2]
        omega2 = initial_conditions[i, 3]

        for step in range(n_steps):
            # RK4 step
            k1_t1, k1_t2, k1_o1, k1_o2 = _derivatives_single(
                theta1, theta2, omega1, omega2,
                m1, m2, l1, l2, g, friction,
            )

            ht = 0.5 * dt
            k2_t1, k2_t2, k2_o1, k2_o2 = _derivatives_single(
                theta1 + ht * k1_t1, theta2 + ht * k1_t2,
                omega1 + ht * k1_o1, omega2 + ht * k1_o2,
                m1, m2, l1, l2, g, friction,
            )

            k3_t1, k3_t2, k3_o1, k3_o2 = _derivatives_single(
                theta1 + ht * k2_t1, theta2 + ht * k2_t2,
                omega1 + ht * k2_o1, omega2 + ht * k2_o2,
                m1, m2, l1, l2, g, friction,
            )

            k4_t1, k4_t2, k4_o1, k4_o2 = _derivatives_single(
                theta1 + dt * k3_t1, theta2 + dt * k3_t2,
                omega1 + dt * k3_o1, omega2 + dt * k3_o2,
                m1, m2, l1, l2, g, friction,
            )

            dt6 = dt / 6.0
            theta1 = theta1 + dt6 * (k1_t1 + 2 * k2_t1 + 2 * k3_t1 + k4_t1)
            theta2 = theta2 + dt6 * (k1_t2 + 2 * k2_t2 + 2 * k3_t2 + k4_t2)
            omega1 = omega1 + dt6 * (k1_o1 + 2 * k2_o1 + 2 * k3_o1 + k4_o1)
            omega2 = omega2 + dt6 * (k1_o2 + 2 * k2_o2 + 2 * k3_o2 + k4_o2)

            # Energy-based freeze check
            if check_energy and step % energy_check_interval == 0:
                energy = _total_energy_single(
                    theta1, theta2, omega1, omega2,
                    m1, m2, l1, l2, g,
                )
                if energy < saddle_energy_val:
                    break

        final_state[i, 0] = np.float32(theta1)
        final_state[i, 1] = np.float32(theta2)
        final_state[i, 2] = np.float32(omega1)
        final_state[i, 3] = np.float32(omega2)

    return final_state


class NumbaBackend:
    """Numba JIT-compiled compute backend.

    First call incurs JIT compilation overhead (~2-5s). Subsequent calls
    use the cached compiled version.
    """

    def simulate_batch(
        self,
        params: DoublePendulumParams,
        initial_conditions: np.ndarray,
        t_end: float,
        dt: float,
        n_samples: int,
        cancel_check: callable | None = None,
        progress_callback: callable | None = None,
        saddle_energy_val: float | None = None,
    ) -> BatchResult:
        """Simulate N trajectories using Numba-parallelized RK4.

        Note: cancel_check is not supported inside the Numba kernel.
        Cancellation happens between progressive levels only.
        """
        n_steps = int(t_end / dt)
        sample_every = max(1, n_steps // n_samples)

        ics = initial_conditions.astype(np.float64)
        se = np.inf if saddle_energy_val is None else float(saddle_energy_val)

        snapshots, final_velocities = _rk4_batch_numba(
            ics,
            params.m1, params.m2, params.l1, params.l2, params.g,
            params.friction,
            n_steps, dt, sample_every, n_samples,
            se,
        )

        if progress_callback is not None:
            progress_callback(n_steps, n_steps)

        return BatchResult(snapshots, final_velocities)

    def simulate_basin_batch(
        self,
        params: DoublePendulumParams,
        initial_conditions: np.ndarray,
        t_end: float,
        dt: float,
        cancel_check: callable | None = None,
        progress_callback: callable | None = None,
        saddle_energy_val: float | None = None,
    ) -> BasinResult:
        """Simulate N trajectories, return only the final state.

        Note: cancel_check is not supported inside the Numba kernel.
        Cancellation happens between progressive levels only.
        """
        n_steps = int(t_end / dt)
        ics = initial_conditions.astype(np.float64)
        se = np.inf if saddle_energy_val is None else float(saddle_energy_val)

        final_state = _rk4_basin_numba(
            ics,
            params.m1, params.m2, params.l1, params.l2, params.g,
            params.friction,
            n_steps, dt, se,
        )

        if progress_callback is not None:
            progress_callback(n_steps, n_steps)

        return BasinResult(final_state)

    @staticmethod
    def warmup() -> None:
        """Trigger JIT compilation with a tiny dummy grid.

        Call this at app startup in a background thread to avoid
        the 2-5s compilation delay on first fractal render.
        """
        dummy_ics = np.zeros((4, 4), dtype=np.float64)
        dummy_ics[:, 0] = [0.1, 0.2, 0.3, 0.4]
        dummy_ics[:, 1] = [0.1, 0.2, 0.3, 0.4]
        _rk4_batch_numba(
            dummy_ics,
            1.0, 1.0, 1.0, 1.0, 9.81, 0.0,
            100, 0.01, 10, 10,
            np.inf,
        )
        _rk4_basin_numba(
            dummy_ics,
            1.0, 1.0, 1.0, 1.0, 9.81, 0.0,
            100, 0.01,
            np.inf,
        )
