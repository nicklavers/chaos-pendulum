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

from simulation import DoublePendulumParams


@njit(cache=True)
def _derivatives_single(theta1, theta2, omega1, omega2,
                         m1, m2, l1, l2, g):
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

    return omega1, omega2, alpha1, alpha2


@njit(parallel=True, cache=True)
def _rk4_batch_numba(
    initial_conditions,  # (N, 4) float64
    m1, m2, l1, l2, g,
    n_steps,
    dt,
    sample_every,
    n_samples,
):
    """Numba-compiled parallel RK4 batch integration.

    Each trajectory is computed independently in parallel via prange.
    Returns (N, 2, n_samples) float32 of unwrapped [theta1, theta2].
    """
    n_traj = initial_conditions.shape[0]
    snapshots = np.zeros((n_traj, 2, n_samples), dtype=np.float32)

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
                theta1, theta2, omega1, omega2, m1, m2, l1, l2, g
            )

            ht = 0.5 * dt
            k2_t1, k2_t2, k2_o1, k2_o2 = _derivatives_single(
                theta1 + ht * k1_t1, theta2 + ht * k1_t2,
                omega1 + ht * k1_o1, omega2 + ht * k1_o2,
                m1, m2, l1, l2, g
            )

            k3_t1, k3_t2, k3_o1, k3_o2 = _derivatives_single(
                theta1 + ht * k2_t1, theta2 + ht * k2_t2,
                omega1 + ht * k2_o1, omega2 + ht * k2_o2,
                m1, m2, l1, l2, g
            )

            k4_t1, k4_t2, k4_o1, k4_o2 = _derivatives_single(
                theta1 + dt * k3_t1, theta2 + dt * k3_t2,
                omega1 + dt * k3_o1, omega2 + dt * k3_o2,
                m1, m2, l1, l2, g
            )

            dt6 = dt / 6.0
            theta1 = theta1 + dt6 * (k1_t1 + 2 * k2_t1 + 2 * k3_t1 + k4_t1)
            theta2 = theta2 + dt6 * (k1_t2 + 2 * k2_t2 + 2 * k3_t2 + k4_t2)
            omega1 = omega1 + dt6 * (k1_o1 + 2 * k2_o1 + 2 * k3_o1 + k4_o1)
            omega2 = omega2 + dt6 * (k1_o2 + 2 * k2_o2 + 2 * k3_o2 + k4_o2)

        # Final snapshot
        if sample_idx < n_samples:
            snapshots[i, 0, sample_idx] = np.float32(theta1)
            snapshots[i, 1, sample_idx] = np.float32(theta2)

    return snapshots


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
    ) -> np.ndarray:
        """Simulate N trajectories using Numba-parallelized RK4.

        Note: cancel_check is not supported inside the Numba kernel.
        Cancellation happens between progressive levels only.
        """
        n_steps = int(t_end / dt)
        sample_every = max(1, n_steps // n_samples)

        ics = initial_conditions.astype(np.float64)

        snapshots = _rk4_batch_numba(
            ics,
            params.m1, params.m2, params.l1, params.l2, params.g,
            n_steps, dt, sample_every, n_samples,
        )

        if progress_callback is not None:
            progress_callback(n_steps, n_steps)

        return snapshots

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
            1.0, 1.0, 1.0, 1.0, 9.81,
            100, 0.01, 10, 10,
        )
