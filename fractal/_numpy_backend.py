"""NumPy vectorized RK4 backend for fractal computation.

All N trajectories advance through each timestep simultaneously as a
single (N, 4) NumPy array. This eliminates Python-level loops over
trajectories entirely, exploiting SIMD and cache locality.

IMPORTANT: No in-place mutation (uses states = states + delta, never +=).
This keeps the code JAX-compatible for future migration.

Physics equations here duplicate simulation.py's derivatives() but operate
on (N, 4) arrays. See test_numpy_backend.py for cross-validation tests.
"""

from __future__ import annotations

import numpy as np

from simulation import DoublePendulumParams


class NumpyBackend:
    """Pure NumPy vectorized RK4 compute backend."""

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
        """Simulate N trajectories, return angle snapshots.

        Args:
            params: Physics parameters.
            initial_conditions: (N, 4) array [theta1, theta2, omega1, omega2].
            t_end: Simulation end time in seconds.
            dt: RK4 step size.
            n_samples: Number of snapshots to store.
            cancel_check: Optional callable returning True to abort.
            progress_callback: Optional callable(steps_done, total_steps).

        Returns:
            (N, 2, n_samples) float32 array of unwrapped [theta1, theta2].
        """
        return rk4_batch_with_snapshots(
            params, initial_conditions, t_end, dt, n_samples,
            cancel_check, progress_callback,
        )


def derivatives_batch(states: np.ndarray, params: DoublePendulumParams) -> np.ndarray:
    """Compute derivatives for N trajectories simultaneously.

    Args:
        states: (N, 4) array with columns [theta1, theta2, omega1, omega2].
        params: Physics parameters.

    Returns:
        (N, 4) array of derivatives [d_theta1, d_theta2, d_omega1, d_omega2].

    Physics equations match simulation.py derivatives() exactly.
    See that file for the Lagrangian derivation.
    """
    theta1 = states[:, 0]
    theta2 = states[:, 1]
    omega1 = states[:, 2]
    omega2 = states[:, 3]

    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

    delta = theta1 - theta2
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    denom = m1 + m2 - m2 * cos_delta**2

    alpha1 = (
        -m2 * l1 * omega1**2 * sin_delta * cos_delta
        - m2 * l2 * omega2**2 * sin_delta
        - (m1 + m2) * g * np.sin(theta1)
        + m2 * g * np.sin(theta2) * cos_delta
    ) / (l1 * denom)

    alpha2 = (
        (m1 + m2) * l1 * omega1**2 * sin_delta
        + (m1 + m2) * g * np.sin(theta1) * cos_delta
        + m2 * l2 * omega2**2 * sin_delta * cos_delta
        - (m1 + m2) * g * np.sin(theta2)
    ) / (l2 * denom)

    # Build result as new array (no in-place mutation for JAX compat)
    result = np.empty_like(states)
    result[:, 0] = omega1
    result[:, 1] = omega2
    result[:, 2] = alpha1
    result[:, 3] = alpha2

    return result


def rk4_batch_with_snapshots(
    params: DoublePendulumParams,
    initial_conditions: np.ndarray,
    t_end: float,
    dt: float,
    n_samples: int,
    cancel_check: callable | None = None,
    progress_callback: callable | None = None,
) -> np.ndarray:
    """Run vectorized RK4 integration with angle snapshot capture.

    Args:
        params: Physics parameters.
        initial_conditions: (N, 4) float32 array.
        t_end: End time.
        dt: Step size.
        n_samples: Number of evenly-spaced snapshots.
        cancel_check: If callable returns True, abort early.
        progress_callback: Called with (steps_done, total_steps).

    Returns:
        (N, 2, n_samples) float32 array of unwrapped [theta1, theta2].
        If cancelled, returns partial results (remaining samples are 0).
    """
    n_steps = int(t_end / dt)
    sample_every = max(1, n_steps // n_samples)

    n_trajectories = initial_conditions.shape[0]
    snapshots = np.zeros((n_trajectories, 2, n_samples), dtype=np.float32)

    states = initial_conditions.astype(np.float64)
    sample_idx = 0

    for step in range(n_steps):
        # Cancellation check every 100 steps (~1s response time)
        if cancel_check is not None and step % 100 == 0:
            if cancel_check():
                return snapshots

        # Progress reporting
        if progress_callback is not None and step % 100 == 0:
            progress_callback(step, n_steps)

        # Capture snapshot (both theta1 and theta2)
        if step % sample_every == 0 and sample_idx < n_samples:
            snapshots[:, 0, sample_idx] = states[:, 0].astype(np.float32)
            snapshots[:, 1, sample_idx] = states[:, 1].astype(np.float32)
            sample_idx = sample_idx + 1

        # RK4 step (no in-place mutation for JAX compat)
        k1 = derivatives_batch(states, params)
        k2 = derivatives_batch(states + 0.5 * dt * k1, params)
        k3 = derivatives_batch(states + 0.5 * dt * k2, params)
        k4 = derivatives_batch(states + dt * k3, params)

        states = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Capture final snapshot if needed
    if sample_idx < n_samples:
        snapshots[:, 0, sample_idx] = states[:, 0].astype(np.float32)
        snapshots[:, 1, sample_idx] = states[:, 1].astype(np.float32)

    # Final progress
    if progress_callback is not None:
        progress_callback(n_steps, n_steps)

    return snapshots
