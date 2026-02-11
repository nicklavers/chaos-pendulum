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

from fractal.compute import BasinResult, BatchResult
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
        saddle_energy_val: float | None = None,
    ) -> BatchResult:
        """Simulate N trajectories, return angle snapshots + final velocities.

        Args:
            params: Physics parameters.
            initial_conditions: (N, 4) array [theta1, theta2, omega1, omega2].
            t_end: Simulation end time in seconds.
            dt: RK4 step size.
            n_samples: Number of snapshots to store.
            cancel_check: Optional callable returning True to abort.
            progress_callback: Optional callable(steps_done, total_steps).
            saddle_energy_val: If provided, freeze trajectories whose total
                energy drops below this value (basin mode early termination).

        Returns:
            BatchResult with snapshots (N, 2, n_samples) float32 and
            final_velocities (N, 2) float32.
        """
        return rk4_batch_with_snapshots(
            params, initial_conditions, t_end, dt, n_samples,
            cancel_check, progress_callback, saddle_energy_val,
        )

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

        Optimised for basin mode: no intermediate snapshots are stored,
        reducing memory and avoiding snapshot bookkeeping overhead.

        Args:
            params: Physics parameters.
            initial_conditions: (N, 4) array [theta1, theta2, omega1, omega2].
            t_end: Simulation end time in seconds.
            dt: RK4 step size.
            cancel_check: Optional callable returning True to abort.
            progress_callback: Optional callable(steps_done, total_steps).
            saddle_energy_val: If provided, freeze trajectories whose total
                energy drops below this value (basin mode early termination).

        Returns:
            BasinResult with final_state (N, 4) float32.
        """
        return rk4_basin_final_state(
            params, initial_conditions, t_end, dt,
            cancel_check, progress_callback, saddle_energy_val,
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

    # Linear viscous damping
    mu = params.friction
    alpha1 = alpha1 - mu * omega1
    alpha2 = alpha2 - mu * omega2

    # Build result as new array (no in-place mutation for JAX compat)
    result = np.empty_like(states)
    result[:, 0] = omega1
    result[:, 1] = omega2
    result[:, 2] = alpha1
    result[:, 3] = alpha2

    return result


# How often to check energy for early termination (every N steps)
_ENERGY_CHECK_INTERVAL = 50


def total_energy_batch(
    states: np.ndarray, params: DoublePendulumParams,
) -> np.ndarray:
    """Compute total energy for N trajectories simultaneously.

    Args:
        states: (N, 4) float64 array [theta1, theta2, omega1, omega2].
        params: Physics parameters.

    Returns:
        (N,) float64 array of total energies (T + V).
    """
    theta1 = states[:, 0]
    theta2 = states[:, 1]
    omega1 = states[:, 2]
    omega2 = states[:, 3]
    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

    kinetic = (
        0.5 * (m1 + m2) * l1**2 * omega1**2
        + 0.5 * m2 * l2**2 * omega2**2
        + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )
    potential = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
    return kinetic + potential


def rk4_single_trajectory(
    params: DoublePendulumParams,
    theta1: float,
    theta2: float,
    t_end: float,
    dt: float,
) -> np.ndarray:
    """Run RK4 for a single trajectory, returning every timestep.

    Used for on-demand trajectory visualization (inspect tool). No snapshot
    sampling, no freeze logic — just raw integration returning the full
    state at every step.

    Args:
        params: Physics parameters.
        theta1: Initial angle 1 (radians).
        theta2: Initial angle 2 (radians).
        t_end: Simulation end time.
        dt: Step size.

    Returns:
        (n_steps + 1, 4) float32 array [theta1, theta2, omega1, omega2].
        Row 0 is the initial state.
    """
    n_steps = int(t_end / dt)
    # Single trajectory as (1, 4) to reuse derivatives_batch
    states = np.array([[theta1, theta2, 0.0, 0.0]], dtype=np.float64)

    trajectory = np.empty((n_steps + 1, 4), dtype=np.float32)
    trajectory[0] = states[0].astype(np.float32)

    for step in range(n_steps):
        k1 = derivatives_batch(states, params)
        k2 = derivatives_batch(states + 0.5 * dt * k1, params)
        k3 = derivatives_batch(states + 0.5 * dt * k2, params)
        k4 = derivatives_batch(states + dt * k3, params)

        states = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[step + 1] = states[0].astype(np.float32)

    return trajectory


def rk4_batch_with_snapshots(
    params: DoublePendulumParams,
    initial_conditions: np.ndarray,
    t_end: float,
    dt: float,
    n_samples: int,
    cancel_check: callable | None = None,
    progress_callback: callable | None = None,
    saddle_energy_val: float | None = None,
) -> BatchResult:
    """Run vectorized RK4 integration with angle snapshot capture.

    Args:
        params: Physics parameters.
        initial_conditions: (N, 4) float32 array.
        t_end: End time.
        dt: Step size.
        n_samples: Number of evenly-spaced snapshots.
        cancel_check: If callable returns True, abort early.
        progress_callback: Called with (steps_done, total_steps).
        saddle_energy_val: If provided, freeze trajectories whose total
            energy drops below this threshold.

    Returns:
        BatchResult with snapshots (N, 2, n_samples) float32 and
        final_velocities (N, 2) float32.
        If cancelled, returns partial results (remaining samples are 0).
    """
    n_steps = int(t_end / dt)
    sample_every = max(1, n_steps // n_samples)

    n_trajectories = initial_conditions.shape[0]
    snapshots = np.zeros((n_trajectories, 2, n_samples), dtype=np.float32)

    states = initial_conditions.astype(np.float64)
    sample_idx = 0

    # Freeze tracking for early termination
    frozen = np.zeros(n_trajectories, dtype=bool)
    all_frozen = False

    for step in range(n_steps):
        # Cancellation check every 100 steps (~1s response time)
        if cancel_check is not None and step % 100 == 0:
            if cancel_check():
                break

        # Progress reporting
        if progress_callback is not None and step % 100 == 0:
            progress_callback(step, n_steps)

        # Early exit when all trajectories are frozen
        if all_frozen:
            break

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

        states_next = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Frozen trajectories keep their current state
        if np.any(frozen):
            states = np.where(frozen[:, np.newaxis], states, states_next)
        else:
            states = states_next

        # Energy-based freeze check
        if (
            saddle_energy_val is not None
            and step % _ENERGY_CHECK_INTERVAL == 0
            and not all_frozen
        ):
            energies = total_energy_batch(states, params)
            newly_frozen = (~frozen) & (energies < saddle_energy_val)
            if np.any(newly_frozen):
                # Fill remaining snapshot slots for newly frozen trajectories
                for s in range(sample_idx, n_samples):
                    snapshots[newly_frozen, 0, s] = (
                        states[newly_frozen, 0].astype(np.float32)
                    )
                    snapshots[newly_frozen, 1, s] = (
                        states[newly_frozen, 1].astype(np.float32)
                    )
                frozen = frozen | newly_frozen
                all_frozen = bool(np.all(frozen))

    # Capture final snapshot if needed
    if sample_idx < n_samples:
        snapshots[:, 0, sample_idx] = states[:, 0].astype(np.float32)
        snapshots[:, 1, sample_idx] = states[:, 1].astype(np.float32)

    # Build final velocities sidecar
    final_velocities = np.empty((n_trajectories, 2), dtype=np.float32)
    final_velocities[:, 0] = states[:, 2].astype(np.float32)
    final_velocities[:, 1] = states[:, 3].astype(np.float32)

    # Final progress
    if progress_callback is not None:
        progress_callback(n_steps, n_steps)

    return BatchResult(snapshots, final_velocities)


def rk4_basin_final_state(
    params: DoublePendulumParams,
    initial_conditions: np.ndarray,
    t_end: float,
    dt: float,
    cancel_check: callable | None = None,
    progress_callback: callable | None = None,
    saddle_energy_val: float | None = None,
) -> BasinResult:
    """Run vectorized RK4 and return only the final state (no snapshots).

    Optimised for basin mode where only the endpoint matters.

    Args:
        params: Physics parameters.
        initial_conditions: (N, 4) float32 array.
        t_end: End time.
        dt: Step size.
        cancel_check: If callable returns True, abort early.
        progress_callback: Called with (steps_done, total_steps).
        saddle_energy_val: If provided, freeze trajectories whose total
            energy drops below this threshold.

    Returns:
        BasinResult with final_state (N, 4) float32 and
        convergence_times (N,) float32.
    """
    n_steps = int(t_end / dt)
    n_trajectories = initial_conditions.shape[0]

    states = initial_conditions.astype(np.float64)

    # Freeze tracking for early termination
    frozen = np.zeros(n_trajectories, dtype=bool)
    all_frozen = False

    # Convergence time: default to t_end (ran to completion)
    convergence_times = np.full(n_trajectories, t_end, dtype=np.float32)

    for step in range(n_steps):
        # Cancellation check every 100 steps (~1s response time)
        if cancel_check is not None and step % 100 == 0:
            if cancel_check():
                break

        # Progress reporting
        if progress_callback is not None and step % 100 == 0:
            progress_callback(step, n_steps)

        # Early exit when all trajectories are frozen
        if all_frozen:
            break

        # RK4 step (no in-place mutation for JAX compat)
        k1 = derivatives_batch(states, params)
        k2 = derivatives_batch(states + 0.5 * dt * k1, params)
        k3 = derivatives_batch(states + 0.5 * dt * k2, params)
        k4 = derivatives_batch(states + dt * k3, params)

        states_next = states + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Frozen trajectories keep their current state
        if np.any(frozen):
            states = np.where(frozen[:, np.newaxis], states, states_next)
        else:
            states = states_next

        # Energy-based freeze check
        if (
            saddle_energy_val is not None
            and step % _ENERGY_CHECK_INTERVAL == 0
            and not all_frozen
        ):
            energies = total_energy_batch(states, params)
            newly_frozen = (~frozen) & (energies < saddle_energy_val)
            if np.any(newly_frozen):
                convergence_times[newly_frozen] = np.float32(step * dt)
                frozen = frozen | newly_frozen
                all_frozen = bool(np.all(frozen))

    # Final progress
    if progress_callback is not None:
        progress_callback(n_steps, n_steps)

    return BasinResult(states.astype(np.float32), convergence_times)
