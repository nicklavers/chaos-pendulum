"""Basin solver: DOP853 adaptive integrator for basin-mode computation.

Integrates each trajectory independently using SciPy's solve_ivp with
energy-based early termination via events. Returns only the final state
(no intermediate snapshots), since basin mode only needs winding numbers.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from simulation import DoublePendulumParams, derivatives, total_energy
from fractal.compute import BasinResult

logger = logging.getLogger(__name__)

# How often to check for cancellation (every N trajectories)
_CANCEL_CHECK_INTERVAL = 100

# Default tolerances — sufficient for winding number classification
_RTOL = 1e-8
_ATOL = 1e-8


def _make_energy_event(
    params: DoublePendulumParams,
    saddle_energy_val: float,
):
    """Create a terminal event function for energy-based early termination.

    The event fires (returns zero) when total energy drops below the
    saddle-point energy threshold, meaning the trajectory can never
    change basin.
    """

    def event(t, state):
        return total_energy(state, params) - saddle_energy_val

    event.terminal = True
    event.direction = -1  # only trigger on downward crossing
    return event


def simulate_basin_batch(
    params: DoublePendulumParams,
    initial_conditions: np.ndarray,
    t_end: float,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    saddle_energy_val: float | None = None,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> BasinResult:
    """Integrate N trajectories with DOP853 and return only final states.

    Each trajectory is integrated independently, allowing adaptive step
    sizing and per-trajectory early termination.

    Args:
        params: Double pendulum physical parameters (including friction).
        initial_conditions: (N, 4) float32 array [theta1, theta2, omega1, omega2].
        t_end: Maximum integration time.
        cancel_check: Optional callable returning True to abort.
        progress_callback: Optional callable(done, total) for UI updates.
        saddle_energy_val: Energy threshold for early termination.
            When provided and friction > 0, trajectories whose energy
            drops below this value are terminated early.
        rtol: Relative tolerance for DOP853.
        atol: Absolute tolerance for DOP853.

    Returns:
        BasinResult with final_state (N, 4) float32.
    """
    n_trajectories = initial_conditions.shape[0]
    final_state = np.empty((n_trajectories, 4), dtype=np.float32)

    # Build the derivatives function (closure over params)
    def rhs(t, y):
        return derivatives(t, y, params)

    # Build energy event if threshold provided and friction active
    events = None
    if saddle_energy_val is not None and params.friction > 0:
        events = _make_energy_event(params, saddle_energy_val)

    for i in range(n_trajectories):
        # Cancellation check
        if cancel_check is not None and i % _CANCEL_CHECK_INTERVAL == 0:
            if cancel_check():
                # Fill remaining with zeros and return partial result
                final_state[i:] = 0.0
                return BasinResult(final_state)

        # Progress reporting
        if progress_callback is not None and i % _CANCEL_CHECK_INTERVAL == 0:
            progress_callback(i, n_trajectories)

        y0 = initial_conditions[i].astype(np.float64)

        sol = solve_ivp(
            fun=rhs,
            t_span=(0.0, t_end),
            y0=y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            events=events,
            dense_output=False,
        )

        # Extract final state (last column of solution)
        if sol.success and sol.y.size > 0:
            final_state[i] = sol.y[:, -1].astype(np.float32)
        else:
            # Integration failed — use initial conditions as fallback
            logger.warning("solve_ivp failed for trajectory %d: %s", i, sol.message)
            final_state[i] = y0.astype(np.float32)

    # Final progress report
    if progress_callback is not None:
        progress_callback(n_trajectories, n_trajectories)

    return BasinResult(final_state)
