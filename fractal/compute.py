"""Fractal compute: ComputeBackend Protocol, IC grid builder, backend selection.

The ComputeBackend Protocol abstracts the batch simulation contract.
Three backends are auto-selected via try/except ImportError:
  JAX/Metal > Numba > NumPy
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from simulation import DoublePendulumParams

logger = logging.getLogger(__name__)

# Default number of angle snapshots to store per trajectory
DEFAULT_N_SAMPLES = 96


class BatchResult(NamedTuple):
    """Immutable result from a batch simulation.

    Supports tuple unpacking: ``snapshots, velocities = result``.
    """

    snapshots: np.ndarray        # (N, 2, n_samples) float32
    final_velocities: np.ndarray  # (N, 2) float32 [omega1, omega2]


@dataclass(frozen=True)
class FractalViewport:
    """Defines the visible region of the fractal in physics space."""

    center_theta1: float
    center_theta2: float
    span_theta1: float
    span_theta2: float
    resolution: int


@dataclass(frozen=True)
class FractalTask:
    """Immutable specification for a fractal compute job."""

    params: DoublePendulumParams
    viewport: FractalViewport
    t_end: float
    dt: float
    n_samples: int
    basin: bool = False


class ComputeBackend(Protocol):
    """Protocol for pluggable fractal compute backends."""

    def simulate_batch(
        self,
        params: DoublePendulumParams,
        initial_conditions: np.ndarray,  # (N, 4)
        t_end: float,
        dt: float,
        n_samples: int,
        cancel_check: callable | None = None,
        progress_callback: callable | None = None,
        saddle_energy_val: float | None = None,
    ) -> BatchResult:
        """Simulate N trajectories and return angle snapshots + final velocities.

        When saddle_energy_val is provided and friction > 0, trajectories whose
        total energy drops below this threshold are frozen early (their angles
        can never change basin).
        """
        ...


def build_initial_conditions(viewport: FractalViewport) -> np.ndarray:
    """Generate (N, 4) array of ICs from viewport. omega1=omega2=0.

    X-axis: theta1, Y-axis: theta2.
    Returns array with shape (resolution*resolution, 4).
    """
    res = viewport.resolution
    half_span1 = viewport.span_theta1 / 2
    half_span2 = viewport.span_theta2 / 2

    theta1_vals = np.linspace(
        viewport.center_theta1 - half_span1,
        viewport.center_theta1 + half_span1,
        res,
        dtype=np.float32,
    )
    theta2_vals = np.linspace(
        viewport.center_theta2 - half_span2,
        viewport.center_theta2 + half_span2,
        res,
        dtype=np.float32,
    )

    # meshgrid: theta1 varies along columns (x-axis), theta2 along rows (y-axis)
    t1_grid, t2_grid = np.meshgrid(theta1_vals, theta2_vals)

    n = res * res
    ics = np.zeros((n, 4), dtype=np.float32)
    ics[:, 0] = t1_grid.ravel()
    ics[:, 1] = t2_grid.ravel()
    # omega1, omega2 = 0 (already zeros)

    return ics


def saddle_energy(params: DoublePendulumParams) -> float:
    """Compute the lowest saddle-point potential energy for a double pendulum.

    Two candidate saddle configurations (at rest, omega=0):
      - theta1=pi, theta2=0: V = +(m1+m2)*g*l1 - m2*g*l2
      - theta1=0, theta2=pi: V = -(m1+m2)*g*l1 + m2*g*l2

    Returns the minimum, which is the energy barrier below which a
    trajectory can never change basin.
    """
    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g
    # V(a,b) = -(m1+m2)*g*l1*cos(a) - m2*g*l2*cos(b)
    # cos(pi) = -1, cos(0) = 1
    v_pi_0 = (m1 + m2) * g * l1 - m2 * g * l2
    v_0_pi = -(m1 + m2) * g * l1 + m2 * g * l2
    return min(v_pi_0, v_0_pi)


def get_default_backend() -> ComputeBackend:
    """Auto-select the best available compute backend.

    Priority: JAX/Metal > Numba > NumPy.
    """
    try:
        from fractal._jax_backend import JaxBackend
        logger.info("Using JAX/Metal compute backend")
        return JaxBackend()
    except ImportError:
        pass

    try:
        from fractal._numba_backend import NumbaBackend
        logger.info("Using Numba compute backend")
        return NumbaBackend()
    except ImportError:
        pass

    from fractal._numpy_backend import NumpyBackend
    logger.info("Using NumPy compute backend")
    return NumpyBackend()


def get_progressive_levels(backend: ComputeBackend) -> list[int]:
    """Return the list of progressive resolution levels for a backend.

    Faster backends use fewer (larger) preview levels.
    """
    backend_name = type(backend).__name__

    if backend_name == "JaxBackend":
        # GPU is fast enough for single-level at 256
        return [256]

    if backend_name == "NumbaBackend":
        return [128, 256]

    # NumPy fallback: 3-level progressive
    return [64, 128, 256]
