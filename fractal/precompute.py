"""Precompute pipeline: generate the data cube for instant slider response.

Standalone script (not part of the GUI) that iterates all parameter
combinations in a ParamGridSpec, runs basin simulations, and stores
the results as a compressed .npz file.

Usage:
    python -m fractal.precompute [--output data/cube_v1.npz] [--workers N]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import multiprocessing
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from fractal.compute import (
    BasinResult,
    FractalViewport,
    build_initial_conditions,
    saddle_energy,
)
from fractal.data_cube import DEFAULT_GRID, ParamGridSpec, save_cube
from fractal.winding import extract_winding_numbers_relative
from simulation import DoublePendulumParams

logger = logging.getLogger(__name__)

# RK4 step size (matches DEFAULT_DT in view.py)
_DT = 0.01


class SliceSpec(NamedTuple):
    """Specification for a single slice to compute."""

    flat_index: int
    m1: float
    m2: float
    l1: float
    l2: float
    friction: float
    theta_resolution: int


class SliceResult(NamedTuple):
    """Result from computing a single slice."""

    flat_index: int
    n1: np.ndarray       # (R, R) int8
    n2: np.ndarray       # (R, R) int8
    brightness: np.ndarray  # (R, R) uint8


def compute_slice(spec: SliceSpec) -> SliceResult:
    """Compute a single basin slice for the given parameters.

    This function is designed to run in a worker process. It imports
    the backend lazily to avoid issues with multiprocessing.

    Args:
        spec: Slice specification with physics parameters.

    Returns:
        SliceResult with winding numbers and brightness.
    """
    from fractal._numpy_backend import rk4_basin_final_state

    R = spec.theta_resolution
    params = DoublePendulumParams(
        m1=spec.m1,
        m2=spec.m2,
        l1=spec.l1,
        l2=spec.l2,
        friction=spec.friction,
    )

    # Build IC grid at full [0, 2pi]^2 viewport
    viewport = FractalViewport(
        center_theta1=math.pi,
        center_theta2=math.pi,
        span_theta1=2 * math.pi,
        span_theta2=2 * math.pi,
        resolution=R,
    )
    ics = build_initial_conditions(viewport)

    # Compute t_end from friction (capped at 100s for low-friction)
    mu = max(spec.friction, 0.01)
    t_end = min(5.0 / mu, 100.0)

    # Compute saddle energy for early termination
    saddle_val = saddle_energy(params)

    # Run basin simulation
    result: BasinResult = rk4_basin_final_state(
        params=params,
        initial_conditions=ics,
        t_end=t_end,
        dt=_DT,
        saddle_energy_val=saddle_val,
    )

    # Extract winding numbers (relative to initial conditions)
    theta1_init = ics[:, 0].astype(np.float32)
    theta2_init = ics[:, 1].astype(np.float32)
    theta1_final = result.final_state[:, 0].astype(np.float32)
    theta2_final = result.final_state[:, 1].astype(np.float32)

    n1, n2 = extract_winding_numbers_relative(
        theta1_final, theta2_final, theta1_init, theta2_init,
    )

    # Pre-bake brightness from convergence times
    brightness_flat = _bake_brightness(result.convergence_times)

    return SliceResult(
        flat_index=spec.flat_index,
        n1=n1.reshape(R, R).astype(np.int8),
        n2=n2.reshape(R, R).astype(np.int8),
        brightness=brightness_flat.reshape(R, R),
    )


def _bake_brightness(convergence_times: np.ndarray) -> np.ndarray:
    """Convert convergence times to pre-baked brightness values.

    Maps the 0.3-1.0 brightness range to uint8 0-255.
    Fast-converging trajectories are bright, slow ones are dark.

    Args:
        convergence_times: (N,) float32 array.

    Returns:
        (N,) uint8 array with brightness values.
    """
    t_min = convergence_times.min()
    t_max = convergence_times.max()

    if t_max <= t_min:
        # All same convergence time — full brightness
        return np.full(convergence_times.shape, 255, dtype=np.uint8)

    # Normalize to [0, 1]: 0 = fastest, 1 = slowest
    normalized = (convergence_times - t_min) / (t_max - t_min)

    # Fast = bright: fast → 1.0, slow → 0.3
    brightness_float = 1.0 - 0.7 * normalized

    # Map [0.3, 1.0] → [0, 255]
    brightness_uint8 = (brightness_float * 255.0).astype(np.uint8)

    return brightness_uint8


def build_slice_specs(grid: ParamGridSpec) -> list[SliceSpec]:
    """Build the list of all slice specifications from a grid.

    Args:
        grid: Parameter grid specification.

    Returns:
        List of SliceSpec in flat-index order.
    """
    specs: list[SliceSpec] = []
    R = grid.theta_resolution

    for i_m1, m1 in enumerate(grid.m1_values):
        for i_m2, m2 in enumerate(grid.m2_values):
            for i_l1, l1 in enumerate(grid.l1_values):
                for i_l2, l2 in enumerate(grid.l2_values):
                    for i_mu, mu in enumerate(grid.friction_values):
                        flat_idx = grid.flat_index(
                            i_m1, i_m2, i_l1, i_l2, i_mu,
                        )
                        specs.append(SliceSpec(
                            flat_index=flat_idx,
                            m1=m1,
                            m2=m2,
                            l1=l1,
                            l2=l2,
                            friction=mu,
                            theta_resolution=R,
                        ))

    return specs


def precompute_cube(
    grid: ParamGridSpec,
    output_path: str | Path,
    n_workers: int | None = None,
) -> None:
    """Run the full precomputation pipeline.

    Args:
        grid: Parameter grid specification.
        output_path: Path for the output .npz file.
        n_workers: Number of worker processes (default: CPU count).
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    R = grid.theta_resolution
    n_slices = grid.n_slices
    specs = build_slice_specs(grid)

    logger.info(
        "Precomputing %d slices at %dx%d using %d workers",
        n_slices, R, R, n_workers,
    )

    # Allocate output arrays
    n1_data = np.zeros((n_slices, R, R), dtype=np.int8)
    n2_data = np.zeros((n_slices, R, R), dtype=np.int8)
    brightness_data = np.zeros((n_slices, R, R), dtype=np.uint8)

    t0 = time.monotonic()

    if n_workers <= 1:
        # Sequential mode (useful for debugging)
        for i, spec in enumerate(specs):
            result = compute_slice(spec)
            n1_data[result.flat_index] = result.n1
            n2_data[result.flat_index] = result.n2
            brightness_data[result.flat_index] = result.brightness
            if (i + 1) % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    "  %d/%d slices (%.1f slices/s)",
                    i + 1, n_slices, rate,
                )
    else:
        # Parallel mode
        with multiprocessing.Pool(n_workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(compute_slice, specs, chunksize=8),
            ):
                n1_data[result.flat_index] = result.n1
                n2_data[result.flat_index] = result.n2
                brightness_data[result.flat_index] = result.brightness
                if (i + 1) % 100 == 0:
                    elapsed = time.monotonic() - t0
                    rate = (i + 1) / elapsed
                    logger.info(
                        "  %d/%d slices (%.1f slices/s)",
                        i + 1, n_slices, rate,
                    )

    elapsed = time.monotonic() - t0
    logger.info(
        "Precomputation complete: %d slices in %.1f s (%.1f slices/s)",
        n_slices, elapsed, n_slices / elapsed,
    )

    # Save to disk
    save_cube(output_path, grid, n1_data, n2_data, brightness_data)


def main() -> None:
    """CLI entry point for precomputation."""
    parser = argparse.ArgumentParser(
        description="Precompute the fractal data cube for instant slider response.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cube_v1.npz",
        help="Output path for the .npz file (default: data/cube_v1.npz)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    precompute_cube(DEFAULT_GRID, args.output, args.workers)


if __name__ == "__main__":
    main()
