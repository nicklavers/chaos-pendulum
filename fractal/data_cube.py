"""Data cube: precomputed 7D basin lookup for instant slider response.

The fractal is a function of 7 parameters: (theta1, theta2, m1, m2, l1, l2, mu).
This module stores a precomputed cube of basin results across a grid of the
5 physics parameters (m1, m2, l1, l2, mu) at a coarse theta resolution (64x64).

When a slider changes, the nearest grid point is looked up instantly (O(1))
and displayed as a preview while the exact computation runs in the background.

Storage format: compressed .npz with three arrays per "cube":
  - n1: (n_slices, R, R) int8 — winding number axis 1
  - n2: (n_slices, R, R) int8 — winding number axis 2
  - brightness: (n_slices, R, R) uint8 — pre-baked convergence brightness

The slice index is computed as a flat index into the Cartesian product of
the parameter grids, in C-order (last axis varies fastest):
  flat_idx = i_m1 * (n_m2 * n_l1 * n_l2 * n_mu)
           + i_m2 * (n_l1 * n_l2 * n_mu)
           + i_l1 * (n_l2 * n_mu)
           + i_l2 * n_mu
           + i_mu
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class CubeSlice(NamedTuple):
    """Single 2D slice of the data cube at a specific parameter combination.

    Attributes:
        n1: (R, R) int8 array of winding number axis 1.
        n2: (R, R) int8 array of winding number axis 2.
        brightness: (R, R) uint8 array of pre-baked brightness (0-255).
    """

    n1: np.ndarray
    n2: np.ndarray
    brightness: np.ndarray


@dataclass(frozen=True)
class ParamGridSpec:
    """Specification for the physics parameter grid.

    Each tuple contains the discrete values sampled for that parameter.
    The Cartesian product of all tuples defines the full grid.

    Attributes:
        m1_values: Mass of bob 1 grid points.
        m2_values: Mass of bob 2 grid points.
        l1_values: Length of arm 1 grid points.
        l2_values: Length of arm 2 grid points.
        friction_values: Damping coefficient grid points.
        theta_resolution: Side length of the theta grid (e.g. 64).
    """

    m1_values: tuple[float, ...]
    m2_values: tuple[float, ...]
    l1_values: tuple[float, ...]
    l2_values: tuple[float, ...]
    friction_values: tuple[float, ...]
    theta_resolution: int

    @property
    def n_slices(self) -> int:
        """Total number of parameter combinations in the grid."""
        return (
            len(self.m1_values)
            * len(self.m2_values)
            * len(self.l1_values)
            * len(self.l2_values)
            * len(self.friction_values)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the 5D parameter grid."""
        return (
            len(self.m1_values),
            len(self.m2_values),
            len(self.l1_values),
            len(self.l2_values),
            len(self.friction_values),
        )

    def flat_index(
        self,
        i_m1: int,
        i_m2: int,
        i_l1: int,
        i_l2: int,
        i_mu: int,
    ) -> int:
        """Convert 5D grid indices to a flat slice index (C-order)."""
        n_m2 = len(self.m2_values)
        n_l1 = len(self.l1_values)
        n_l2 = len(self.l2_values)
        n_mu = len(self.friction_values)
        return (
            i_m1 * (n_m2 * n_l1 * n_l2 * n_mu)
            + i_m2 * (n_l1 * n_l2 * n_mu)
            + i_l1 * (n_l2 * n_mu)
            + i_l2 * n_mu
            + i_mu
        )


@dataclass(frozen=True)
class CubeMetadata:
    """Metadata for a precomputed data cube file.

    Stored alongside the .npz as a JSON sidecar for human readability.

    Attributes:
        version: File format version string.
        grid: The parameter grid specification.
    """

    version: str
    grid: ParamGridSpec


# Default grid: 5x5x5x5x8 = 5,000 slices at 64x64 theta
DEFAULT_GRID = ParamGridSpec(
    m1_values=(0.5, 1.0, 2.0, 3.0, 5.0),
    m2_values=(0.5, 1.0, 1.4, 2.5, 5.0),
    l1_values=(0.3, 0.6, 1.0, 1.5, 3.0),
    l2_values=(0.1, 0.3, 0.6, 1.0, 2.0),
    friction_values=(0.1, 0.2, 0.38, 0.6, 1.0, 1.5, 2.5, 4.0),
    theta_resolution=64,
)


def _nearest_index(values: tuple[float, ...], target: float) -> int:
    """Find the index of the nearest value in a sorted tuple.

    Uses bisect for O(log n) lookup, then checks the two candidates
    bracketing the target.

    Args:
        values: Sorted tuple of grid values.
        target: The value to find the nearest match for.

    Returns:
        Index of the nearest value in the tuple.
    """
    n = len(values)
    if n == 0:
        raise ValueError("Empty values tuple")
    if n == 1:
        return 0

    pos = bisect_left(values, target)

    if pos == 0:
        return 0
    if pos >= n:
        return n - 1

    # Compare the two candidates
    before = values[pos - 1]
    after = values[pos]
    if abs(target - before) <= abs(target - after):
        return pos - 1
    return pos


class DataCubeIndex:
    """In-memory index for fast nearest-neighbor lookup into a data cube.

    Holds three 3D arrays (n1, n2, brightness) indexed by flat slice index,
    plus the grid spec for coordinate mapping.

    The cube is read-only after construction and never evicted.
    """

    def __init__(
        self,
        grid: ParamGridSpec,
        n1_data: np.ndarray,
        n2_data: np.ndarray,
        brightness_data: np.ndarray,
    ) -> None:
        """Initialize the cube index.

        Args:
            grid: Parameter grid specification.
            n1_data: (n_slices, R, R) int8 array.
            n2_data: (n_slices, R, R) int8 array.
            brightness_data: (n_slices, R, R) uint8 array.

        Raises:
            ValueError: If array shapes don't match the grid spec.
        """
        expected_shape = (grid.n_slices, grid.theta_resolution, grid.theta_resolution)
        for name, arr in [
            ("n1_data", n1_data),
            ("n2_data", n2_data),
            ("brightness_data", brightness_data),
        ]:
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {arr.shape} != expected {expected_shape}"
                )

        self._grid = grid
        self._n1 = n1_data
        self._n2 = n2_data
        self._brightness = brightness_data

    @property
    def grid(self) -> ParamGridSpec:
        """The parameter grid specification."""
        return self._grid

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self._n1.nbytes + self._n2.nbytes + self._brightness.nbytes

    def nearest_lookup(
        self,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        friction: float,
    ) -> CubeSlice:
        """Look up the nearest precomputed slice for given parameters.

        Performs 5 bisect operations on small arrays, then a single
        array index. Total time: O(1) for practical grid sizes.

        Args:
            m1: Mass of bob 1.
            m2: Mass of bob 2.
            l1: Length of arm 1.
            l2: Length of arm 2.
            friction: Damping coefficient.

        Returns:
            CubeSlice with the nearest precomputed data.
        """
        g = self._grid
        i_m1 = _nearest_index(g.m1_values, m1)
        i_m2 = _nearest_index(g.m2_values, m2)
        i_l1 = _nearest_index(g.l1_values, l1)
        i_l2 = _nearest_index(g.l2_values, l2)
        i_mu = _nearest_index(g.friction_values, friction)

        flat_idx = g.flat_index(i_m1, i_m2, i_l1, i_l2, i_mu)

        return CubeSlice(
            n1=self._n1[flat_idx],
            n2=self._n2[flat_idx],
            brightness=self._brightness[flat_idx],
        )

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> DataCubeIndex:
        """Load a data cube from a compressed .npz file.

        Expects the .npz to contain:
          - "n1": (n_slices, R, R) int8
          - "n2": (n_slices, R, R) int8
          - "brightness": (n_slices, R, R) uint8

        And a sibling JSON file with grid metadata:
          - "{stem}_meta.json"

        Args:
            npz_path: Path to the .npz file.

        Returns:
            DataCubeIndex instance.

        Raises:
            FileNotFoundError: If the file or its metadata is missing.
            ValueError: If the data is malformed.
        """
        npz_path = Path(npz_path)
        meta_path = npz_path.with_name(npz_path.stem + "_meta.json")

        if not npz_path.exists():
            raise FileNotFoundError(f"Cube data not found: {npz_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Cube metadata not found: {meta_path}")

        # Load metadata
        with open(meta_path) as f:
            meta_dict = json.load(f)

        grid_dict = meta_dict["grid"]
        grid = ParamGridSpec(
            m1_values=tuple(grid_dict["m1_values"]),
            m2_values=tuple(grid_dict["m2_values"]),
            l1_values=tuple(grid_dict["l1_values"]),
            l2_values=tuple(grid_dict["l2_values"]),
            friction_values=tuple(grid_dict["friction_values"]),
            theta_resolution=int(grid_dict["theta_resolution"]),
        )

        # Load arrays
        with np.load(str(npz_path)) as data:
            n1_data = data["n1"]
            n2_data = data["n2"]
            brightness_data = data["brightness"]

        logger.info(
            "Loaded data cube: %d slices at %dx%d (%.1f MB)",
            grid.n_slices,
            grid.theta_resolution,
            grid.theta_resolution,
            (n1_data.nbytes + n2_data.nbytes + brightness_data.nbytes) / 1e6,
        )

        return cls(grid, n1_data, n2_data, brightness_data)


def save_cube(
    npz_path: str | Path,
    grid: ParamGridSpec,
    n1_data: np.ndarray,
    n2_data: np.ndarray,
    brightness_data: np.ndarray,
    version: str = "1",
) -> None:
    """Save a data cube to compressed .npz and JSON metadata.

    Args:
        npz_path: Output path for the .npz file.
        grid: Parameter grid specification.
        n1_data: (n_slices, R, R) int8 array.
        n2_data: (n_slices, R, R) int8 array.
        brightness_data: (n_slices, R, R) uint8 array.
        version: Format version string.
    """
    npz_path = Path(npz_path)
    meta_path = npz_path.with_name(npz_path.stem + "_meta.json")

    # Ensure output directory exists
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Save compressed arrays
    np.savez_compressed(
        str(npz_path),
        n1=n1_data.astype(np.int8),
        n2=n2_data.astype(np.int8),
        brightness=brightness_data.astype(np.uint8),
    )

    # Save metadata JSON
    meta = {
        "version": version,
        "grid": {
            "m1_values": list(grid.m1_values),
            "m2_values": list(grid.m2_values),
            "l1_values": list(grid.l1_values),
            "l2_values": list(grid.l2_values),
            "friction_values": list(grid.friction_values),
            "theta_resolution": grid.theta_resolution,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved data cube to %s (%.1f MB)", npz_path, npz_path.stat().st_size / 1e6)
