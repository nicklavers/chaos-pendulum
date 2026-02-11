"""Winding number colormaps: map integer winding pairs (n1, n2) to BGRA color.

Each damped trajectory settles to a final state. The winding number is
the integer part of the total accumulated rotations:

    n = round(theta_final / (2 * pi))

The color domain is Z^2 — an infinite 2D integer lattice. Each colormap
assigns a distinct color to each (n1, n2) pair, maximizing visual
discrimination between adjacent basins.

Two colormaps are provided:
  - Modular Grid (5x5): repeating 25-color palette via modular arithmetic
  - Basin Hash: golden-ratio hash for pseudo-random RGB

All functions take (n1, n2) int32 arrays and return (N, 4) uint8 BGRA.

Brightness modulation: when convergence times are provided, pixel brightness
is scaled so that fast-converging trajectories (basin interiors) are bright
and slow-converging ones (basin boundaries) are dark.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from fractal.bivariate import _hsv_to_bgra


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_winding_numbers(
    theta1: np.ndarray, theta2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract integer winding numbers from unwrapped angle arrays.

    Args:
        theta1: (N,) float32/64 array of unwrapped theta1 values.
        theta2: (N,) float32/64 array of unwrapped theta2 values.

    Returns:
        (n1, n2): tuple of (N,) int32 arrays — full rotation counts.
    """
    two_pi = 2.0 * math.pi
    n1 = np.round(theta1 / two_pi).astype(np.int32)
    n2 = np.round(theta2 / two_pi).astype(np.int32)
    return n1, n2


# ---------------------------------------------------------------------------
# Colormap functions: (n1, n2) int32 arrays -> (N, 4) uint8 BGRA
# ---------------------------------------------------------------------------

# 5x5 modular grid palette (25 visually distinct colors, BGRA)
_MODULAR_PALETTE = np.array([
    [180,  50,  50, 255],  # (0,0) dark red
    [ 50, 180,  50, 255],  # (1,0) green
    [ 50,  50, 200, 255],  # (2,0) blue
    [ 50, 200, 200, 255],  # (3,0) yellow
    [200,  50, 200, 255],  # (4,0) magenta
    [200, 200,  50, 255],  # (0,1) cyan
    [100, 100, 220, 255],  # (1,1) salmon
    [ 50, 150, 100, 255],  # (2,1) dark green
    [180, 100,  50, 255],  # (3,1) teal
    [120,  50, 180, 255],  # (4,1) purple
    [ 80, 180, 220, 255],  # (0,2) orange
    [220, 220,  80, 255],  # (1,2) light cyan
    [140,  80, 140, 255],  # (2,2) mid purple
    [ 80, 220, 140, 255],  # (3,2) lime
    [220, 140,  80, 255],  # (4,2) brown
    [160, 160,  40, 255],  # (0,3) dark cyan
    [ 40, 120, 160, 255],  # (1,3) olive
    [200, 120, 200, 255],  # (2,3) pink
    [120, 200,  80, 255],  # (3,3) chartreuse
    [100,  80, 200, 255],  # (4,3) indigo
    [ 60,  60, 160, 255],  # (0,4) navy
    [160, 200, 160, 255],  # (1,4) sage
    [ 60, 160, 200, 255],  # (2,4) gold
    [200,  60,  60, 255],  # (3,4) blue-gray
    [160, 160, 160, 255],  # (4,4) gray
], dtype=np.uint8)


def winding_modular_grid(
    n1: np.ndarray, n2: np.ndarray,
) -> np.ndarray:
    """5x5 modular grid: (n1 mod 5, n2 mod 5) indexes a 25-color palette.

    Creates a repeating checkerboard pattern. Each 5x5 tile of basins
    gets a unique color assignment.

    Args:
        n1: (N,) int32 array of theta1 winding numbers.
        n2: (N,) int32 array of theta2 winding numbers.

    Returns:
        (N, 4) uint8 BGRA array.
    """
    idx = (n1 % 5) + 5 * (n2 % 5)
    # Ensure non-negative modular index
    idx = idx % 25
    return _MODULAR_PALETTE[idx]


# Golden ratio for hash-based coloring
_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def winding_basin_hash(
    n1: np.ndarray, n2: np.ndarray,
) -> np.ndarray:
    """Golden-ratio hash of (n1, n2) to pseudo-random RGB.

    Maximizes visual distinction between adjacent basins by using
    an irrational-number hash to spread nearby integer pairs across
    the full color space.

    Args:
        n1: (N,) int32 array of theta1 winding numbers.
        n2: (N,) int32 array of theta2 winding numbers.

    Returns:
        (N, 4) uint8 BGRA array.
    """
    n = n1.shape[0]
    n1f = n1.astype(np.float64)
    n2f = n2.astype(np.float64)

    # Hash each pair to a hue, saturation, and value
    hash_val = (n1f * _PHI + n2f * (_PHI * _PHI)) % 1.0
    h = (hash_val * 360.0).astype(np.float32)

    # Vary saturation slightly with a second hash
    hash2 = ((n1f * 0.7071 + n2f * 1.4142) % 1.0).astype(np.float32)
    s = np.float32(0.6) + np.float32(0.35) * hash2

    # Vary value with a third hash
    hash3 = ((n1f * 1.7321 + n2f * 0.5774) % 1.0).astype(np.float32)
    v = np.float32(0.5) + np.float32(0.45) * hash3

    # Origin basin gets a special dark color
    at_origin = (n1 == 0) & (n2 == 0)
    s = np.where(at_origin, np.float32(0.0), s)
    v = np.where(at_origin, np.float32(0.20), v)

    return _hsv_to_bgra(h, s, v)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

WINDING_COLORMAPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "Modular Grid (5\u00d75)": winding_modular_grid,
    "Basin Hash": winding_basin_hash,
}


# ---------------------------------------------------------------------------
# Single-point helpers
# ---------------------------------------------------------------------------

def get_single_winding_color(
    n1: int, n2: int,
    colormap_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[int, int, int, int]:
    """Get the BGRA color for a single (n1, n2) winding pair.

    Wraps the vectorized colormap call with length-1 arrays.

    Args:
        n1: Winding number for theta1.
        n2: Winding number for theta2.
        colormap_fn: Function (n1_int32, n2_int32) -> (N, 4) uint8 BGRA.

    Returns:
        (b, g, r, a) tuple of uint8 values.
    """
    n1_arr = np.array([n1], dtype=np.int32)
    n2_arr = np.array([n2], dtype=np.int32)
    bgra = colormap_fn(n1_arr, n2_arr)  # (1, 4)
    return (int(bgra[0, 0]), int(bgra[0, 1]), int(bgra[0, 2]), int(bgra[0, 3]))


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def winding_to_argb(
    theta1: np.ndarray,
    theta2: np.ndarray,
    colormap_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    resolution: int,
    convergence_times: np.ndarray | None = None,
) -> np.ndarray:
    """Map two unwrapped angle arrays to BGRA pixel array via winding colormaps.

    Extracts winding numbers from the final unwrapped angles, applies the
    colormap, and optionally modulates brightness by convergence time
    (fast convergence = bright, slow = dark).

    Args:
        theta1: (N,) float array of unwrapped theta1 values.
        theta2: (N,) float array of unwrapped theta2 values.
        colormap_fn: Function (n1_int32, n2_int32) -> (N, 4) uint8 BGRA.
        resolution: Grid side length (N = resolution^2).
        convergence_times: (N,) float32 array of convergence times.
            When provided, pixel brightness is modulated by how quickly
            each trajectory converged.

    Returns:
        (resolution, resolution, 4) uint8 BGRA array.
    """
    n1, n2 = extract_winding_numbers(theta1, theta2)
    pixels = colormap_fn(n1, n2)

    if convergence_times is not None:
        pixels = _apply_brightness_modulation(pixels, convergence_times)

    return pixels.reshape(resolution, resolution, 4)


def _apply_brightness_modulation(
    pixels: np.ndarray,
    convergence_times: np.ndarray,
) -> np.ndarray:
    """Scale pixel brightness by normalized convergence time.

    Fast-converging trajectories (basin interiors) are bright; slow-converging
    ones (basin boundaries) are dark. Brightness ranges from 0.3 to 1.0.

    Args:
        pixels: (N, 4) uint8 BGRA array (will be copied, not mutated).
        convergence_times: (N,) float32 convergence times.

    Returns:
        (N, 4) uint8 BGRA array with brightness-modulated BGR channels.
    """
    t_min = convergence_times.min()
    t_max = convergence_times.max()

    if t_max <= t_min:
        # All same convergence time — no modulation
        return pixels

    # Normalize to [0, 1]: 0 = fastest, 1 = slowest
    normalized = (convergence_times - t_min) / (t_max - t_min)

    # Fast = bright: fast → 1.0, slow → 0.3
    brightness = np.float32(1.0) - np.float32(0.7) * normalized

    # Apply to B, G, R channels (indices 0, 1, 2); leave A (index 3) alone
    result = pixels.copy()
    bgr = result[:, :3].astype(np.float32) * brightness[:, np.newaxis]
    result[:, :3] = bgr.astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Legend builder
# ---------------------------------------------------------------------------

def build_winding_legend(
    colormap_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_range: int = 3,
    cell_size: int = 6,
) -> np.ndarray:
    """Build a 2D grid legend showing winding number colors.

    Creates a (2*n_range+1) x (2*n_range+1) grid of cells, where each
    cell shows the color for the winding pair (n1, n2) with n1, n2 in
    [-n_range, +n_range].

    Args:
        colormap_fn: Winding colormap function.
        n_range: Range of winding numbers to show (default +-3).
        cell_size: Pixel size of each cell (default 6).

    Returns:
        (grid_px, grid_px, 4) uint8 BGRA array where grid_px =
        (2*n_range+1) * cell_size.
    """
    grid_n = 2 * n_range + 1
    ns = np.arange(-n_range, n_range + 1, dtype=np.int32)
    n1_grid, n2_grid = np.meshgrid(ns, ns)
    n1_flat = n1_grid.ravel()
    n2_flat = n2_grid.ravel()

    colors = colormap_fn(n1_flat, n2_flat)  # (grid_n^2, 4)
    color_grid = colors.reshape(grid_n, grid_n, 4)

    # Scale up by repeating each cell
    grid_px = grid_n * cell_size
    legend = np.zeros((grid_px, grid_px, 4), dtype=np.uint8)
    for row in range(grid_n):
        for col in range(grid_n):
            r0 = row * cell_size
            c0 = col * cell_size
            legend[r0:r0 + cell_size, c0:c0 + cell_size] = color_grid[row, col]

    return legend
