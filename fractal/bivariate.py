"""Bivariate torus colormaps: map (theta1, theta2) -> BGRA color.

Each torus colormap is a pure function mapping two angle arrays to a
BGRA pixel array. Both input angles are 2pi-periodic, so the color
domain is T^2 = [0, 2pi) x [0, 2pi) — a torus.

All functions are vectorized NumPy operations for performance
(< 10ms at 256x256 = 65536 pixels).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def _hsl_to_bgra(
    h: np.ndarray, s: np.ndarray, l: np.ndarray,
) -> np.ndarray:
    """Convert HSL arrays to BGRA uint8 array.

    Args:
        h: Hue in [0, 360), float32.
        s: Saturation in [0, 1], float32.
        l: Lightness in [0, 1], float32.

    Returns:
        (N, 4) uint8 BGRA array.
    """
    # Chroma
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    h_prime = h / 60.0
    x = c * (1.0 - np.abs(h_prime % 2.0 - 1.0))
    m = l - c / 2.0

    n = h.shape[0]
    r = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)

    sector = h_prime.astype(np.int32) % 6

    mask0 = sector == 0
    mask1 = sector == 1
    mask2 = sector == 2
    mask3 = sector == 3
    mask4 = sector == 4
    mask5 = sector == 5

    r[mask0] = c[mask0]; g[mask0] = x[mask0]
    r[mask1] = x[mask1]; g[mask1] = c[mask1]
    g[mask2] = c[mask2]; b[mask2] = x[mask2]
    g[mask3] = x[mask3]; b[mask3] = c[mask3]
    r[mask4] = x[mask4]; b[mask4] = c[mask4]
    r[mask5] = c[mask5]; b[mask5] = x[mask5]

    r = r + m
    g = g + m
    b = b + m

    bgra = np.empty((n, 4), dtype=np.uint8)
    bgra[:, 0] = np.clip(b * 255, 0, 255).astype(np.uint8)
    bgra[:, 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
    bgra[:, 2] = np.clip(r * 255, 0, 255).astype(np.uint8)
    bgra[:, 3] = 255

    return bgra


def _hsv_to_bgra(
    h: np.ndarray, s: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    """Convert HSV arrays to BGRA uint8 array.

    Args:
        h: Hue in [0, 360), float32.
        s: Saturation in [0, 1], float32.
        v: Value in [0, 1], float32.

    Returns:
        (N, 4) uint8 BGRA array.
    """
    c = v * s
    h_prime = h / 60.0
    x = c * (1.0 - np.abs(h_prime % 2.0 - 1.0))
    m = v - c

    n = h.shape[0]
    r = np.zeros(n, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)

    sector = h_prime.astype(np.int32) % 6

    mask0 = sector == 0
    mask1 = sector == 1
    mask2 = sector == 2
    mask3 = sector == 3
    mask4 = sector == 4
    mask5 = sector == 5

    r[mask0] = c[mask0]; g[mask0] = x[mask0]
    r[mask1] = x[mask1]; g[mask1] = c[mask1]
    g[mask2] = c[mask2]; b[mask2] = x[mask2]
    g[mask3] = x[mask3]; b[mask3] = c[mask3]
    r[mask4] = x[mask4]; b[mask4] = c[mask4]
    r[mask5] = c[mask5]; b[mask5] = x[mask5]

    r = r + m
    g = g + m
    b = b + m

    bgra = np.empty((n, 4), dtype=np.uint8)
    bgra[:, 0] = np.clip(b * 255, 0, 255).astype(np.uint8)
    bgra[:, 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
    bgra[:, 2] = np.clip(r * 255, 0, 255).astype(np.uint8)
    bgra[:, 3] = 255

    return bgra


# ---------------------------------------------------------------------------
# Torus colormap functions
# ---------------------------------------------------------------------------

def torus_hue_lightness(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Map (theta1, theta2) to color: theta1 -> hue, theta2 -> cyclic lightness.

    HSL with H = theta1 * (360 / 2pi), S = 0.85,
    L = 0.35 + 0.30 * (0.5 + 0.5 * cos(theta2)).
    Lightness cycles smoothly in [0.35, 0.65].

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    h = (theta1 % two_pi) / two_pi * 360.0
    s = np.full_like(h, 0.85)
    l = np.float32(0.35) + np.float32(0.30) * (
        np.float32(0.5) + np.float32(0.5) * np.cos(theta2 % two_pi)
    )
    return _hsl_to_bgra(h, s, l)


def torus_rgb_sinusoid(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Map (theta1, theta2) to color via phase-offset sinusoids.

    R = 128 + 127 * sin(theta1)
    G = 128 + 127 * sin(theta1 + 2pi/3 + theta2)
    B = 128 + 127 * sin(theta1 + 4pi/3 - theta2)

    Each channel is 2pi-periodic in both angles. The +/- theta2
    coupling creates diagonal color gradients.

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi
    offset = two_pi / np.float32(3.0)

    r = np.float32(128) + np.float32(127) * np.sin(t1)
    g = np.float32(128) + np.float32(127) * np.sin(t1 + offset + t2)
    b = np.float32(128) + np.float32(127) * np.sin(t1 + 2 * offset - t2)

    n = theta1.shape[0]
    bgra = np.empty((n, 4), dtype=np.uint8)
    bgra[:, 0] = np.clip(b, 0, 255).astype(np.uint8)
    bgra[:, 1] = np.clip(g, 0, 255).astype(np.uint8)
    bgra[:, 2] = np.clip(r, 0, 255).astype(np.uint8)
    bgra[:, 3] = 255

    return bgra


def torus_rgb_sinusoid_aligned(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """RGB sinusoid torus translated so landmarks align with state space.

    Same phase-offset sinusoid structure as the original, but shifted
    so that the four pure-color landmarks sit at physically meaningful
    positions:

        Red   (255,  0,  0) at (pi, pi) — both bobs horizontal
        Black (  0,  0,  0) at ( 0,  0) — both bobs hanging down
        Cyan  (  0,255,255) at ( 0, pi) — bob 1 down, bob 2 horizontal
        White (255,255,255) at (pi,  0) — bob 1 horizontal, bob 2 down

    Formulas (shifted by t1 - pi/2 and simplified):
        R = 128 + 127 * sin(t1 - pi/2)
        G = 128 + 127 * sin(t1 - pi/2 + t2)
        B = 128 + 127 * sin(t1 - pi/2 - t2)

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi
    phi = np.float32(math.pi / 2)

    r = np.float32(128) + np.float32(127) * np.sin(t1 - phi)
    g = np.float32(128) + np.float32(127) * np.sin(t1 - phi + t2)
    b = np.float32(128) + np.float32(127) * np.sin(t1 - phi - t2)

    n = theta1.shape[0]
    bgra = np.empty((n, 4), dtype=np.uint8)
    bgra[:, 0] = np.clip(b, 0, 255).astype(np.uint8)
    bgra[:, 1] = np.clip(g, 0, 255).astype(np.uint8)
    bgra[:, 2] = np.clip(r, 0, 255).astype(np.uint8)
    bgra[:, 3] = 255

    return bgra


def _aligned_base(
    t1: np.ndarray, t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the base aligned RGB channels (pre-clipping floats).

    Returns (r, g, b) float32 arrays, each in approximately [1, 255].
    Used by all aligned variants so the base formula lives in one place.
    """
    phi = np.float32(math.pi / 2)
    r = np.float32(128) + np.float32(127) * np.sin(t1 - phi)
    g = np.float32(128) + np.float32(127) * np.sin(t1 - phi + t2)
    b = np.float32(128) + np.float32(127) * np.sin(t1 - phi - t2)
    return r, g, b


def _pack_bgra(
    r: np.ndarray, g: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Round, clip R/G/B float arrays to [0,255] and pack into (N,4) uint8 BGRA.

    Rounding before clipping prevents float32 imprecision from shifting
    landmark colours by ±1 (e.g. 254.9999 → 255 instead of 254).
    """
    n = r.shape[0]
    bgra = np.empty((n, 4), dtype=np.uint8)
    bgra[:, 0] = np.clip(np.round(b), 0, 255).astype(np.uint8)
    bgra[:, 1] = np.clip(np.round(g), 0, 255).astype(np.uint8)
    bgra[:, 2] = np.clip(np.round(r), 0, 255).astype(np.uint8)
    bgra[:, 3] = 255
    return bgra


def torus_rgb_aligned_yb(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Aligned RGB sinusoid with yellow and blue at diagonal half-pi points.

    Adds a sin(t1)*sin(t2) correction to the R channel. This term is
    zero at all pi-multiples (preserving the 4 primary landmarks) but
    pushes the diagonal half-pi points to pure named colors:

        Yellow (255,255,  0) at (pi/2, pi/2) and (3pi/2, 3pi/2)
        Blue   (  0,  0,255) at (pi/2, 3pi/2) and (3pi/2, pi/2)

    Primary landmarks (unchanged):
        Red (pi,pi), Black (0,0), Cyan (0,pi), White (pi,0).

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi

    r, g, b = _aligned_base(t1, t2)
    r = r + np.float32(127) * np.sin(t1) * np.sin(t2)

    return _pack_bgra(r, g, b)


def torus_rgb_aligned_gm(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Aligned RGB sinusoid with green and magenta at diagonal half-pi points.

    Subtracts a sin(t1)*sin(t2) correction from the R channel (opposite
    sign from the yellow/blue variant):

        Green   (  0,255,  0) at (pi/2, pi/2) and (3pi/2, 3pi/2)
        Magenta (255,  0,255) at (pi/2, 3pi/2) and (3pi/2, pi/2)

    Primary landmarks (unchanged):
        Red (pi,pi), Black (0,0), Cyan (0,pi), White (pi,0).

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi

    r, g, b = _aligned_base(t1, t2)
    r = r - np.float32(127) * np.sin(t1) * np.sin(t2)

    return _pack_bgra(r, g, b)


def torus_rgb_aligned_ybgm(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Aligned RGB sinusoid with yellow, blue, green, magenta at four diagonal points.

    Adds a sin(t2) correction to the R channel. sin(t2) is zero at
    t2 = 0 and t2 = pi (preserving all 4 primary landmarks) but non-zero
    at the four diagonal half-pi points, pushing each to a distinct pure
    color:

        Yellow  (255,255,  1) at (pi/2, pi/2)
        Blue    (  1,  1,255) at (pi/2, 3pi/2)
        Magenta (255,  1,255) at (3pi/2, pi/2)
        Green   (  1,255,  1) at (3pi/2, 3pi/2)

    Primary landmarks (unchanged):
        Red (pi,pi), Black (0,0), Cyan (0,pi), White (pi,0).

    Unlike the YB and GM variants (which use sin(t1)*sin(t2) and
    therefore pair the diagonal corners), the sin(t2) term assigns a
    unique color to each of the four diagonal half-pi points. The
    remaining 8 "axis" half-pi points stay at neutral mid-gray (128,128,128).

    The visual character is intermediate between the YB/GM checkerboard
    and the 6-color fluidity: four cleanly separated color regions with
    strong landmark identity.

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi

    r, g, b = _aligned_base(t1, t2)
    r = r + np.float32(127) * np.sin(t2)

    return _pack_bgra(r, g, b)


def torus_rgb_aligned_6color(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Aligned RGB sinusoid with all 12 half-pi reference points as pure colors.

    Uses three independent cross-terms — sin(t1)*sin(t2), sin(t1)*cos(t2),
    and cos(t1)*sin(t2) — distributed across R, G, B channels. All three
    terms vanish at pi-multiples, preserving the 4 primary landmarks.

    The result places a pure named color at every multiple of pi/2:

        (  0,   0) BLACK   | (  0, pi/2) BLUE    | (  0, pi) CYAN    | (  0,3pi/2) GREEN
        (pi/2, 0) GREEN    | (pi/2,pi/2) YELLOW  | (pi/2,pi) MAGENTA | (pi/2,3pi/2) BLUE
        ( pi,  0) WHITE    | ( pi, pi/2) YELLOW  | ( pi, pi) RED     | ( pi,3pi/2) MAGENTA
        (3pi/2,0) MAGENTA  | (3pi/2,pi/2) BLUE   | (3pi/2,pi) GREEN  | (3pi/2,3pi/2) YELLOW

    Correction terms (added to the aligned base):
        R += 127 * ( sin(t1)*sin(t2) - sin(t1)*cos(t2) )
        G += 127 * ( sin(t1)*cos(t2) - cos(t1)*sin(t2) )
        B += 127 * (-sin(t1)*cos(t2) + cos(t1)*sin(t2) )

    Note: the G and B corrections simplify to ±sin(t1-t2):
        G += 127 * sin(t1 - t2)
        B -= 127 * sin(t1 - t2)

    This produces broad saturated regions (~40% of pixels hit a channel
    boundary), giving the map a vivid, poster-like quality.

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi

    r, g, b = _aligned_base(t1, t2)

    s1s2 = np.sin(t1) * np.sin(t2)
    s1c2 = np.sin(t1) * np.cos(t2)
    c1s2 = np.cos(t1) * np.sin(t2)

    r = r + np.float32(127) * (s1s2 - s1c2)
    g = g + np.float32(127) * (s1c2 - c1s2)
    b = b + np.float32(127) * (-s1c2 + c1s2)

    return _pack_bgra(r, g, b)


def torus_warm_cool(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Map (theta1, theta2) to color via warm/cool cyclic blending.

    blend = 0.5 + 0.5 * cos(theta1) cycles between warm and cool.
    theta2 modulates shade within each palette family.

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi

    blend = np.float32(0.5) + np.float32(0.5) * np.cos(t1)

    # Warm palette (reds/oranges)
    warm_r = np.float32(200) + np.float32(55) * np.cos(t2)
    warm_g = np.float32(100) + np.float32(80) * np.sin(t2)
    warm_b = np.float32(50) + np.float32(40) * np.cos(
        t2 + np.float32(math.pi / 3)
    )

    # Cool palette (blues/teals)
    cool_r = np.float32(40) + np.float32(40) * np.cos(
        t2 + np.float32(math.pi)
    )
    cool_g = np.float32(100) + np.float32(80) * np.sin(
        t2 + np.float32(math.pi / 2)
    )
    cool_b = np.float32(180) + np.float32(75) * np.cos(t2)

    r = blend * warm_r + (1.0 - blend) * cool_r
    g = blend * warm_g + (1.0 - blend) * cool_g
    b = blend * warm_b + (1.0 - blend) * cool_b

    n = theta1.shape[0]
    bgra = np.empty((n, 4), dtype=np.uint8)
    bgra[:, 0] = np.clip(b, 0, 255).astype(np.uint8)
    bgra[:, 1] = np.clip(g, 0, 255).astype(np.uint8)
    bgra[:, 2] = np.clip(r, 0, 255).astype(np.uint8)
    bgra[:, 3] = 255

    return bgra


def torus_diagonal_hue(
    theta1: np.ndarray, theta2: np.ndarray,
) -> np.ndarray:
    """Map (theta1, theta2) via diagonal hue + perpendicular value.

    H = (theta1 + theta2) mod 2pi -> hue (0-360)
    V = 0.55 + 0.45 * cos(theta1 - theta2) -> value
    S = 1.0

    Iso-hue lines run at 45 degrees; value varies perpendicular to them.

    Args:
        theta1: (N,) float32, wrapped to [0, 2pi).
        theta2: (N,) float32, wrapped to [0, 2pi).

    Returns:
        (N, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1 = theta1 % two_pi
    t2 = theta2 % two_pi

    h = ((t1 + t2) % two_pi) / two_pi * 360.0
    s = np.ones_like(h)
    v = np.float32(0.55) + np.float32(0.45) * np.cos(t1 - t2)

    return _hsv_to_bgra(h, s, v)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TORUS_COLORMAPS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "RGB Sinusoid Torus": torus_rgb_sinusoid,
    "RGB Sinusoid Aligned": torus_rgb_sinusoid_aligned,
    "RGB Aligned + Yellow/Blue": torus_rgb_aligned_yb,
    "RGB Aligned + Green/Magenta": torus_rgb_aligned_gm,
    "RGB Aligned + YBGM": torus_rgb_aligned_ybgm,
    "RGB Aligned + 6-Color": torus_rgb_aligned_6color,
    "Hue-Lightness Torus": torus_hue_lightness,
    "Warm-Cool Torus": torus_warm_cool,
    "Diagonal Hue Torus": torus_diagonal_hue,
}


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def bivariate_to_argb(
    theta1: np.ndarray,
    theta2: np.ndarray,
    colormap_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    resolution: int,
) -> np.ndarray:
    """Map two angle arrays to ARGB32 pixel array via a torus colormap.

    Args:
        theta1: (N,) float32 array of theta1 values (may be unwrapped).
        theta2: (N,) float32 array of theta2 values (may be unwrapped).
        colormap_fn: Function (theta1, theta2) -> (N, 4) uint8 BGRA.
        resolution: Grid side length (N = resolution^2).

    Returns:
        (resolution, resolution, 4) uint8 BGRA array.
    """
    two_pi = np.float32(2.0 * math.pi)
    t1_wrapped = theta1 % two_pi
    t2_wrapped = theta2 % two_pi
    pixels = colormap_fn(t1_wrapped, t2_wrapped)
    return pixels.reshape(resolution, resolution, 4)


# ---------------------------------------------------------------------------
# Legend builder
# ---------------------------------------------------------------------------

def build_torus_legend(
    colormap_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    size: int = 64,
) -> np.ndarray:
    """Build a 2D square legend image showing the torus colormap.

    Args:
        colormap_fn: Torus colormap function.
        size: Legend grid resolution (size x size pixels).

    Returns:
        (size, size, 4) uint8 BGRA array. Horizontal axis = theta1,
        vertical axis = theta2, both spanning [0, 2pi).
    """
    two_pi = 2.0 * math.pi
    t1 = np.linspace(0, two_pi, size, endpoint=False, dtype=np.float32)
    t2 = np.linspace(0, two_pi, size, endpoint=False, dtype=np.float32)
    t1_grid, t2_grid = np.meshgrid(t1, t2)
    t1_flat = t1_grid.ravel()
    t2_flat = t2_grid.ravel()
    pixels = colormap_fn(t1_flat, t2_flat)
    return pixels.reshape(size, size, 4)
