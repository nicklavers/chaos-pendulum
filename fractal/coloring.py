"""Color mapping pipeline: HSV LUT, angle to ARGB, QImage construction.

Converts raw angle values (theta1 or theta2) into displayable QImage pixels
via a pre-computed 4096-entry lookup table. The LUT approach makes
time-slider scrubbing instantaneous (< 10ms per frame at 256x256).
"""

import math

import numpy as np
from PyQt6.QtGui import QImage


# Default LUT size. 4096 avoids visible banding; 16 KB fits in L1 cache.
DEFAULT_LUT_SIZE = 4096


def build_hue_lut(size: int = DEFAULT_LUT_SIZE) -> np.ndarray:
    """Build a 360-degree HSV hue wheel lookup table.

    Returns:
        (size, 4) uint8 array in BGRA order (Qt's ARGB32 byte layout on
        little-endian systems is actually BGRA in memory).
    """
    hues = np.linspace(0, 360, size, endpoint=False, dtype=np.float32)

    # HSV to RGB at full saturation and value
    h_sector = hues / 60.0
    sector = h_sector.astype(np.int32) % 6
    f = h_sector - np.floor(h_sector)

    # p = 0 (saturation = 1), q = 1-f, t = f (value = 1)
    v = np.full(size, 255, dtype=np.uint8)
    p = np.zeros(size, dtype=np.uint8)
    q = ((1.0 - f) * 255).astype(np.uint8)
    t = (f * 255).astype(np.uint8)

    r = np.empty(size, dtype=np.uint8)
    g = np.empty(size, dtype=np.uint8)
    b = np.empty(size, dtype=np.uint8)

    mask0 = sector == 0
    mask1 = sector == 1
    mask2 = sector == 2
    mask3 = sector == 3
    mask4 = sector == 4
    mask5 = sector == 5

    r[mask0] = v[mask0]; g[mask0] = t[mask0]; b[mask0] = p[mask0]
    r[mask1] = q[mask1]; g[mask1] = v[mask1]; b[mask1] = p[mask1]
    r[mask2] = p[mask2]; g[mask2] = v[mask2]; b[mask2] = t[mask2]
    r[mask3] = p[mask3]; g[mask3] = q[mask3]; b[mask3] = v[mask3]
    r[mask4] = t[mask4]; g[mask4] = p[mask4]; b[mask4] = v[mask4]
    r[mask5] = v[mask5]; g[mask5] = p[mask5]; b[mask5] = q[mask5]

    # BGRA layout for QImage Format_ARGB32 on little-endian
    lut = np.empty((size, 4), dtype=np.uint8)
    lut[:, 0] = b
    lut[:, 1] = g
    lut[:, 2] = r
    lut[:, 3] = 255  # alpha

    return lut


def build_twilight_lut(size: int = DEFAULT_LUT_SIZE) -> np.ndarray:
    """Build a twilight-inspired colormap LUT (cool-warm-cool cycle).

    Returns:
        (size, 4) uint8 BGRA array.
    """
    t = np.linspace(0, 1, size, dtype=np.float32)
    # Simple twilight: dark blue -> pink -> white -> pink -> dark blue
    r = (128 + 127 * np.sin(2 * math.pi * t)).astype(np.uint8)
    g = (80 + 80 * np.sin(2 * math.pi * t + math.pi / 2)).astype(np.uint8)
    b = (180 + 75 * np.sin(2 * math.pi * t + math.pi)).astype(np.uint8)

    lut = np.empty((size, 4), dtype=np.uint8)
    lut[:, 0] = b
    lut[:, 1] = g
    lut[:, 2] = r
    lut[:, 3] = 255

    return lut


# Available colormaps
COLORMAPS = {
    "HSV Hue Wheel": build_hue_lut,
    "Twilight": build_twilight_lut,
}


def angle_to_argb(
    angles: np.ndarray,
    lut: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Map angle values to ARGB32 pixel array via LUT.

    Args:
        angles: (N,) float32 array of unwrapped angle values.
        lut: (lut_size, 4) uint8 BGRA lookup table.
        resolution: Grid side length (image is resolution x resolution).

    Returns:
        (resolution, resolution, 4) uint8 BGRA array.
    """
    lut_size = lut.shape[0]
    two_pi = 2.0 * math.pi

    # Normalize: wrap to [0, 2*pi) then map to [0, 1)
    normalized = (angles % two_pi) / two_pi

    # LUT index
    indices = (normalized * lut_size).astype(np.int32) % lut_size

    # Look up colors
    pixels = lut[indices]

    return pixels.reshape(resolution, resolution, 4)


def interpolate_angle(
    snapshots: np.ndarray,
    time_index: float,
) -> np.ndarray:
    """Interpolate angle values between adjacent snapshots.

    Args:
        snapshots: (N, n_samples) float32 array.
        time_index: Float index into the sample dimension.
            Integer values select exact samples; fractional values
            interpolate linearly between adjacent samples.

    Returns:
        (N,) float32 array of interpolated angle values.
    """
    n_samples = snapshots.shape[1]
    idx = max(0.0, min(float(time_index), n_samples - 1.0))

    lo = int(idx)
    hi = min(lo + 1, n_samples - 1)

    if lo == hi:
        return snapshots[:, lo]

    frac = np.float32(idx - lo)
    # Linear interpolation (works correctly on unwrapped angles)
    return snapshots[:, lo] * (1.0 - frac) + snapshots[:, hi] * frac


def numpy_to_qimage(argb: np.ndarray) -> QImage:
    """Create a QImage from an ARGB32 pixel array with GC safety.

    Args:
        argb: (H, W, 4) uint8 BGRA array (contiguous).

    Returns:
        QImage with Format_ARGB32. The numpy array is attached to the
        QImage as _numpy_ref to prevent garbage collection.
    """
    h, w = argb.shape[:2]
    # Ensure contiguous
    data = np.ascontiguousarray(argb)
    stride = 4 * w
    image = QImage(data.data, w, h, stride, QImage.Format.Format_ARGB32)
    # Prevent GC of the numpy array while QImage is alive
    image._numpy_ref = data
    return image
