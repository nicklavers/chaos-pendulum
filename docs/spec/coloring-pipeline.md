# Coloring Pipeline

Maps unwrapped angle values (theta1 or theta2) to ARGB32 pixel arrays.
File: `fractal/coloring.py` (~170 lines).

> Cross-ref: [data-shapes.md](data-shapes.md) for array shapes.
> Cross-ref: [ADR-0007](../adr/0007-dual-angle-snapshots.md) for why both angles are stored.

## Pipeline

```
snapshots[:, angle_index, :]    (N, n_samples) float32, unwrapped angle
    |                           angle_index: 0=theta1, 1=theta2 (default)
    v
interpolate_angle()             lerp between adjacent samples at time_index
    |
    v
angles (N,) float32             one angle per grid cell at current time
    |
    v
normalize: (angles % (2*pi)) / (2*pi)
    |
    v
LUT index: (normalized * LUT_SIZE).astype(int32) % LUT_SIZE
    |
    v
LUT[index]                      (N,) uint32 ARGB32 pixels
    |
    v
reshape to (resolution, resolution)
    |
    v
QImage(data, w, h, stride, Format_ARGB32)
```

## Key Functions

### `interpolate_angle(snapshots, time_index) -> (N,) float32`

Linear interpolation between adjacent samples. `time_index` is a float in
`[0, n_samples - 1]`. Vectorized NumPy lerp, < 0.1ms for 256×256.

Works on any `(N, n_samples)` 2D array. Callers pass the selected angle
slice (`snapshots[:, angle_index, :]`) for coloring, or a specific slice
for the inspect tool.

### `angle_to_argb(angles, lut, resolution) -> (res, res, 4) uint8`

Applies modulo normalization + LUT lookup. Produces ARGB32 pixel array.

### `build_hue_lut(size=4096) -> (size,) uint32`

Pre-computes the HSV hue-to-ARGB32 lookup table. 4096 entries avoids visible
color banding. 16 KB total, fits in L1 cache. Rebuilt on colormap change (~1ms).

### `numpy_to_qimage(argb) -> QImage`

Zero-copy QImage from ARGB array. Attaches a `_numpy_ref` attribute to the
QImage to prevent the backing NumPy array from being garbage-collected while
Qt is rendering.

## Colormap Support

Multiple colormaps available (HSV, twilight, viridis, magma, etc.). Each
generates its own LUT via `build_hue_lut()`. The canvas rebuilds its image
when the colormap changes.

## Frame Latency

Total scrub frame: interpolation (<0.1ms) + LUT lookup (~2–3ms) +
QImage rebuild (~5ms) = **< 10ms** (60fps capable).
