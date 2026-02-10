# Coloring Pipeline

Maps angle values to ARGB32 pixel arrays. Two paths: **univariate** (single
angle via 1D LUT) and **bivariate** (both angles via torus colormap function).

> Cross-ref: [data-shapes.md](data-shapes.md) for array shapes.
> Cross-ref: [ADR-0007](../adr/0007-dual-angle-snapshots.md) for why both angles are stored.
> Cross-ref: [ADR-0010](../adr/0010-bivariate-torus-colormaps.md) for torus colormap design.

## Univariate Path (single angle)

File: `fractal/coloring.py` (~170 lines).

Used when `angle_index` is 0 (theta1) or 1 (theta2).

```
snapshots[:, angle_index, :]    (N, n_samples) float32, unwrapped angle
    |
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
LUT[index]                      (N, 4) uint8 BGRA pixels
    |
    v
reshape to (resolution, resolution, 4)
    |
    v
QImage(data, w, h, stride, Format_ARGB32)
```

### Univariate Key Functions

#### `interpolate_angle(snapshots, time_index) -> (N,) float32`

Linear interpolation between adjacent samples. `time_index` is a float in
`[0, n_samples - 1]`. Vectorized NumPy lerp, < 0.1ms for 256x256.

Works on any `(N, n_samples)` 2D array. Callers pass the selected angle
slice (`snapshots[:, angle_index, :]`) for coloring, or a specific slice
for the inspect tool.

#### `angle_to_argb(angles, lut, resolution) -> (res, res, 4) uint8`

Applies modulo normalization + LUT lookup. Produces BGRA pixel array.

#### `build_hue_lut(size=4096) -> (size, 4) uint8`

Pre-computes the HSV hue-to-BGRA lookup table. 4096 entries avoids visible
color banding. 16 KB total, fits in L1 cache. Rebuilt on colormap change (~1ms).

#### `numpy_to_qimage(argb) -> QImage`

Zero-copy QImage from BGRA array. Attaches a `_numpy_ref` attribute to the
QImage to prevent the backing NumPy array from being garbage-collected while
Qt is rendering.

### Univariate Colormaps

Registry: `COLORMAPS: dict[str, Callable]` in `fractal/coloring.py`.

| Name | Description |
|------|-------------|
| HSV Hue Wheel | Full 360-degree hue cycle at full saturation |
| Twilight | Cool-warm-cool sinusoidal cycle |

Each entry maps a name to a LUT builder function `() -> (4096, 4) uint8`.

## Bivariate Path (both angles)

File: `fractal/bivariate.py` (~577 lines).

Used when `angle_index` is 2 (both). Maps `(theta1, theta2)` pairs directly
to BGRA pixels without a LUT.

```
snapshots[:, 0, :]   (N, n_samples) theta1
snapshots[:, 1, :]   (N, n_samples) theta2
    |                    |
    v                    v
interpolate_angle()  interpolate_angle()
    |                    |
    v                    v
theta1 (N,)          theta2 (N,)
    \                  /
     v                v
colormap_fn(theta1 % 2pi, theta2 % 2pi)
    |
    v
(N, 4) uint8 BGRA
    |
    v
reshape to (resolution, resolution, 4)
    |
    v
QImage
```

### Bivariate Key Functions

#### `bivariate_to_argb(theta1, theta2, colormap_fn, resolution) -> (res, res, 4) uint8`

Wraps both angles mod 2pi, calls the colormap function, reshapes to grid.

#### `build_torus_legend(colormap_fn, size=64) -> (size, size, 4) uint8`

Builds a 2D square legend image via meshgrid over [0, 2pi) x [0, 2pi).
Used for the torus legend overlay on the canvas.

### Torus Colormaps

Registry: `TORUS_COLORMAPS: dict[str, Callable]` in `fractal/bivariate.py`.

All functions have signature `(theta1: ndarray, theta2: ndarray) -> (N, 4) uint8 BGRA`.
Both inputs are 2pi-periodic (torus domain T^2).

| Name | Description | Landmark Colors |
|------|-------------|-----------------|
| RGB Sinusoid Torus | Phase-offset sinusoids: R=sin(t1), G=sin(t1+2pi/3+t2), B=sin(t1+4pi/3-t2) | Unaligned |
| RGB Sinusoid Aligned | Same formula shifted by t1-pi/2 to align landmarks to state space | Red(pi,pi), Black(0,0), Cyan(0,pi), White(pi,0) |
| RGB Aligned + Yellow/Blue | Aligned + R correction via sin(t1)*sin(t2) | +Yellow(pi/2,pi/2), Blue(pi/2,3pi/2) |
| RGB Aligned + Green/Magenta | Aligned - R correction via sin(t1)*sin(t2) | +Green(pi/2,pi/2), Magenta(pi/2,3pi/2) |
| RGB Aligned + YBGM | Aligned + R correction via sin(t2) — 4 distinct colors at diagonals | +Yellow(pi/2,pi/2), Blue(pi/2,3pi/2), Magenta(3pi/2,pi/2), Green(3pi/2,3pi/2) |
| RGB Aligned + 6-Color | Aligned + 3 cross-term corrections on all channels | All 16 pi/2-grid points are pure named colors |
| Hue-Lightness Torus | t1->hue (HSL), t2->cyclic lightness | — |
| Warm-Cool Torus | t1->warm/cool blend, t2->shade within palette | — |
| Diagonal Hue Torus | H=(t1+t2), V=0.55+0.45*cos(t1-t2) (HSV) | — |

### Aligned Variant Design

The six RGB-sinusoid-based colormaps share a base formula:

```
R = 128 + 127 * sin(t1 - pi/2)
G = 128 + 127 * sin(t1 - pi/2 + t2)
B = 128 + 127 * sin(t1 - pi/2 - t2)
```

Channel values range [1, 255] (not [0, 255]) because 128 + 127*(-1) = 1.

The half-pi variants add correction terms — `sin(t1)*sin(t2)` (YB, GM),
`sin(t2)` (YBGM), or products `sin(t1)*sin(t2)`, `sin(t1)*cos(t2)`,
`cos(t1)*sin(t2)` (6-Color). All terms vanish when both coordinates are
multiples of pi, preserving the 4 primary landmarks while controlling colors
at half-pi points.

### Helpers

- `_aligned_base(t1, t2)` — shared base formula returning (r, g, b) floats
- `_pack_bgra(r, g, b)` — round, clip to [0,255], pack into BGRA uint8
- `_hsl_to_bgra(h, s, l)` — HSL to BGRA conversion
- `_hsv_to_bgra(h, s, v)` — HSV to BGRA conversion

## Winding Number Path (basin mode)

File: `fractal/winding.py` (~273 lines).

Used in basin mode. The basin solver (`fractal/basin_solver.py`) returns
`BasinResult(final_state: (N, 4))` — only the final state of each trajectory.
The view extracts theta1/theta2 columns and passes them to the canvas, which
maps final unwrapped angles to integer winding numbers, then to BGRA pixels.

```
BasinResult.final_state[:, 0]   (N,) theta1 final
BasinResult.final_state[:, 1]   (N,) theta2 final
    |                              |
    v                              v
extract_winding_numbers()
    |                    |
    v                    v
n1 (N,) int32        n2 (N,) int32      round(theta / 2pi)
    \                  /
     v                v
colormap_fn(n1, n2)
    |
    v
(N, 4) uint8 BGRA
    |
    v
reshape to (resolution, resolution, 4)
    |
    v
QImage
```

### Winding Key Functions

#### `extract_winding_numbers(theta1, theta2) -> (n1, n2)` int32 arrays

Rounds each unwrapped angle to the nearest integer multiple of 2pi.

#### `winding_to_argb(theta1, theta2, colormap_fn, resolution) -> (res, res, 4) uint8`

Full pipeline: extract winding numbers, apply colormap, reshape to grid.

#### `build_winding_legend(colormap_fn, n_range, cell_size) -> ndarray`

Builds a 2D legend image showing the winding number grid for the legend overlay.

### Winding Colormaps

Registry: `WINDING_COLORMAPS: dict[str, Callable]` in `fractal/winding.py`.

All functions have signature `(n1: ndarray[int32], n2: ndarray[int32]) -> (N, 4) uint8 BGRA`.

| Name | Description |
|------|-------------|
| Direction + Brightness | Hue from atan2(n1,n2), brightness from distance to origin |
| Modular Grid (5x5) | 5x5 repeating color grid, each cell a distinct hue |
| Basin Hash | Hash-based coloring for maximum adjacent-basin contrast |

## Frame Latency

Univariate: interpolation (<0.1ms) + LUT lookup (~2-3ms) +
QImage rebuild (~5ms) = **< 10ms** (60fps capable).

Bivariate: interpolation x2 (<0.2ms) + colormap function (~3-5ms) +
QImage rebuild (~5ms) = **< 10ms** (60fps capable, verified by benchmark tests).

Winding: extraction (<0.1ms) + colormap (<1ms) + QImage rebuild (~5ms) =
**< 10ms**. No time interpolation needed (basin mode displays final state only).
