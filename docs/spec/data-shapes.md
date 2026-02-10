# Data Shapes

Core data structures, array shapes, and signal payloads shared across modules.

> **Canonical**: This is the single source of truth for all shared data types.
> Other specs reference this file rather than redefining shapes.

## Frozen Dataclasses

All config/state objects are frozen (immutable). Signals carry immutable payloads.
NumPy arrays passed between modules are treated as immutable — consumers never
modify them in place.

### `DoublePendulumParams` (simulation.py)

```python
@dataclass(frozen=True)
class DoublePendulumParams:
    m1: float = 1.0    # mass of bob 1
    m2: float = 1.0    # mass of bob 2
    l1: float = 1.0    # length of arm 1
    l2: float = 1.0    # length of arm 2
    g: float = 9.81    # gravitational acceleration
```

### `FractalViewport` (fractal/compute.py)

```python
@dataclass(frozen=True)
class FractalViewport:
    center_theta1: float
    center_theta2: float
    span_theta1: float
    span_theta2: float
    resolution: int
```

Zoom limits: max span 2π per axis, min span 0.001 radians.

### `FractalTask` (fractal/compute.py)

```python
@dataclass(frozen=True)
class FractalTask:
    params: DoublePendulumParams
    viewport: FractalViewport
    t_end: float
    dt: float
    n_samples: int
```

### `CacheKey` (fractal/cache.py)

```python
@dataclass(frozen=True)
class CacheKey:
    resolution: int
    center_theta1_q: int     # quantized to 1e-6 radian quantum
    center_theta2_q: int
    span_theta1_q: int
    span_theta2_q: int
    params_hash: int          # hash of (m1, m2, l1, l2, g)
```

Constructed via `CacheKey.from_viewport(viewport, params)`.

## Array Shapes

### Snapshot Array — `(N, 2, n_samples)` float32

The central data structure. Produced by compute backends, stored in cache,
consumed by canvas and inspect tool.

- **N** = `resolution²` (e.g. 65,536 for 256×256)
- **2** = `[theta1, theta2]` — index 0 is theta1, index 1 is theta2
- **n_samples** = `DEFAULT_N_SAMPLES` (96) — uniformly spaced timesteps
- **dtype** = float32
- **Values** = unwrapped angles (not modulo 2π). Modulo applied at color-mapping time.

Why unwrapped: preserves winding number, avoids interpolation artifacts at
the 0/2π boundary.

### Initial Conditions — `(N, 4)` float32

Input to compute backends.

- **4** = `[theta1, theta2, omega1, omega2]`
- omega1 = omega2 = 0 for the standard fractal grid
- Built by `build_initial_conditions(viewport)` in `fractal/compute.py`

### BGRA Pixel Array — `(resolution, resolution, 4)` uint8

Output of `angle_to_argb()` (univariate) or `bivariate_to_argb()` (bivariate).
Channel order in memory: BGRA (B=index 0, G=1, R=2, A=3). This is Qt's
ARGB32 format on little-endian systems. The alpha channel is always 255.

### HSV LUT — `(4096, 4)` uint8

Pre-computed hue-to-BGRA lookup table. 16 KB, fits in L1 cache.
Rebuilt on colormap change (~1ms). Used only in the univariate path.

### Torus Colormap Registry

`TORUS_COLORMAPS: dict[str, Callable]` in `fractal/bivariate.py`. Maps names
to functions with signature `(theta1: ndarray, theta2: ndarray) -> (N, 4) uint8`.
9 entries. Used only in the bivariate path.

## Signal Payloads

| Signal | Source | Payload |
|--------|--------|---------|
| `level_complete` | `FractalWorker` | `(int, np.ndarray)` — resolution, `(N, 2, n_samples)` |
| `progress` | `FractalWorker` | `(int, int)` — steps_done, total_steps |
| `viewport_changed` | `FractalCanvas` | `FractalViewport` |
| `ic_selected` | `FractalCanvas` | `(float, float)` — theta1, theta2 (Ctrl+click) |
| `hover_updated` | `FractalCanvas` | `(float, float)` — theta1, theta2 (inspect mode) |
| `time_index_changed` | `FractalControls` | `float` — slider position [0, n_samples-1] |
| `colormap_changed` | `FractalControls` | `str` — colormap name (univariate mode) |
| `torus_colormap_changed` | `FractalControls` | `str` — torus colormap name (bivariate mode) |
| `resolution_changed` | `FractalControls` | `int` — new resolution |
| `physics_changed` | `FractalControls` | (no payload) |
| `t_end_changed` | `FractalControls` | (no payload) |
| `tool_mode_changed` | `FractalControls` | `str` — "zoom", "pan", or "inspect" |
| `angle_selection_changed` | `FractalControls` | `int` — 0 (theta1), 1 (theta2), or 2 (both) |
| `zoom_out_clicked` | `FractalControls` | (no payload) |

## Constants

| Constant | Value | Location |
|----------|-------|----------|
| `DEFAULT_N_SAMPLES` | 96 | `fractal/compute.py` |
| `DEFAULT_DT` | 0.01 | `fractal/view.py` |
| `LUT_SIZE` | 4096 | `fractal/coloring.py` |
| `QUANTIZE_FACTOR` | 1e-6 | `fractal/cache.py` |
