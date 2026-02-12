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
@dataclass
class DoublePendulumParams:
    m1: float = 1.0       # mass of bob 1
    m2: float = 1.0       # mass of bob 2
    l1: float = 1.0       # length of arm 1
    l2: float = 1.0       # length of arm 2
    g: float = 9.81       # gravitational acceleration
    friction: float = 0.0 # linear viscous damping coefficient
```

The `friction` parameter applies linear viscous damping (`-friction * omega`) to both
angular accelerations. At `friction=0` the system is conservative (Hamiltonian).

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
    basin: bool = False   # basin mode: damped simulation, final-state display
```

When `basin=True`, the worker computes saddle energy and enables early termination.

### `BatchResult` (fractal/compute.py)

```python
class BatchResult(NamedTuple):
    snapshots: np.ndarray        # (N, 2, n_samples) float32
    final_velocities: np.ndarray # (N, 2) float32 [omega1, omega2]
```

Immutable result from angle-mode batch simulation. Supports tuple unpacking:
`snapshots, velocities = result`.

### `BasinResult` (fractal/compute.py)

```python
class BasinResult(NamedTuple):
    final_state: np.ndarray       # (N, 4) float32 [theta1, theta2, omega1, omega2]
    convergence_times: np.ndarray  # (N,) float32 — time to cross energy threshold
```

Immutable result from basin-mode simulation. Stores the final state of each
trajectory and the time at which it converged (crossed the saddle-energy
threshold). If no energy threshold is set, `convergence_times` values equal
`t_end`. Supports tuple unpacking: `final_state, conv_times = result`.

### `CacheKey` (fractal/cache.py)

```python
@dataclass(frozen=True)
class CacheKey:
    resolution: int
    center_theta1_q: int     # quantized to 1e-6 radian quantum
    center_theta2_q: int
    span_theta1_q: int
    span_theta2_q: int
    params_hash: int          # hash of (m1, m2, l1, l2, g, friction)
```

Constructed via `CacheKey.from_viewport(viewport, params)`. The `params_hash`
includes `friction` so that different damping coefficients produce different keys.

### `TrajectoryInfo` (fractal/animated_diagram.py)

```python
@dataclass(frozen=True)
class TrajectoryInfo:
    trajectory: np.ndarray          # (n_frames, 4) float32 subsampled state
    color_rgb: tuple[int, int, int] # basin color as (R, G, B)
```

Immutable per-trajectory data passed to `MultiTrajectoryDiagram.set_trajectories()`.
The trajectory is already subsampled (every `FRAME_SUBSAMPLE`th step).

### `PinnedTrajectory` (fractal/inspect_column.py)

```python
@dataclass(frozen=True)
class PinnedTrajectory:
    row_id: str
    theta1_init: float
    theta2_init: float
    trajectory: np.ndarray          # (n_frames, 4) float32 subsampled state
    n1: int                         # winding number for theta1
    n2: int                         # winding number for theta2
    color_rgb: tuple[int, int, int] # basin color as (R, G, B)
```

Immutable record for a pinned trajectory in the inspect column. Stored in
`InspectColumn._pinned` dict, rebuilt immutably on add/remove/recolor.

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
| `level_complete` | `FractalWorker` | `(int, np.ndarray, np.ndarray)` — resolution, data, extra. Angle mode: data is snapshots `(N, 2, n_samples)`, extra is final_velocities `(N, 2)`. Basin mode: data is final_state `(N, 4)`, extra is convergence_times `(N,)`. |
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
| `display_mode_changed` | `FractalControls` | `str` — "angle" or "basin" |
| `winding_colormap_changed` | `FractalControls` | `str` — winding colormap name (basin mode) |
| `pan_started` | `FractalCanvas` | (no payload) — emitted at start of pan drag |
| `trajectory_pinned` | `FractalCanvas` | `(str, float, float)` — row_id, theta1, theta2 (inspect click) |
| `row_removed` | `InspectColumn` | `str` — row_id of removed trajectory |
| `all_cleared` | `InspectColumn` | (no payload) |
| `indicator_hovered` | `InspectColumn` | `str` — row_id of hovered indicator |
| `indicator_unhovered` | `InspectColumn` | `str` — row_id of unhovered indicator |

### Winding Number Arrays — `(N,)` int32

Produced by `extract_winding_numbers_relative()` in `fractal/winding.py`.
Each value is `round(theta_final / 2π) - round(theta_init / 2π)` — the net
number of full rotations relative to the initial position. This relative
definition eliminates off-by-one errors near full-rotation boundaries.

The legacy `extract_winding_numbers()` (absolute: `round(theta / 2π)`) is
retained for backward compatibility in tests but is not used by the UI.
See [ADR-0014](../adr/0014-relative-winding-numbers.md).

## Key Functions

### `saddle_energy(params) -> float` (fractal/compute.py)

Computes the lowest saddle-point potential energy for the double pendulum.
Below this energy, a trajectory can never change basin (winding number).
Used by the worker to enable early termination in basin mode.

## Constants

| Constant | Value | Location |
|----------|-------|----------|
| `DEFAULT_N_SAMPLES` | 96 | `fractal/compute.py` |
| `DEFAULT_DT` | 0.01 | `fractal/view.py` |
| `LUT_SIZE` | 4096 | `fractal/coloring.py` |
| `QUANTIZE_FACTOR` | 1e-6 | `fractal/cache.py` |
| `ANIMATION_INTERVAL_MS` | 33 | `fractal/inspect_column.py` |
| `FRAME_SUBSAMPLE` | 6 | `fractal/inspect_column.py` |
| `PAUSE_FRAMES` | 30 | `fractal/animated_diagram.py` |
| `TRAIL_LENGTH` | 15 | `fractal/animated_diagram.py` |
| `SETTLE_BUFFER_SECONDS` | 5.0 | `fractal/animated_diagram.py` |
| `TAIL_WIDTH` | 6.5 | `fractal/arrow_arc.py` |
| `ARC_GAP_DEGREES` | 4.0 | `fractal/arrow_arc.py` |
| `SINGLE_GAP_DEGREES` | 40.0 | `fractal/arrow_arc.py` |
| `TAPER_STEPS` | 32 | `fractal/arrow_arc.py` |
