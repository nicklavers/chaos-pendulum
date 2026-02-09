# Fractal Compute

Compute backends and progressive rendering pipeline. Files: `fractal/compute.py`,
`fractal/_numpy_backend.py`, `fractal/_numba_backend.py`, `fractal/_jax_backend.py`.

> Cross-ref: [data-shapes.md](data-shapes.md) for array shapes and `FractalTask`.

## ComputeBackend Protocol

```python
class ComputeBackend(Protocol):
    def simulate_batch(
        self,
        params: DoublePendulumParams,
        initial_conditions: np.ndarray,  # (N, 4)
        t_end: float,
        dt: float,
        n_samples: int,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:  # (N, 2, n_samples) float32
        ...
```

Returns unwrapped `[theta1, theta2]` at `n_samples` uniformly spaced timesteps.

## Backend Selection

Auto-selected via `get_default_backend()` with try/except ImportError fallback:

| Priority | Backend | File | Dependency | Typical Speed (256×256) |
|----------|---------|------|------------|------------------------|
| 1 | JAX/Metal | `_jax_backend.py` | jax, jax-metal | < 1s (GPU) |
| 2 | Numba | `_numba_backend.py` | numba | 0.3–0.5s (CPU JIT) |
| 3 | NumPy | `_numpy_backend.py` | (none) | 10–15s (CPU) |

## RK4 Integration

All backends use fixed-step RK4 (not adaptive). Step size `dt = 0.01`
(3,000 steps for 30s simulation).

Fixed-step accuracy is sufficient for fractal visualization — sub-pixel
shifts in basin boundaries are invisible. The scalar DOP853 in
`simulation.py` is preserved for pendulum mode where energy conservation
matters.

## Vectorized Batch Processing

All trajectories advance through each timestep simultaneously as a single
`(N, 4)` array. This eliminates Python-level loops over trajectories.

The NumPy backend implements `derivatives_batch()` which duplicates the
physics from `simulation.py`'s scalar `derivatives()`. Both operate on the
same equations but different array shapes.

## Progressive Rendering Levels

Each level is a full vectorized batch at a given resolution, displayed
immediately via nearest-neighbor upscaling.

**Without Numba** (3 levels): 64×64 → 128×128 → 256×256
**With Numba** (2 levels): 128×128 → 256×256

The level list is parameterized by backend speed via `get_progressive_levels()`.

**Nearest-neighbor upscaling** (not bilinear) preserves the blocky pixel grid
that communicates "this is a preview."

## Cancellation

The RK4 loop checks a cancellation flag every ~100 steps (~1s response time).
If the user pans, zooms, or changes parameters during computation, the current
level is cancelled and the pipeline restarts.

## Grid Builder

`build_initial_conditions(viewport) -> (N, 4)` generates the initial condition
grid from a `FractalViewport`. All omega values are zero.

## JIT Warmup (Numba)

A tiny 2×2 dummy computation runs at app startup in a background QThread so
the first fractal render does not incur the 2–5s JIT compilation delay.

## JAX/Metal Backend (Phase 3+)

Placeholder. Will use `jax.vmap` for trajectory parallelism and `jax.lax.scan`
for the RK4 time loop. See [ADR-0001](../adr/0001-vectorized-rk4.md) for
design constraints ensuring JAX compatibility.

## Performance Estimates

| Resolution | Trajectories | NumPy | Numba | Storage (96 samples) |
|------------|-------------|-------|-------|---------------------|
| 64×64 | 4,096 | ~0.5s | ~0.02s | 1.5 MB |
| 128×128 | 16,384 | ~2–3s | ~0.08s | 6 MB |
| 256×256 | 65,536 | ~10–15s | ~0.3–0.5s | 24 MB |
| 512×512 | 262,144 | ~40–60s | ~1–2s | 96 MB |
