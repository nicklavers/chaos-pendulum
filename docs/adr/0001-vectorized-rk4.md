# ADR-0001: Vectorized Fixed-Step RK4 for Fractal Compute

**Status**: Accepted
**Date**: 2025-01

## Context

The fractal explorer needs to simulate 65,536+ trajectories (256×256 grid) to
produce a single frame. The existing `simulation.py` uses `scipy.integrate.solve_ivp`
with DOP853 (adaptive-step, per-trajectory) which is accurate but far too slow
for batch use.

## Decision

Use vectorized fixed-step RK4 where all trajectories advance through each
timestep simultaneously as a single `(N, 4)` NumPy array. Step size h = 0.01.

Preserve the existing `simulate()` in `simulation.py` unchanged for pendulum
mode, where single-trajectory adaptive accuracy matters.

## Alternatives Considered

| Approach | Time (256×256) | Drawback |
|----------|---------------|----------|
| Serial DOP853 | ~109 hours | Completely infeasible |
| Multiprocessing DOP853 (8 cores) | ~14 hours | Still infeasible |
| **Vectorized NumPy RK4** | **~10–15s** | Physics duplication |
| Vectorized + Numba JIT | ~0.3–0.5s | Optional dependency |

The speedup is structural (10,000×+), not incremental. NumPy's BLAS-backed
array operations on contiguous arrays exploit SIMD and cache locality.

## Consequences

- Physics equations are duplicated between `simulation.py` (scalar) and
  `_numpy_backend.py` (vectorized batch). Mitigated by cross-validation test.
- Fixed-step RK4 accuracy is sufficient for fractal visualization (sub-pixel
  basin boundary shifts are invisible), but not for energy conservation tracking.
- The `states = states + delta` pattern (new array, no in-place mutation) keeps
  the code JAX-compatible for future GPU acceleration.
