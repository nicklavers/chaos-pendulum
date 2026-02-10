# ADR 0012: DOP853 Adaptive Solver for Basin Mode (Final State Only)

**Status**: Accepted
**Date**: 2026-02
**Supersedes**: ADR-0011

## Context

Basin mode was using the same fixed-step vectorized RK4 integrator as angle
mode (ADR-0001, ADR-0011). This had two inefficiencies:

1. **Synchronized dt**: all trajectories advanced in lockstep at dt=0.01,
   even though basin mode only needs the final state. Trajectories that settle
   quickly still consume the same number of steps as chaotic ones.
2. **Full trajectory storage**: the `(N, 2, n_samples)` snapshot array was
   allocated and filled for every trajectory, even though basin mode only reads
   the final snapshot for winding number computation. This wastes ~12x memory.

ADR-0011 introduced energy-based freeze logic within the RK4 loop, but frozen
trajectories still occupied snapshot memory and the vectorized loop still
iterated over them (skipping their state update but not eliminating overhead).

## Decision

Replace the RK4 batch integrator with **per-trajectory adaptive DOP853**
(`scipy.integrate.solve_ivp`) for basin mode. Store only the final state
`(N, 4)` as a new `BasinResult` type.

### Key design choices

- **Separate solver module** (`fractal/basin_solver.py`): keeps basin logic out
  of the RK4 backends, which remain unchanged for angle mode.
- **Separate result type** (`BasinResult` vs `BatchResult`): explicit contracts
  prevent confusion between the `(N, 4)` final state and the `(N, 2, n_samples)`
  snapshot array.
- **Energy termination via solve_ivp events**: a terminal event fires when total
  energy drops below the saddle threshold, stopping integration immediately for
  that trajectory. Cleaner than the manual freeze-mask approach in ADR-0011.
- **Per-trajectory loop**: `solve_ivp` is scalar, so basin mode loops over N
  trajectories in Python. This is slower than vectorized RK4 for large grids,
  but the adaptive stepper compensates by taking far fewer steps per trajectory.
- **Tolerances**: `rtol=1e-8, atol=1e-8` (sufficient for winding number
  classification; tighter than needed but cheap given adaptive stepping).

### What changed from ADR-0011

- ADR-0011's RK4 freeze logic still applies to **angle mode** (where snapshots
  are needed). Basin mode no longer uses it.
- The worker now branches: basin mode calls `simulate_basin_batch()`, angle mode
  calls `backend.simulate_batch()`.
- The `level_complete` signal payload differs by mode: `(N, 4)` final state
  for basin, `(N, 2, n_samples)` snapshots for angle.
- Cache accepts both 2D `(N, 4)` and 3D `(N, 2, n_samples)` arrays. Mode
  switch clears the cache to prevent shape confusion.

## Alternatives Considered

- **Vectorized DOP853**: SciPy's `solve_ivp` doesn't support batched IVPs.
  Could use `diffrax` (JAX) or hand-roll a vectorized adaptive stepper, but
  the complexity isn't justified yet.
- **Parallelized solve_ivp via ProcessPoolExecutor**: SciPy releases the GIL
  during integration, so thread-based parallelism is viable. Deferred to a
  follow-up if performance is insufficient.
- **Keep RK4 for basin, just drop snapshots**: simpler change, but misses the
  accuracy and efficiency benefits of adaptive stepping.

## Consequences

- Basin mode memory drops from `(N, 2, 96) float32` to `(N, 4) float32` — a
  48x reduction (e.g., 24 MB to 0.5 MB at 256x256).
- Basin mode is currently slower than Numba RK4 for large grids because of the
  Python-level per-trajectory loop. Performance optimization is planned.
- Angle mode is completely unaffected — same RK4 backends, same `BatchResult`.
- The canvas, view, and cache all branch on basin vs angle mode, with clear
  separation of data paths.
- `BasinResult` and `BatchResult` are both immutable `NamedTuple` types.
