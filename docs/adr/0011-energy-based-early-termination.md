# ADR 0011: Energy-Based Early Termination for Basin Mode

**Status**: Accepted
**Date**: 2026-02

## Context

Basin mode simulates damped double pendulums to map where each trajectory
settles (its winding number). With friction > 0, every trajectory eventually
loses energy and converges to a fixed point. However, many trajectories
settle quickly while the simulation runs for the full `t_end` duration.

We needed a way to detect when a trajectory has settled and skip the
remaining integration steps.

## Decision

Compute the **saddle-point energy** — the lowest potential energy barrier
between adjacent basins — and freeze any trajectory whose total energy
drops below it. A frozen trajectory can never accumulate enough energy to
change winding number, so its final state is already determined.

### Implementation

1. `saddle_energy(params)` computes `min(V(pi,0), V(0,pi))` — the two
   candidate saddle configurations at rest.
2. The worker passes `saddle_energy_val` to `simulate_batch()` when
   `task.basin=True` and `params.friction > 0`.
3. Each backend periodically checks `total_energy < saddle_energy_val`:
   - **NumPy**: every 50 steps, vectorized across all trajectories
   - **Numba**: every step, per-trajectory (cheap in compiled code)
4. Frozen trajectories have remaining snapshot slots filled with the
   freeze-time angle and are excluded from further RK4 steps.

### Return type change

`simulate_batch()` now returns `BatchResult(snapshots, final_velocities)`
instead of a bare snapshot array. The velocity sidecar enables direct
convergence checking without finite-differencing.

## Alternatives Considered

- **Velocity threshold**: freeze when `|omega| < epsilon`. Rejected because
  a trajectory can have small velocity while still having enough energy to
  escape (e.g., near the top of a potential hill).
- **Angle stability window**: freeze when angles don't change over N steps.
  Rejected because it conflates the snapshot sampling rate with the physics.
- **Adaptive time step**: rejected for the same reasons as ADR-0001 —
  fixed-step RK4 is simpler and sufficient for fractal visualization.

## Consequences

- Basin mode with high friction is significantly faster (most trajectories
  freeze well before `t_end`).
- `ComputeBackend` protocol gains an optional `saddle_energy_val` parameter.
- All backends must return `BatchResult` instead of bare `np.ndarray`.
- The freeze check adds a small per-step cost (~2% overhead when no
  trajectories freeze).
