# ADR-0007: Dual-Angle Snapshot Storage (theta1 + theta2)

**Status**: Accepted
**Date**: 2025-06

## Context

The inspect tool (ADR-0008) needs to display the pendulum's state at any time t,
which requires both theta1 and theta2. The original backend output was
`(N, n_samples)` storing only theta2 (sufficient for coloring, which maps
theta2 to hue).

## Decision

Change backend output from `(N, n_samples)` to `(N, 2, n_samples)` where
index 0 = theta1, index 1 = theta2. Both stored as unwrapped float32.

## Alternatives Considered

- **Separate theta1 array**: store theta1 in a parallel cache. Doubles the
  cache management complexity and risks theta1/theta2 arrays getting out of
  sync.
- **Recompute theta1 on demand**: rerun the simulation for a single trajectory
  when inspecting. Adds latency to hover and duplicates compute logic for a
  single trajectory.
- **Store full state (theta1, theta2, omega1, omega2)**: `(N, 4, n_samples)`.
  Doubles memory for angular velocities that nothing currently uses.

## Consequences

- Memory increase: 50% (from `(N, n_samples)` to `(N, 2, n_samples)`).
  For 256×256 with 96 samples: 24 MB → 48 MB. Acceptable.
- Cache validation updated: `ndim == 3` check.
- Coloring pipeline unchanged: callers pass `snapshots[:, 1, :]` (theta2 slice).
- All backends updated to store both angles.
- Enables future features that need theta1 (e.g. theta1-based coloring mode).
