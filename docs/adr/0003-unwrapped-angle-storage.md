# ADR-0003: Store Unwrapped Angles, Modulo at Color Time

**Status**: Accepted
**Date**: 2025-01

## Context

The snapshot array stores theta values at sampled timesteps. The question is
whether to store `theta % (2*pi)` (wrapped) or raw accumulated angle (unwrapped).

## Decision

Store unwrapped angles. Apply `% (2*pi)` modulo only at color-mapping time in
the coloring pipeline.

## Alternatives Considered

- **Store wrapped**: saves no space (still float32), but introduces interpolation
  artifacts. If theta at sample k is 6.2 (near 2π) and at sample k+1 is 0.1
  (wrapped), linear interpolation gives ~3.15 — completely wrong. Unwrapped
  values (6.2 and 6.38) interpolate correctly to 6.29.

## Consequences

- Correct interpolation across the 0/2π boundary with no special-case code.
- Preserves winding number for potential future "unwrap toggle" colormap.
- The `% (2*pi)` in the coloring pipeline is a single vectorized operation
  with zero additional cost.
- Unwrapped values can grow large for long simulations (e.g. theta = 50.0
  after many full rotations) but float32 has sufficient precision for this.
