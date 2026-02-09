# ADR-0004: Coarse-to-Fine Progressive Rendering

**Status**: Accepted
**Date**: 2025-01

## Context

Computing 256×256 (65K trajectories) takes 10–15s on NumPy. The user should not
stare at a blank canvas for that long.

## Decision

Compute multiple resolution levels from coarse to fine, displaying each level
immediately via nearest-neighbor upscaling. The level list is parameterized by
backend speed.

Without Numba: 64×64 → 128×128 → 256×256
With Numba: 128×128 → 256×256

## Alternatives Considered

- **Single resolution with loading bar**: functional but poor UX — no preview.
- **Tile-by-tile rendering**: complex compositing, partial images look worse
  than a uniformly coarse preview.
- **32×32 first level**: dropped — 64×64 is fast enough (< 0.5s) and 32×32
  is too blocky to be useful.

## Consequences

- Nearest-neighbor upscaling (not bilinear) preserves the blocky pixel grid
  that communicates "this is a preview." Bilinear would blur sharp fractal
  boundaries.
- Each level is a complete independent computation (no data reuse between
  levels). Simple to implement, but computes more total work than needed.
- Cancellation between levels provides responsive abort (~1s) when the user
  changes viewport during computation.
