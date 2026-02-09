# ADR-0006: Whole-Viewport LRU Cache (Not Tile-Based)

**Status**: Accepted
**Date**: 2025-01

## Context

The fractal explorer needs to cache computed results to avoid recomputation
when the user revisits a viewport or scrubs the time slider. Two approaches:
whole-viewport caching or spatial tile-based caching.

## Decision

Use whole-viewport LRU cache with adaptive memory budget. Each cache entry
stores the complete `(N, 2, n_samples)` snapshot array for one viewport+params
combination.

## Alternatives Considered

- **Tile-based caching**: fixed tiles in angle space (e.g. π/8 × π/8),
  independently cached per resolution. Would allow compositing cached inner
  tiles on zoom-out and only computing border tiles. However, this requires
  restructuring the worker (per-tile compute), cache (spatial indexing), and
  canvas (multi-tile compositing). Deferred to phase 2+.

- **No cache**: recompute on every viewport change. Acceptable with Numba
  (< 0.5s) but painful with NumPy-only (10–15s).

## Consequences

- Simple implementation: `Dict[CacheKey, ndarray]` with `OrderedDict` LRU.
- Zooming out recomputes all pixels from scratch even though the inner region
  was previously cached. This is the main limitation.
- Viewport coordinates quantized to 1e-6 radian quantum to prevent float drift
  from causing cache misses.
- Protected coarse levels (resolution ≤ 64) are never evicted — they provide
  instant zoom-out previews at minimal memory cost (~1.5 MB each).
- Adaptive budget: 512 MB (NumPy), 128 MB (Numba), 64 MB (JAX) — backends
  with faster recomputation need less cache.
