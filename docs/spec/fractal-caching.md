# Fractal Caching

Cache architecture for computed fractal data. File: `fractal/cache.py` (~176 lines).

> Cross-ref: [data-shapes.md](data-shapes.md) for `CacheKey` and snapshot shape.

## Design

Dict-based LRU with adaptive memory budget. Stores complete snapshot arrays
keyed by quantized viewport + physics params hash.

```python
Dict[CacheKey, np.ndarray]  # values are (N, 2, n_samples) float32
```

## Memory Budget

Adaptive based on detected compute backend (faster backends need less cache):

| Backend | Budget | Rationale |
|---------|--------|-----------|
| NumPy | 512 MB | Recomputation is expensive (~10–15s) |
| Numba | 128 MB | Recomputation is cheap (~0.5s) |
| JAX | 64 MB | Recomputation is near-free (< 1s) |

## LRU Eviction

Uses `OrderedDict` for LRU ordering. When a `put()` would exceed budget,
oldest entries are evicted until there is room.

**Protected coarse levels**: Entries with `resolution <= 64` are never evicted.
They are ~1.5 MB each and provide instant zoom-out previews.

## Viewport Quantization

Viewport coordinates are quantized to a 1e-6 radian quantum before cache
lookup. This prevents floating-point drift from causing cache misses on
"same" viewports.

```python
CacheKey.from_viewport(viewport, params)  # quantizes + hashes
```

## Invalidation

- **Physics params change**: `invalidate_params(old_params_hash)` removes all
  entries for the old parameter set. The `params_hash` field in `CacheKey`
  also prevents stale hits.
- **t_end change**: `clear()` removes all entries (simulation duration changed,
  all snapshots are invalid).
- Cache lookup happens **after** the 300ms debounce timer fires, not during
  intermediate pan/zoom events.

## Validation

`put()` validates the array on insertion:
- Must be 3D: `ndim == 3` (shape `(N, 2, n_samples)`)
- Must be float32 dtype
- Raises on invalid data to catch integration bugs early

## Future: Tile-Based Caching (Phase 2+)

The current whole-viewport cache means zooming out recomputes all pixels from
scratch. A tile-based approach (fixed tiles in angle space, independently cached
per resolution level) would allow compositing cached tiles and only computing
new border tiles. Deferred until needed.
