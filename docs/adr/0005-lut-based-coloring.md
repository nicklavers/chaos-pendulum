# ADR-0005: 4096-Entry LUT for Color Mapping

**Status**: Accepted
**Date**: 2025-01

## Context

The display pipeline must convert ~65K theta2 values to ARGB32 pixels every
frame during time-slider scrubbing. This must complete in < 10ms for 60fps.

## Decision

Pre-compute a 4096-entry lookup table mapping normalized angle [0, 1) to
ARGB32 uint32. Color mapping becomes a single vectorized integer index
operation.

## Alternatives Considered

- **256-entry LUT**: visible color banding in gradient regions.
- **Per-pixel HSV computation**: ~50ms per frame for 256×256 — too slow for
  smooth scrubbing.
- **Matplotlib colormap with interpolation**: ~20ms per frame, still too slow
  and adds a heavy dependency for the hot path.

## Consequences

- 4096 entries = 16 KB, fits in L1 cache. Negligible memory.
- LUT rebuild on colormap change takes ~1ms — imperceptible.
- Total scrub frame: interpolation (< 0.1ms) + LUT lookup (~2–3ms) +
  QImage rebuild (~5ms) = < 10ms.
- Adding new colormaps requires only generating a new LUT array.
