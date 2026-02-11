# ADR-0015: Viewport Transition Compositing

**Status**: Accepted
**Date**: 2026-02

## Context

When zooming out, the entire canvas was replaced by a lower-res preview (cube
or cache), losing the high-res detail the user just had. When panning, newly
exposed edges were black until the mouse was released and the progressive
pipeline finished. Both transitions felt jarring.

## Decision

Two independent compositing layers in the canvas `paintEvent`, each solving
one problem:

**Layer 1 — Pan background (during drag only):** A low-res QImage rendered from
the data cube, drawn *behind* the shifting foreground. Fills edges that the
foreground doesn't cover. Provided by `FractalView` in response to the
`pan_started` signal. Cleared on mouse release.

**Layer 2 — Stale overlay (after viewport transitions):** The previous high-res
QImage + its viewport, drawn *on top of* the new lower-res preview at the
correct physics position. Cleared when the progressive pipeline finishes (or on
full-res cache hit).

The stale overlay is **not** saved for zoom-in. When zooming in, the stale image
was rendered at a *wider* viewport than the new one — drawing it on top would
completely cover the main image and prevent progressive levels from showing
through. Zoom-in already works well because the 64x64 first progressive level
arrives fast.

Additionally, the progressive pipeline now filters out resolution levels that
are already cached, skipping straight to the level that actually needs
computing. This avoids redundant work when returning to a viewport where
some but not all levels were previously computed.

## Alternatives Considered

**Single-layer approach (stale only):** Would handle zoom-out but not pan drag.
Pan edges would still be black during drag since the stale overlay is cleared
at pan start (it belongs to the pre-pan viewport).

**Tile-based cache:** Would solve both problems more elegantly (only recompute
new tiles, keep existing tiles in place). Much higher complexity — requires
tile management, compositing, and a completely different cache architecture.
Deferred as a potential Phase 2 improvement.

**Bilinear upscaling of the main image during pan:** Would fill edges by
stretching the foreground beyond its natural bounds. Introduces blurry
artifacts and incorrect physics mapping at the edges.

## Consequences

**Positive:**
- Zoom-out preserves crisp detail in the center while surrounding area sharpens
- Pan drag shows blurry (but correct) data at edges instead of black
- Pan release preserves crisp overlapping region while new areas sharpen
- Progressive level filtering avoids redundant computation on cache partial hits
- No architectural changes to cache, worker, or compute layers

**Negative:**
- Two extra QImage draws per frame in paintEvent (only when overlays are active)
- `_stale_image` holds a copy of the previous QImage (memory cost of one extra
  image, typically ~256 KB for 256x256 ARGB)
- Pan background is limited to the data cube's coverage (full [0, 2π]² at 64x64);
  panning beyond the cube viewport still shows black at edges
