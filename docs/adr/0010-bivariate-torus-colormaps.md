# ADR-0010: Bivariate Torus Colormaps for Dual-Angle Display

**Status**: Accepted
**Date**: 2025-06

## Context

With dual-angle snapshots (ADR-0007), the app can display either theta1 or
theta2 individually via 1D LUT coloring. A natural next step is displaying
both angles simultaneously, encoding the full (theta1, theta2) state as color.

Both angles are 2pi-periodic, so the color domain is a torus T^2 = [0, 2pi) x
[0, 2pi). No standard "torus colormap" library exists; we needed to design our
own color functions.

Key constraints:
- Must be 2pi-periodic in both inputs (torus topology, no seams)
- Must produce visually distinct colors for different (theta1, theta2) pairs
- Must run in < 10ms for 256x256 (65,536 pixels) to keep scrubbing smooth
- Ideally, landmark colors should align with physically meaningful state-space
  positions (e.g. both-bobs-down at (0,0), both-horizontal at (pi,pi))

## Decision

Create a new module `fractal/bivariate.py` with:

1. **Function-based colormaps** (not LUT-based): each torus colormap is a
   vectorized NumPy function `(theta1, theta2) -> (N, 4) uint8 BGRA`. This
   avoids the 2D LUT memory cost (4096x4096 = 64MB) while keeping per-frame
   cost under 10ms.

2. **Nine colormaps** spanning different visual strategies:
   - 3 general-purpose (hue-lightness, warm-cool, diagonal hue)
   - 6 RGB-sinusoid variants with progressive landmark alignment

3. **Landmark-aligned RGB sinusoid family**: the base formula
   `R = 128 + 127*sin(t1 - pi/2)`, `G = 128 + 127*sin(t1 - pi/2 + t2)`,
   `B = 128 + 127*sin(t1 - pi/2 - t2)` places pure red at (pi,pi), black at
   (0,0), cyan at (0,pi), white at (pi,0). Four correction variants add
   sin/cos terms (which vanish at pi-multiples) to assign pure named colors
   to the half-pi grid points without disturbing the primary landmarks.
   The variants span a tradeoff between visual crispness and landmark count:
   YB and GM use `sin(t1)*sin(t2)` on one channel (8 landmarks, checkerboard);
   YBGM uses `sin(t2)` on one channel (8 landmarks, 4 distinct diagonal colors);
   6-Color uses 3 cross-terms across all channels (16 landmarks, fluid regions).

4. **Dynamic colormap combo**: the controls dropdown swaps between `COLORMAPS`
   (univariate) and `TORUS_COLORMAPS` (bivariate) when the angle selector
   changes, using a `_building` guard to prevent signal cascades.

5. **2D square legend**: replaces the donut wheel legend when in bivariate mode.
   64x64 pixels showing the full torus colormap with theta1/theta2 axis labels.

## Alternatives Considered

- **2D LUT**: Pre-compute a 2D array indexed by (quantized_theta1, quantized_theta2).
  At 4096x4096 entries, this costs 64MB per colormap — prohibitive for 8
  colormaps. At 256x256, visible banding. The function approach uses no extra
  memory and lets each colormap use arbitrary math.

- **Existing matplotlib/seaborn 2D colormaps**: No standard library offers
  torus-periodic 2D colormaps. Matplotlib's bivariate colormaps are for
  scalar+scalar on bounded domains, not periodic angles.

- **HSL hue-saturation circle**: Map theta1 to hue, theta2 to saturation.
  Produces perceptually uneven results (low-saturation colors are hard to
  distinguish). Implemented as one option (hue-lightness) but not the primary.

- **Single merged angle**: Map `f(theta1, theta2)` to a 1D colormap. Loses
  the ability to distinguish (theta1, theta2) pairs that map to the same
  scalar. Fundamentally cannot encode 2D information in 1D color.

## Consequences

- New file `fractal/bivariate.py` (~577 lines, 9 colormap functions + helpers).
- Canvas has a dual-path `_rebuild_image()`: univariate via LUT, bivariate
  via function call.
- Controls emit `torus_colormap_changed(str)` in addition to the existing
  `colormap_changed(str)`, dispatched by angle mode.
- Performance verified: all 9 colormaps complete in < 10ms at 256x256 (tested
  with parametrized benchmark).
- Comprehensive test coverage: 145 tests covering output shape, periodicity,
  discrimination, exact landmark values (including all 16 half-pi grid points
  for the 6-color variant and 4 distinct diagonal colors for YBGM),
  performance budgets, and registry consistency.
