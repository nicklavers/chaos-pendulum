# ADR 0013: Convergence-Time Brightness Modulation for Basin Mode

**Status**: Accepted
**Date**: 2026-02

## Context

Basin mode renders each pixel as the solid color of the basin (winding number
pair) that its trajectory settles into. This produces large flat regions of
uniform color, lacking visual texture. Basin *boundaries* — where convergence
is slow and chaotic — look identical to basin *interiors* — where convergence
is fast and predictable.

Meanwhile, the "Direction + Brightness" winding colormap already used brightness
as a variable (encoding distance-from-origin in winding space). Adding a second
brightness dimension (convergence time) would conflict with it, and the
colormap was not well-liked aesthetically.

Separately, `fractal/basin_solver.py` (DOP853 adaptive solver, introduced in
ADR-0012) was never wired into the runtime import graph. Both RK4 backends
(`_numpy_backend.py`, `_numba_backend.py`) implement their own
`simulate_basin_batch()` methods with energy-based freeze logic. The
basin_solver module is dead code.

## Decision

1. **Track convergence time in `BasinResult`**: add a `convergence_times: (N,)`
   float32 field. Each value records the simulation time at which the trajectory
   crossed the saddle-energy threshold (or `t_end` if it never did). Both RK4
   backends and the unused DOP853 solver populate this field.

2. **Modulate pixel brightness by convergence time**: after applying the winding
   colormap, optionally scale each pixel's BGR channels by a brightness factor
   derived from its normalized convergence time. The factor ranges from 0.3
   (dimmest) to 1.0 (brightest). Alpha is unchanged.

3. **Three shading modes** controlled by a UI combo (basin mode only):
   - *Fast = bright* (default): fast convergence → bright, slow → dark. Basin
     interiors glow; boundaries are dark.
   - *Fast = dark*: inverted. Highlights chaotic boundary zones.
   - *Off*: no modulation — flat colors as before.

4. **Remove "Direction + Brightness" winding colormap**: it conflicts with
   convergence-time brightness and was aesthetically inferior. Two colormaps
   remain: "Modular Grid (5x5)" and "Basin Hash".

5. **Document `basin_solver.py` as dead code**: the file is retained but the
   file-map and this ADR note that it is not imported by any runtime module.

## Alternatives Considered

- **Separate colormap variant**: add a new "Modular Grid + Shading" colormap
  instead of post-processing. Rejected because brightness modulation is
  orthogonal to the base colormap — it should compose with any colormap, not
  duplicate each one.

- **Per-tile normalization**: normalize convergence times within each
  progressive rendering tile rather than globally. Rejected because tiles at
  different zoom levels would have inconsistent brightness, and the final
  full-resolution render would look different from intermediate tiles.

- **Continuous HSL lightness**: map convergence time to HSL lightness rather
  than a linear brightness scale. Rejected as over-complicated; the linear
  [0.3, 1.0] range produces clear visual contrast without desaturating colors.

## Consequences

- Basin mode gains visual texture: smooth interiors vs. rough boundaries are
  now visually distinct.
- Shading mode toggle is instant — no recomputation needed, only a pixel
  rebuild (~5-10ms).
- `BasinResult` grows from 1 to 2 fields. Cache packs the extra column into
  `(N, 5)` float32 with no API change.
- The winding colormap registry drops from 3 to 2 entries. Any code referencing
  `winding_direction_brightness` must be updated.
- `basin_solver.py` is formally recognized as dead code. It may be revived if
  adaptive stepping is needed in the future.
