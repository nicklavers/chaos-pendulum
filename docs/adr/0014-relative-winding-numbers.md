# ADR-0014: Relative Winding Numbers Over Absolute

**Status**: Accepted
**Date**: 2026-02

## Context

Basin mode colors each pixel by its winding number — the integer number of
full rotations the trajectory accumulates. The original implementation used
absolute winding numbers: `n = round(theta_final / 2pi)`.

This worked well in the center of the viewport but produced off-by-one errors
near full-rotation boundaries. For example, an initial condition at
theta = 3.10 (just below pi) that ends at theta_final = 9.55 (just above 3pi)
gets absolute winding number round(9.55 / 6.28) = round(1.52) = 2, but the
trajectory only made 1 net rotation from its starting position. The absolute
definition counts the initial offset as part of the winding.

## Decision

Use relative winding numbers everywhere:

    n = round(theta_final / 2pi) - round(theta_init / 2pi)

This measures net rotations from the initial position, eliminating the
off-by-one errors at boundaries.

The UI always uses relative winding numbers. There is no toggle — relative
is strictly better for visualization because it answers the question "how
many times did this pendulum go around?" rather than "what absolute angle
did it end at?"

## Alternatives Considered

### Keep both with a toggle

We briefly implemented an Absolute/Relative combo box in the controls panel.
This added UI complexity (extra signal, extra state in canvas/view/controls)
for no practical benefit — there was no scenario where absolute winding was
preferred.

### Adjust absolute thresholds

Could shift the rounding boundaries to reduce off-by-one cases, but this
only moves the problem rather than eliminating it.

## Consequences

**Positive**:
- Cleaner basin boundaries — no more off-by-one artifacts at full-rotation edges
- Simpler UI — no winding definition toggle
- Less state — no `_winding_definition` field in canvas or view
- Intuitive interpretation — winding number = net rotations from start

**Negative**:
- Requires initial angle arrays to be passed through the pipeline alongside
  final angles, increasing the `display_basin_final()` signature
- The view must reconstruct initial angles from the viewport at the correct
  (data) resolution, not the canvas target resolution, to handle progressive
  rendering correctly
- The legacy `extract_winding_numbers()` function is retained for backward
  compatibility in tests (test_basin_solver.py), adding minor code weight
