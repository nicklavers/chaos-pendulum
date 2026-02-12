# ADR-0016: Freeze-Frame Hover and Settle-Aware Animation Truncation

**Status**: Accepted
**Date**: 2026-02

## Context

The inspect column shows stacked pendulum animations for pinned trajectories.
Two UX problems emerged:

1. **No quick overview**: To understand a trajectory's shape you had to watch
   the full animation play out, or scrub through manually.
2. **Long settling tails**: With friction, trajectories spend most of their
   duration spiraling down to rest. The interesting chaotic phase is a small
   fraction of the total — the rest is visual noise.

We already compute `saddle_energy(params)` for early termination in the batch
solver (ADR-0011). The same threshold cleanly distinguishes "chaotic phase"
from "captured in a basin".

## Decision

### Freeze-frame hover

Hovering a `TrajectoryIndicator` temporarily replaces the stacked animation
with a static freeze-frame view of that single trajectory: full bob2 trace
plus pendulum arms at 5 evenly-spaced keyframes.

### Two-tier trace alpha

The freeze-frame trace uses high opacity (`FREEZE_TRACE_ACTIVE_ALPHA=200`)
before the energy crosses below `saddle_energy(params)` and low opacity
(`FREEZE_TRACE_SETTLED_ALPHA=40`) after, with a 20-frame linear transition.
This makes the chaotic phase visually dominant.

### Settle-based animation truncation

When **all** pinned trajectories have dropped below the saddle energy
threshold, the animation loop is truncated to end `SETTLE_BUFFER_SECONDS`
(5 s) after the latest settle point. This shortens the loop from potentially
hundreds of seconds to just the interesting part plus a brief settling tail.
Adding or removing trajectories recomputes the effective max.

## Alternatives Considered

1. **Velocity-based threshold** (e.g., `|omega| < epsilon`): Simpler, but
   doesn't account for potential energy. A slowly-moving pendulum at the top
   of a swing isn't settled. The energy threshold is physically meaningful.

2. **Fixed animation duration**: Would require the user to guess a good
   `t_end` for every parameter combination. The energy-based approach adapts
   automatically.

3. **Always show full trajectory in freeze-frame**: Without the alpha
   dimming, the settled spiral dominates and obscures the chaotic phase.
   Two-tier alpha was preferred over uniform alpha or trajectory truncation
   because it preserves full information while directing attention.

## Consequences

**Positive**:
- Instant visual summary on hover — no need to watch or scrub.
- Animation loops are shorter and more focused.
- Reuses existing `saddle_energy` infrastructure — no new physics needed.

**Negative**:
- `animated_diagram.py` now imports `fractal/compute.py` (for `saddle_energy`)
  in addition to `simulation.py`. This adds a cross-module dependency.
- `_find_settled_index` iterates the trajectory calling `total_energy` per
  frame. For long trajectories this is O(n) per trajectory, computed once on
  `set_trajectories()` and once per freeze-frame hover. Acceptable for the
  small number of pinned trajectories (typically 1-5).
