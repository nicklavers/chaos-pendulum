# Inspect Tool

Third cursor tool mode for the fractal explorer. Hover to preview pendulum
diagrams; click to pin trajectories into a stacked animation with time scrubbing.

> Cross-ref: [canvas-rendering.md](canvas-rendering.md) for tool mode architecture.
> Cross-ref: [data-shapes.md](data-shapes.md) for `TrajectoryInfo`, `PinnedTrajectory`.
> Cross-ref: [ADR-0016](../adr/0016-freeze-frame-settle-aware-animation.md) for freeze-frame and settle-truncation design rationale.

## Components

### InspectColumn (fractal/inspect_column.py, ~660 lines)

Top-level `QWidget` shown to the left of the canvas in inspect mode.

**Layout** (top to bottom):
- Header row: "Pinned Trajectories" label + "Clear All" button
- Hover section: `PendulumDiagram` (initial) + `PendulumDiagram` (at-t, angle mode) or `WindingCircle` (basin mode)
- `MultiTrajectoryDiagram`: single stacked animation showing all pinned trajectories
- Scrub controls: `QSlider` + play/pause button + time label
- Indicator row: horizontal row of `TrajectoryIndicator` circles

**Key state**:
- `_pinned: dict[str, PinnedTrajectory]` — immutable, rebuilt on add/remove
- `_insertion_order: tuple[str, ...]` — preserves pin order
- `_primary_id: str | None` — which trajectory is foregrounded (full opacity + trail)
- `_dt`, `_t_end` — simulation time params for scrub label computation

**Key methods**:
- `add_row(row_id, theta1, theta2, trajectory, params, n1, n2)` — pin a trajectory
- `remove_row(row_id)` — remove a pinned trajectory
- `update_winding(row_id, n1, n2)` — update winding numbers + color for a trajectory
- `get_pinned() -> dict[str, PinnedTrajectory]` — read-only access to pinned data
- `get_marker_colors() -> dict[str, tuple[int, int, int]]` — current marker colors
- `set_winding_colormap(name)` — recolor all trajectories for a new colormap

**Signals**: `row_removed(str)`, `all_cleared()`, `indicator_hovered(str)`,
`indicator_unhovered(str)`

### MultiTrajectoryDiagram (fractal/animated_diagram.py, ~373 lines)

`QWidget` that draws multiple double-pendulum trajectories in a single animation.

- **Primary trajectory**: full-opacity basin-colored bobs, with accumulated trail
- **Ghost trajectories**: reduced alpha (`max(30, min(90, 1200 // n_traj))`), no trail
- **1-second pause** at frame 0 (PAUSE_FRAMES=30 at 30fps) before auto-advancing
- `set_trajectories(infos, params)` — replace all trajectories, reset animation
- `set_primary(index)` — change which trajectory is foregrounded
- `set_frame(frame_idx)` — jump to specific frame (for external scrubbing)
- `advance_frame()` — auto-advance by one frame (called by timer)
- `max_frames` property — length of longest trajectory
- `frame_idx` property — current frame index

### TrajectoryIndicator (fractal/trajectory_indicator.py, ~321 lines)

Vertical Venn diagram (two linked rings) with tapered arcs encoding winding
numbers. Top ring = arm 1 (n1), bottom ring = arm 2 (n2). Each ring shows
|n| tapered arc segments spanning 360/|n| degrees each, fat at the tail and
tapering to a point; n=0 shows a thin dotted outline. Positive n = CW
(clockwise), negative = CCW (counter-clockwise).

- 32px diameter per sub-circle, 35% overlap (~11px) → ~53px total vertical span
- Neutral gray (55,55,70) background circles; basin-colored tapered arcs with dark outline
- Winding number digit (13pt bold, +prefix for positive) offset toward circle center
- Multi-pass draw order: bg circles → top arcs → bottom arcs → digits
- `set_highlighted(bool)` — white border around unified figure-8 contour
- `set_color(rgb)` — update fill color (e.g. on colormap change)
- `set_winding(n1, n2)` — update winding numbers and trigger repaint
- **Signals**: `clicked(str)`, `remove_clicked(str)`, `hovered(str)`, `unhovered(str)`

### PendulumDiagram (fractal/pendulum_diagram.py, ~133 lines)

Small `QWidget` that draws a double-pendulum stick figure at given angles.

- `set_state(theta1, theta2)` — update angles, trigger repaint
- `set_params(params)` — update physics params (arm lengths affect drawing)
- `set_label(text)` — label above diagram (e.g. "Initial (t=0)", "At t = 12.5 s")
- Colors: bob1 orange (255,120,80), bob2 cyan (80,200,255), arms white, pivot grey

### WindingCircle (fractal/winding_circle.py, ~192 lines)

Vertical Venn diagram for basin-mode hover. Same linked-rings design as
TrajectoryIndicator but dynamically sized to fit its container (typically
110x110px). Neutral gray background circles with basin-colored tapered arcs
and dark outline. Multi-pass draw order matches TrajectoryIndicator.

### TaperedArc (fractal/arrow_arc.py, ~187 lines)

Pure geometry + QPainter drawing functions for tapered arc segments. Shared
by both TrajectoryIndicator and WindingCircle. Convention: 0° = top,
positive = clockwise. Direction: positive n = CW, negative n = CCW.

- `compute_tapered_arcs(n)` — returns list of `TaperedArc(start_deg, span_deg, clockwise)`
- `build_tapered_arc_polygon(cx, cy, arc_r, ...)` — returns QPolygonF for a single tapered arc
- `draw_tapered_arcs(painter, cx, cy, radius, n, fill_color, outline_color)` — render all arcs
- n=0: thin dotted circle; n≥1: |n| tapered arc polygons (fat tail → pointed tip)
- Jaunty rotation offset so all winding numbers start at the same angle as n=1

## Time Scrubbing

The scrub controls below the animation let the user scrub through time:

- **QSlider**: range 0 to `max_frames - 1`, synced bidirectionally with animation
- **Play/pause button**: toggleable (auto-plays on first pin). Pauses/resumes the animation timer.
- **Time label**: shows `t = X.X s` computed as `frame_idx * FRAME_SUBSAMPLE * dt`
- **Scrub interaction**: dragging the slider pauses auto-advance; releasing resumes if playing
- **Signal blocking**: slider sync during auto-advance uses `blockSignals` to prevent recursive loops

## Data Flow: Hover

```
mouseMoveEvent (inspect mode, cursor within image area)
    |
    v
FractalCanvas.hover_updated(theta1, theta2)
    |
    v
FractalView._on_hover_updated(theta1, theta2)
    |
    |-- angle mode: look up snapshots at (flat_idx, time_index)
    |     --> InspectColumn.update_hover_angle(...)
    |
    |-- basin mode: look up final_state at flat_idx, extract relative winding numbers
          (using hovered theta1/theta2 as initial angles)
          --> InspectColumn.update_hover_basin(...)
```

## Data Flow: Pin Trajectory

```
mouseReleaseEvent (inspect mode, click on canvas)
    |
    v
FractalCanvas.trajectory_pinned(row_id, theta1, theta2)
    |
    v
FractalView._on_trajectory_pinned(row_id, theta1, theta2)
    |-- Compute t_end (basin: 5/friction, capped 500; angle: user slider)
    |-- Run rk4_single_trajectory(params, theta1, theta2, t_end, dt)
    |-- Extract relative winding numbers from final state (using theta1/theta2 as init)
    |-- Look up basin color, update canvas marker color
    |-- inspect_column.set_time_params(t_end, dt)
    |-- inspect_column.add_row(row_id, theta1, theta2, trajectory, params, n1, n2)
         |
         |-- Subsample trajectory by FRAME_SUBSAMPLE (every 6th step)
         |-- Look up basin color via current winding colormap
         |-- Create frozen PinnedTrajectory
         |-- Create TrajectoryIndicator widget (with hover signals)
         |-- Rebuild MultiTrajectoryDiagram (all trajectories as TrajectoryInfo tuple)
         |-- Update scrub slider range
         |-- Auto-play if first trajectory
```

## Interaction

- Inspect mode: cursor changes to `PointingHandCursor`
- Hover updates diagrams in real-time
- Click pins a trajectory (adds to animation + indicator row + canvas marker)
- Canvas markers are colored X shapes matching the basin colormap color
- Click an indicator circle to foreground that trajectory
- Hover an indicator circle to highlight the corresponding canvas marker (larger size)
- Hover an indicator's X button to remove it
- "Clear All" removes all pinned trajectories and canvas markers
- Scrub slider or play/pause controls the animation timeline
- Colormap changes propagate to all indicator colors, the animation, and canvas markers

## Freeze-Frame Hover

Hovering over a `TrajectoryIndicator` temporarily replaces the normal
stacked animation with a static freeze-frame view of that single trajectory.

**Behaviour**:
- On indicator `hovered(row_id)`: build `TrajectoryInfo` from the corresponding
  `PinnedTrajectory`, call `MultiTrajectoryDiagram.enter_freeze_frame(tinfo)`,
  stop the animation timer.
- On indicator `unhovered(row_id)`: call `exit_freeze_frame()`, restart the
  timer if `_scrub_playing` is true.
- `_on_play_toggled`: skip `_anim_timer.start()` while `is_frozen`.
- `remove_row` / `clear_all`: call `exit_freeze_frame()` before cleanup.

**Freeze-frame rendering** (`_paint_freeze_frame`):
1. Full bob2 trace as per-segment polyline with **two-tier alpha**: segments before
   the system settles into a basin are drawn at high opacity (`FREEZE_TRACE_ACTIVE_ALPHA`);
   segments after settling are drawn at low opacity (`FREEZE_TRACE_SETTLED_ALPHA`).
   A short transition region (`FREEZE_TRACE_TRANSITION` frames) blends linearly between
   the two. The settle point is detected by comparing `total_energy(state, params)`
   against `saddle_energy(params)` — the first frame where energy drops below the
   saddle threshold. If friction is zero or energy never drops, full active alpha is used.
2. Pendulum arms at 5 evenly-spaced keyframes (0%, 25%, 50%, 75%, 100%).
   Gray (220,220,220) arms with alpha ramp 50→200. Basin-colored bobs (5px radius)
   with matching alpha.
3. Pivot drawn on top (same as normal mode).

**Constants** (in `fractal/animated_diagram.py`):

| Constant | Value | Description |
|----------|-------|-------------|
| `FREEZE_TRACE_WIDTH` | 2 | Trace line width (px) |
| `FREEZE_TRACE_ACTIVE_ALPHA` | 200 | Pre-settling trace opacity |
| `FREEZE_TRACE_SETTLED_ALPHA` | 40 | Post-settling trace opacity |
| `FREEZE_TRACE_TRANSITION` | 20 | Transition region width (frames) |
| `FREEZE_ARM_ALPHA_MIN` | 50 | Earliest keyframe arm opacity |
| `FREEZE_ARM_ALPHA_MAX` | 200 | Latest keyframe arm opacity |
| `FREEZE_KEYFRAME_COUNT` | 5 | Number of keyframe poses |
| `FREEZE_BOB_RADIUS` | 5 | Keyframe bob radius (px) |
| `SETTLE_BUFFER_SECONDS` | 5.0 | Post-settle buffer before truncation |

## Settle-Based Animation Truncation

When all pinned trajectories have dropped below the saddle energy threshold
(i.e. they are permanently captured in a basin), the animation is truncated
so it ends `SETTLE_BUFFER_SECONDS` after the **latest** trajectory settles.

**Mechanism**:
- `MultiTrajectoryDiagram._compute_effective_max()` runs `_find_settled_index()`
  on each trajectory. If every trajectory has a settle point, the effective max is
  `max(settle_indices) + ceil(SETTLE_BUFFER_SECONDS / dt_per_frame)`.
- If any trajectory never settles (friction=0 or energy stays above threshold),
  no truncation occurs — the full trajectory length is used.
- `_effective_max` is recomputed on every `set_trajectories()` call, so
  adding/removing trajectories automatically updates the animation duration.
- The `max_frames` property returns `min(actual_max, effective_max)` when
  truncation is active. The scrub slider range, animation loop, and time label
  all derive from `max_frames`, so they update automatically.
- `InspectColumn.set_time_params()` calls `set_dt_per_frame(FRAME_SUBSAMPLE * dt)`
  so the diagram knows the correct time scale.

## Constants

| Constant | Value | Location |
|----------|-------|----------|
| `ANIMATION_INTERVAL_MS` | 33 | `fractal/inspect_column.py` |
| `FRAME_SUBSAMPLE` | 6 | `fractal/inspect_column.py` |
| `PAUSE_FRAMES` | 30 | `fractal/animated_diagram.py` |
| `TRAIL_LENGTH` | 15 | `fractal/animated_diagram.py` |
