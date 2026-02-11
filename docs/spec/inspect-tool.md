# Inspect Tool

Third cursor tool mode for the fractal explorer. Hover to preview pendulum
diagrams; click to pin trajectories into a stacked animation with time scrubbing.

> Cross-ref: [canvas-rendering.md](canvas-rendering.md) for tool mode architecture.
> Cross-ref: [data-shapes.md](data-shapes.md) for `TrajectoryInfo`, `PinnedTrajectory`.

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

## Constants

| Constant | Value | Location |
|----------|-------|----------|
| `ANIMATION_INTERVAL_MS` | 33 | `fractal/inspect_column.py` |
| `FRAME_SUBSAMPLE` | 6 | `fractal/inspect_column.py` |
| `PAUSE_FRAMES` | 30 | `fractal/animated_diagram.py` |
| `TRAIL_LENGTH` | 15 | `fractal/animated_diagram.py` |
