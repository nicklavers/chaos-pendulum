# Inspect Tool

Third cursor tool mode for the fractal explorer. Hover to preview pendulum
diagrams; click to pin trajectories into a stacked animation with time scrubbing.

> Cross-ref: [canvas-rendering.md](canvas-rendering.md) for tool mode architecture.
> Cross-ref: [data-shapes.md](data-shapes.md) for `TrajectoryInfo`, `PinnedTrajectory`.

## Components

### InspectColumn (fractal/inspect_column.py, ~609 lines)

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

### TrajectoryIndicator (fractal/trajectory_indicator.py, ~220 lines)

Small clickable circle widget with basin color fill, winding numbers inside,
initial angles below, and an X button (shown on hover) for removal.

- `set_highlighted(bool)` — white border for primary trajectory
- `set_color(rgb)` — update fill color (e.g. on colormap change)
- Contrasting text color auto-selected via brightness formula
- **Signals**: `clicked(str)`, `remove_clicked(str)`, `hovered(str)`, `unhovered(str)`

### PendulumDiagram (fractal/pendulum_diagram.py, ~133 lines)

Small `QWidget` that draws a double-pendulum stick figure at given angles.

- `set_state(theta1, theta2)` — update angles, trigger repaint
- `set_params(params)` — update physics params (arm lengths affect drawing)
- `set_label(text)` — label above diagram (e.g. "Initial (t=0)", "At t = 12.5 s")
- Colors: bob1 orange (255,120,80), bob2 cyan (80,200,255), arms white, pivot grey

### WindingCircle (fractal/winding_circle.py, ~139 lines)

Small colored circle widget for basin-mode hover. Displays winding numbers
(n1, n2) with the basin colormap color fill.

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
    |-- basin mode: look up final_state at flat_idx, extract winding numbers
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
    |-- Extract winding numbers from final state
    |-- inspect_column.set_time_params(t_end, dt)
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
