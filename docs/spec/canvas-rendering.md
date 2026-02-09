# Canvas Rendering

Drawing pipeline for the fractal canvas: image display, axes, legend, ghost
rectangle, and tool modes. File: `fractal/canvas.py` (~887 lines).

> Cross-ref: [data-shapes.md](data-shapes.md) for `FractalViewport` and signals.
> Cross-ref: [coloring-pipeline.md](coloring-pipeline.md) for pixel array generation.

## FractalCanvas(QWidget)

Central display widget. Receives `(N, 2, n_samples)` snapshot data, renders
the current time-slice as a QImage, and draws overlays (axes, legend, ghost rect).

### Coordinate Mapping

`QTransform` maps between pixel space and (theta1, theta2) physics space.

- **X-axis**: theta1 (horizontal)
- **Y-axis**: theta2 (vertical)
- Margins: `AXIS_MARGIN_LEFT = 72px`, `AXIS_MARGIN_BOTTOM = 52px`

### Image Display

`display(snapshots, time_index)`:
1. Stores `_current_snapshots`
2. Calls `_rebuild_image()` which passes `snapshots[:, 1, :]` (theta2 slice)
   through the coloring pipeline
3. Result: `QImage` rendered via `QPainter.drawImage()` with nearest-neighbor
   transform

### Axes and Labels

- Tick positions via `_generate_ticks()`: prefers multiples of π/4
- Labels via `_format_angle()`: π-fraction notation (e.g. "π/2", "3π/4")
- Dashed reference lines at θ=π through the image area
- Axis titles "θ₁" (horizontal) and "θ₂" (vertical, rotated)

### Color Wheel Legend

Donut-shaped legend in the bottom-right corner showing theta2-to-color mapping.

- 72 pie segments using active LUT colors
- `LEGEND_OUTER_RADIUS = 36`, `LEGEND_INNER_RADIUS = 22`
- Tick marks and labels at 0, π/2, π, 3π/2
- "θ₂" label centered in the donut

**Angle convention**: θ=0 at 6 o'clock (bottom, pendulum hanging down), angles
increase clockwise. Qt angles go counter-clockwise from 3 o'clock, so:
- Pie segments: `start_deg = -90.0 + i * span_angle`
- Tick marks: `screen_rad = π/2 - angle_rad`

### Ghost Rectangle (Zoom-Out Feedback)

When zooming out, a semi-transparent rectangle shows where the previous viewport
sits within the new (larger) viewport.

- Appears at full opacity during computation
- Fade-out timer starts after final render (`activate_pending_ghost()`)
- `GHOST_FADE_MS = 2000ms`, `GHOST_INITIAL_ALPHA = 220`

## Tool Modes

Three modes managed by `QButtonGroup` (exclusive toggle) in controls.
See [inspect-tool.md](inspect-tool.md) for the inspect mode data flow.

| Mode | Constant | Cursor | Behavior |
|------|----------|--------|----------|
| Zoom | `TOOL_ZOOM` | CrossCursor | Drag rectangle to zoom in |
| Pan | `TOOL_PAN` | OpenHand / ClosedHand | Drag to pan with live image offset |
| Inspect | `TOOL_INSPECT` | PointingHand | Hover emits `hover_updated` signal |

All modes support Ctrl+click for IC selection (emits `ic_selected`, launches
trajectory in pendulum mode).

## Pan/Zoom Mechanics

- **Zoom**: Mouse wheel centered on cursor (1.2× per notch)
- **Pan**: Left-click drag with live image offset (image tracks cursor
  before recomputation)
- **Debounce**: 300ms single-shot QTimer after last interaction, then
  constructs `FractalViewport` and emits `viewport_changed`
- During debounce: existing QImage stretched/translated for instant feedback
