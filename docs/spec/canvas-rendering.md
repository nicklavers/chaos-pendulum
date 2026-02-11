# Canvas Rendering

Drawing pipeline for the fractal canvas: image display, axes, legend, ghost
rectangle, viewport transition compositing, and tool modes.
File: `fractal/canvas.py` (~1167 lines).

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
2. Calls `_rebuild_image()` which takes one of two paths:
   - **Univariate** (`_angle_index` 0 or 1): passes `snapshots[:, angle_index, :]`
     through `interpolate_angle()` then `angle_to_argb()` (1D LUT)
   - **Bivariate** (`_angle_index == 2`): interpolates both angle slices, calls
     `bivariate_to_argb()` with the active torus colormap function
3. Result: `QImage` rendered via `QPainter.drawImage()` with nearest-neighbor
   transform

`set_angle_index(index)`: Accepts 0 (theta1), 1 (theta2), or 2 (both).
Triggers `_rebuild_image()`.

`set_torus_colormap(name)`: Switches the active torus colormap from
`TORUS_COLORMAPS` and invalidates the cached legend. Only rebuilds image
if currently in bivariate mode.

**Basin mode**: `display_basin_final(theta1_final, theta2_final, convergence_times, theta1_init, theta2_init)` stores final and initial angle arrays and calls `_rebuild_image_winding()`, which passes both to `winding_to_argb()` for relative winding number extraction. The view reconstructs init angles at the data's own resolution (not the canvas target resolution) to handle progressive rendering correctly. Default winding colormap: "Basin Hash".

### Axes and Labels

- Tick positions via `_generate_ticks()`: prefers multiples of π/4
- Labels via `_format_angle()`: π-fraction notation (e.g. "π/2", "3π/4")
- Dashed reference lines at θ=π through the image area
- Axis titles "θ₁" (horizontal) and "θ₂" (vertical, rotated)

### Legends

Two legend styles, selected by angle mode:

#### Color Wheel Legend (univariate)

Donut-shaped legend in the bottom-right corner showing single-angle-to-color
mapping. Drawn by `_draw_legend()`.

- 72 pie segments using active LUT colors
- `LEGEND_OUTER_RADIUS = 36`, `LEGEND_INNER_RADIUS = 22`
- Tick marks and labels at 0, π/2, π, 3π/2
- "θ₂" (or "θ₁") label centered in the donut

**Angle convention**: θ=0 at 6 o'clock (bottom, pendulum hanging down), angles
increase clockwise. Qt angles go counter-clockwise from 3 o'clock, so:
- Pie segments: `start_deg = -90.0 + i * span_angle`
- Tick marks: `screen_rad = π/2 - angle_rad`

#### Torus Legend (bivariate)

64x64 pixel square legend in the bottom-right corner showing the 2D torus
colormap. Drawn by `_draw_torus_legend()`.

- Built via `build_torus_legend(colormap_fn, 64)` from `fractal/bivariate.py`
- Cached as `_cached_torus_legend_image`; invalidated on torus colormap change
- Axis labels: "θ₁" below (horizontal), "θ₂" left (vertical, rotated)
- Corner labels: "0" and "2π" on both axes
- Light border around the square

### Ghost Rectangle (Zoom-Out Feedback)

When zooming out, a semi-transparent rectangle shows where the previous viewport
sits within the new (larger) viewport.

- Appears at full opacity during computation
- Fade-out timer starts after final render (`activate_pending_ghost()`)
- `GHOST_FADE_MS = 2000ms`, `GHOST_INITIAL_ALPHA = 220`

## Viewport Transition Compositing

Two-layer compositing system that provides visual continuity during viewport
changes (zoom-out and pan). See [ADR-0015](../adr/0015-viewport-transition-compositing.md).

### paintEvent Layer Order

1. Dark background fill (`QColor(20, 20, 30)`)
2. **Pan background** — low-res cube preview at full [0, 2π]² (during pan drag only)
3. **Main image** (`_current_image`, shifted during pan)
4. **Stale overlay** — previous high-res image at mapped sub-region (after transitions)
5. Axes, legend, markers, ghost rect (existing overlays)

All image layers (2–4) are clipped to the image area via `setClipRect`.

### Stale Overlay

Preserves high-res detail from the previous viewport while the progressive
pipeline sharpens the new, wider view.

**Fields**: `_stale_image: QImage | None`, `_stale_viewport: FractalViewport | None`

**Lifecycle**:
- `save_stale(viewport=None)` — snapshots `_current_image.copy()` and its viewport.
  Defaults to current viewport (correct for zoom-out, called before spans change).
- `clear_stale()` — called by `FractalView` when progressive pipeline finishes,
  on full-res cache hit, or on physics param change.
- Drawn on top of the main image at the correct physics position using
  `_physics_to_pixel()` coordinate mapping.

**When set**:
- **Zoom out**: `save_stale()` called before spans change in `zoom_out()`.
- **Pan release**: `save_stale(anchor_viewport)` called with the pre-pan viewport.
- **Pan start**: `clear_stale()` called (clean slate for the drag).

**Not set for zoom-in**: The stale image would completely cover the new zoomed-in
view since it was rendered at a wider viewport. The progressive pipeline starts
fast enough (64x64) that zoom-in works well without a stale overlay.

### Pan Background

Low-res cube data drawn behind the shifting foreground during pan drag, filling
exposed edges that would otherwise be black.

**Fields**: `_pan_background: QImage | None`, `_pan_bg_viewport: FractalViewport | None`

**Lifecycle**:
- `set_pan_background(image, viewport)` — called by `FractalView._on_pan_started()`
  with a cube-rendered QImage at the full [0, 2π]² viewport.
- `clear_pan_background()` — called on pan release in `mouseReleaseEvent`.
- Only drawn while `_panning` is true.

**Signal**: `pan_started` (no payload) emitted in `mousePressEvent` when a pan
drag begins, so `FractalView` can render and provide the cube background.

### `render_cube_to_qimage(cube_slice, colormap_fn) -> QImage`

Module-level function that renders a `CubeSlice` to a `QImage` using the given
winding colormap. Extracted from `display_basin_from_cube` for reuse by the pan
background path in `FractalView`.

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

### Pinned Trajectory Markers

When a trajectory is pinned in inspect mode, a colored X marker is drawn on
the canvas at the corresponding (theta1, theta2) location.

**Data**: `_pinned_markers: dict[str, tuple[float, float, tuple[int, int, int]]]`
maps row_id to `(theta1, theta2, color_rgb)`.

**Drawing**: Each marker is drawn as a two-layer X — a dark outline (3px wide)
drawn behind a colored fill (2px wide, using the basin colormap color).

**Highlight**: When the user hovers over a `TrajectoryIndicator` circle in the
inspect column, the corresponding marker grows larger (8px half-size vs 5px
normal) and the outline widens (4px vs 3px). Tracked by
`_highlighted_marker_id`.

**Color updates**: Marker colors update when the winding colormap changes,
synced through `update_marker_color()`.

**API**:
- `add_marker(row_id, theta1, theta2, color_rgb=(255,255,255))` — add marker
- `update_marker_color(row_id, color_rgb)` — change marker color
- `highlight_marker(row_id)` — enlarge marker on indicator hover
- `unhighlight_marker()` — restore normal size
- `remove_marker(row_id)` — remove by ID
- `clear_markers()` — remove all

## Pan/Zoom Mechanics

- **Zoom**: Mouse wheel centered on cursor (1.2× per notch)
- **Pan**: Left-click drag with live image offset (image tracks cursor
  before recomputation)
- **Debounce**: 300ms single-shot QTimer after last interaction, then
  constructs `FractalViewport` and emits `viewport_changed`
- During debounce: existing QImage stretched/translated for instant feedback
