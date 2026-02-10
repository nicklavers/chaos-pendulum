# Controls UI

Controls panel for fractal mode. File: `fractal/controls.py` (~512 lines).

> Cross-ref: [data-shapes.md](data-shapes.md) for signal payloads.
> Cross-ref: [inspect-tool.md](inspect-tool.md) for the inspect panel.
> Cross-ref: [coloring-pipeline.md](coloring-pipeline.md) for colormap registries.

## FractalControls(QWidget)

Right-side panel in the fractal mode splitter. Contains all user-adjustable
parameters and tool selection.

## Layout (top to bottom)

### Tool Selection

`QButtonGroup` with three exclusive toggle buttons:
- **Zoom** — default, rectangle zoom
- **Pan** — drag to pan
- **Inspect** — hover to inspect pendulum states

Emits `tool_mode_changed(str)` with "zoom", "pan", or "inspect".

Status hint label below buttons updates per mode (e.g. "Draw rectangle to zoom",
"Hover to inspect").

### Time Controls

- **Time slider**: `QSlider` spanning `[0, 1000]` mapped to `[0, n_samples-1]`
- **Time label**: "t = X.X s" updated during scrubbing
- **t_end spinner**: simulation end time (changing invalidates all cache)

Emits `time_index_changed(float)`.

### Colormap

Dropdown for colormap selection. Contents change dynamically based on angle mode:
- **Univariate** (theta1 or theta2): shows `COLORMAPS` entries (HSV, Twilight)
- **Bivariate** (both): shows `TORUS_COLORMAPS` entries (9 torus colormaps)

The `_update_colormap_options()` method swaps dropdown items when the angle
selection changes, using the `_building` guard to prevent spurious signal
emissions during the swap.

Emits `colormap_changed(str)` in univariate mode,
`torus_colormap_changed(str)` in bivariate mode.

### Display Angle

Dropdown to select which angle is colored in the fractal:
- "θ₂ (bob 2)" (data=1)
- "θ₁ (bob 1)" (data=0)
- "Both (θ₁, θ₂)" (default, data=2) — enables bivariate torus colormaps

Default is "Both" with the "RGB Aligned + YBGM" torus colormap pre-selected.
The colormap combo is initially populated with torus colormaps to match.

Display-only — no recomputation needed, only recoloring.
Emits `angle_selection_changed(int)` with 0, 1, or 2.

### Resolution

Dropdown for grid resolution (64, 128, 256, 512).
Emits `resolution_changed(int)`.

### Zoom Out Button

Emits `zoom_out_clicked`. Canvas handles the actual zoom-out logic.

### Display Mode Toggle

Toggle button to switch between angle mode and basin mode:
- **Angle mode** (default): time-scrubbing display, shows evolved angle at chosen time
- **Basin mode**: damped simulation, shows final winding numbers with friction-based coloring

When basin mode is active, time controls are hidden and the colormap dropdown
switches to winding colormaps. A friction slider appears for adjusting the
damping coefficient.

Emits `display_mode_changed(str)` with "angle" or "basin".

### Physics Parameters

Collapsible section containing `PhysicsParamsWidget` (from `ui_common.py`).
Sliders for m1, m2, l1, l2, g, and friction. Note: "Changing physics will
recompute the fractal" — discourages casual adjustment of expensive parameters.

Emits `physics_changed` (no payload).

### Inspect Panel

`QGroupBox` (hidden by default, shown when inspect tool is active).
Contains two `PendulumDiagram` widgets side by side:
- Left: "Initial (t=0)" — pendulum at hovered (θ₁, θ₂)
- Right: "At t = X.X s" — pendulum at evolved state

Angle readout labels below each diagram: "θ₁ = X.XX, θ₂ = X.XX".

## Signals

| Signal | Payload | When emitted |
|--------|---------|--------------|
| `time_index_changed` | `float` | Time slider moved |
| `colormap_changed` | `str` | Colormap dropdown changed (univariate mode) |
| `torus_colormap_changed` | `str` | Colormap dropdown changed (bivariate mode) |
| `resolution_changed` | `int` | Resolution dropdown changed |
| `physics_changed` | (none) | Any physics slider moved |
| `t_end_changed` | (none) | Duration slider moved |
| `zoom_out_clicked` | (none) | Zoom out button clicked |
| `tool_mode_changed` | `str` | Tool button toggled ("zoom", "pan", "inspect") |
| `angle_selection_changed` | `int` | Angle dropdown changed (0, 1, or 2) |

## Key Methods

- `get_params() -> DoublePendulumParams` — read physics sliders
- `set_params(params)` — update physics sliders
- `get_angle_index() -> int` — selected angle (0=theta1, 1=theta2, 2=both)
- `get_time_index() -> float` — current slider position as sample index
- `get_t_end() -> float` — simulation duration
- `update_time_label(t_end)` — refresh "t = X.X s" display
- `update_inspect(theta1, theta2, theta1_at_t, theta2_at_t, t_value)` — update
  both diagrams and angle labels
- `set_inspect_params(params)` — pass physics params to diagram widgets
- `show_inspect_panel(visible)` — show/hide the inspect group box
