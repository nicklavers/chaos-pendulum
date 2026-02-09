# Controls UI

Controls panel for fractal mode. File: `fractal/controls.py` (~368 lines).

> Cross-ref: [data-shapes.md](data-shapes.md) for signal payloads.
> Cross-ref: [inspect-tool.md](inspect-tool.md) for the inspect panel.

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

Dropdown for colormap selection (HSV, twilight, viridis, magma, etc.).
Emits `colormap_changed(str)`.

### Display Angle

Dropdown to select which angle is colored in the fractal: "θ₂ (bob 2)"
(default) or "θ₁ (bob 1)". Display-only — no recomputation needed.
Emits `angle_selection_changed(int)` with 0 (theta1) or 1 (theta2).

### Resolution

Dropdown for grid resolution (64, 128, 256, 512).
Emits `resolution_changed(int)`.

### Zoom Out Button

Emits `zoom_out_clicked`. Canvas handles the actual zoom-out logic.

### Physics Parameters

Collapsible section containing `PhysicsParamsWidget` (from `ui_common.py`).
Sliders for m1, m2, l1, l2, g. Note: "Changing physics will recompute the
fractal" — discourages casual adjustment of expensive parameters.

Emits `physics_changed` (no payload).

### Inspect Panel

`QGroupBox` (hidden by default, shown when inspect tool is active).
Contains two `PendulumDiagram` widgets side by side:
- Left: "Initial (t=0)" — pendulum at hovered (θ₁, θ₂)
- Right: "At t = X.X s" — pendulum at evolved state

Angle readout labels below each diagram: "θ₁ = X.XX, θ₂ = X.XX".

## Key Methods

- `get_params() -> DoublePendulumParams` — read physics sliders
- `set_params(params)` — update physics sliders
- `get_angle_index() -> int` — selected angle (0=theta1, 1=theta2)
- `get_time_index() -> float` — current slider position as sample index
- `get_t_end() -> float` — simulation duration
- `update_time_label(t_end)` — refresh "t = X.X s" display
- `update_inspect(theta1, theta2, theta1_at_t, theta2_at_t, t_value)` — update
  both diagrams and angle labels
- `set_inspect_params(params)` — pass physics params to diagram widgets
- `show_inspect_panel(visible)` — show/hide the inspect group box
