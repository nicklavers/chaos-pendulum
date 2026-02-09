# Inspect Tool

Third cursor tool mode for the fractal explorer. Hover over the fractal to see
pendulum stick-figure diagrams at initial and evolved states.

> Cross-ref: [canvas-rendering.md](canvas-rendering.md) for tool mode architecture.
> Cross-ref: [data-shapes.md](data-shapes.md) for snapshot shape.

## Components

### PendulumDiagram (fractal/pendulum_diagram.py, ~133 lines)

Small `QWidget` that draws a double-pendulum stick figure at given angles.

- `set_state(theta1, theta2)` — update angles, trigger repaint
- `set_params(params)` — update physics params (arm lengths affect drawing)
- `set_label(text)` — label above diagram (e.g. "t = 0", "t = 12.5 s")
- Uses `simulation.positions()` for coordinate computation
- Colors: bob1 orange (255,120,80), bob2 cyan (80,200,255), arms white, pivot grey
- Minimum size: 120×120

### Inspect Panel (in fractal/controls.py)

`QGroupBox` containing two `PendulumDiagram` widgets side by side:
- Left: initial state (θ₁, θ₂ at t=0)
- Right: evolved state (θ₁(t), θ₂(t) at current slider time)

Angle readout labels below each diagram.
Hidden by default; shown when inspect tool is active.

## Data Flow

```
mouseMoveEvent (inspect mode, cursor within image area)
    |
    v
FractalCanvas.hover_updated(theta1, theta2)     # physics coords under cursor
    |
    v
FractalView._on_hover_updated(theta1, theta2)
    |
    |-- Convert physics coords to grid indices:
    |     nx = (theta1 - vmin1) / span1
    |     ny = (theta2 - vmin2) / span2
    |     col = clamp(round(nx * (res-1)), 0, res-1)
    |     row = clamp(round(ny * (res-1)), 0, res-1)
    |     flat_idx = row * res + col
    |
    |-- Look up theta1 at time t:
    |     interpolate_theta2(snapshots[:, 0, :], time_index)[flat_idx]
    |
    |-- Look up theta2 at time t:
    |     interpolate_theta2(snapshots[:, 1, :], time_index)[flat_idx]
    |
    |-- Compute t_value from slider position and t_end
    |
    v
FractalControls.update_inspect(
    theta1_init, theta2_init,     # initial angles (hovered position)
    theta1_at_t, theta2_at_t,    # evolved angles at current time
    t_value                       # time in seconds
)
    |
    v
Update both PendulumDiagram widgets and angle labels
```

## Interaction

- Inspect mode: cursor changes to `PointingHandCursor`
- Hover updates diagrams in real-time
- No click action (Ctrl+click still works for IC selection across all modes)
- Tool mode controlled by `QButtonGroup` in controls
- Inspect panel visibility toggles with tool mode
