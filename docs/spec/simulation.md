# Simulation Engine

Physics engine for the double pendulum. Lives in `simulation.py` (~119 lines).

> Cross-ref: [data-shapes.md](data-shapes.md) for `DoublePendulumParams`.

## Purpose

Provides the scalar (single-trajectory) physics used by pendulum mode and
the `positions()` helper used by the inspect tool's stick-figure diagrams.

The fractal mode uses a separate vectorized batch implementation
([fractal-compute.md](fractal-compute.md)) for performance. Both implement
the same equations (including friction damping) — a cross-validation test
ensures they stay in sync.

## Friction (Viscous Damping)

The `DoublePendulumParams.friction` parameter applies linear viscous damping
to both angular accelerations: `alpha -= friction * omega`. At `friction=0`
(default) the system is conservative (Hamiltonian). With `friction > 0`,
energy dissipates monotonically and trajectories settle toward equilibrium.

## Key Functions

### `derivatives(t, state, params) -> list[float]`

Computes `[dtheta1/dt, dtheta2/dt, domega1/dt, domega2/dt]` for a single
state vector `[theta1, theta2, omega1, omega2]`.

Used by `scipy.integrate.solve_ivp` (DOP853) in pendulum mode for
adaptive-step integration with high accuracy (energy conservation matters
for the single-trajectory display).

### `positions(state, params) -> (x1, y1, x2, y2)`

Converts `[theta1, theta2, ...]` to Cartesian positions of both bobs
relative to the pivot. Used by:
- Pendulum mode canvas for drawing the pendulum
- `fractal/pendulum_diagram.py` for the inspect tool stick figures

Coordinate convention: x increases right, y increases **down** from pivot
(gravity direction). theta=0 means the arm hangs straight down.

### `simulate(params, t_span, y0, ...) -> solution`

Runs `solve_ivp` with DOP853 method. Only used in pendulum mode.
Not used by fractal mode (too slow for batch trajectories).

### `total_energy(state, params) -> float`

Computes total mechanical energy (T + V) for a single state. Used for energy
conservation monitoring and cross-validated against the batch version in
`fractal/_numpy_backend.py`.

## Module Boundary

`simulation.py` is imported by both `pendulum/` and `fractal/` packages but
has **no dependencies on either**. It depends only on `scipy` and `numpy`.
