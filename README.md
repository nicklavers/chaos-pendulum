# Chaos Pendulum

An interactive double pendulum simulator and fractal explorer for visualizing chaotic dynamics and sensitive dependence on initial conditions.

## What is a double pendulum?

A double pendulum is one of the simplest physical systems that exhibits chaotic behavior. Two rigid arms connected by frictionless pivots follow deterministic equations of motion, yet tiny differences in starting position lead to wildly divergent trajectories. This project lets you see that divergence in real time.

## Modes

The application has two modes, accessible via the sidebar toggle:

### Pendulum Mode

Animate individual double pendulum trajectories with real-time rendering:

- Interactive controls for initial angles, angular velocities, masses, and arm lengths
- Phase space visualization (angle vs. angular velocity)
- Trace overlay showing the trajectory of the outer bob
- Side-by-side comparison of nearby initial conditions
- Energy conservation monitoring

### Fractal Explorer

Visualize how pendulum behavior varies across a grid of initial conditions. Each pixel represents a different starting (theta1, theta2) pair; color encodes the evolved state at a chosen time.

- **Bivariate display** (default): colors both angles simultaneously using torus colormaps that respect the 2pi-periodic topology of the state space
- **Univariate display**: colors a single angle via 1D LUT colormaps (HSV hue wheel, Twilight)
- Pan, zoom, and scrub through time with smooth animation
- Inspect tool: hover over any pixel to see the pendulum configuration at that initial condition
- Progressive rendering at multiple resolution levels (64, 128, 256, 512)
- LRU cache for previously computed zoom levels

#### Torus colormaps

The fractal explorer includes 9 torus colormaps for bivariate display. The default, **RGB Aligned + YBGM**, uses a landmark-aligned RGB sinusoid formula that places physically meaningful colors at key state-space positions:

- **Black** at (0, 0) -- both bobs hanging straight down
- **White** at (pi, 0) -- bob 1 horizontal, bob 2 down
- **Red** at (pi, pi) -- both bobs horizontal
- **Cyan** at (0, pi) -- bob 1 down, bob 2 horizontal
- **Yellow, Blue, Magenta, Green** at the four diagonal half-pi points

Other torus colormaps offer different visual tradeoffs, from clean checkerboard patterns (Yellow/Blue, Green/Magenta) to full 16-landmark coverage (6-Color) to hue-based and warm/cool schemes.

## Getting started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/nicklavers/chaos-pendulum.git
cd chaos-pendulum
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: install [Numba](https://numba.pydata.org/) for faster fractal computation (~5x speedup):

```bash
pip install numba
```

### Run

```bash
python main.py
```

### Tests

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v
```

## Physics

### The system

The double pendulum consists of two point masses m1 and m2 attached to rigid, massless rods of lengths l1 and l2. The first rod hangs from a fixed pivot; the second hangs from the end of the first. The state of the system is fully described by four variables: the angles of the two arms from vertical (theta1, theta2) and their angular velocities (omega1, omega2).

### Lagrangian formulation

Rather than tracking forces directly, we derive equations of motion from the Lagrangian L = T - V, where T is kinetic energy and V is gravitational potential energy.

The position of each mass in Cartesian coordinates:

```
x1 = l1 * sin(theta1)
y1 = -l1 * cos(theta1)

x2 = l1 * sin(theta1) + l2 * sin(theta2)
y2 = -l1 * cos(theta1) - l2 * cos(theta2)
```

The kinetic energy of the system:

```
T = (1/2) * m1 * (x1_dot^2 + y1_dot^2)
  + (1/2) * m2 * (x2_dot^2 + y2_dot^2)
```

which expands to:

```
T = (1/2) * (m1 + m2) * l1^2 * omega1^2
  + (1/2) * m2 * l2^2 * omega2^2
  + m2 * l1 * l2 * omega1 * omega2 * cos(theta1 - theta2)
```

The potential energy (measuring from the pivot):

```
V = -(m1 + m2) * g * l1 * cos(theta1) - m2 * g * l2 * cos(theta2)
```

### Equations of motion

Applying the Euler-Lagrange equations d/dt(dL/d(omega)) - dL/d(theta) = 0 to each angle yields two coupled second-order ODEs. Solving for the angular accelerations:

```
alpha1 = [-m2 * l1 * omega1^2 * sin(delta) * cos(delta)
           - m2 * l2 * omega2^2 * sin(delta)
           - (m1 + m2) * g * sin(theta1)
           + m2 * g * sin(theta2) * cos(delta)]
          / [l1 * (m1 + m2 - m2 * cos^2(delta))]

alpha2 = [(m1 + m2) * l1 * omega1^2 * sin(delta)
           + (m1 + m2) * g * sin(theta1) * cos(delta)
           - m2 * l2 * omega2^2 * sin(delta) * cos(delta)
           - (m1 + m2) * g * sin(theta2)]
          / [l2 * (m1 + m2 - m2 * cos^2(delta))]
```

where `delta = theta1 - theta2` and `alpha` denotes angular acceleration.

### Numerical integration

The coupled second-order ODEs are rewritten as four first-order ODEs by introducing omega1 and omega2 as independent state variables.

- **Pendulum mode**: integrated using SciPy's `solve_ivp` with the DOP853 (8th-order Dormand-Prince) adaptive step-size method.
- **Fractal mode**: uses a custom vectorized RK4 integrator that advances thousands of trajectories in parallel via NumPy broadcasting. An optional Numba JIT backend provides ~5x additional speedup.

### Why it's chaotic

The double pendulum is a Hamiltonian system -- total energy is conserved, and there is no dissipation. Despite being fully deterministic, the system exhibits sensitive dependence on initial conditions: two trajectories starting with an angular difference as small as 1e-9 radians will diverge exponentially, with the Lyapunov exponent characterizing the rate of separation. This is the hallmark of deterministic chaos, and the reason the long-term behavior of a double pendulum is effectively unpredictable.

## Project structure

```
chaos-pendulum/
    main.py                  Entry point, creates AppWindow
    simulation.py            Physics engine (Lagrangian equations of motion)
    app_window.py            QStackedWidget, mode switching, param sync
    ui_common.py             Shared widgets (LoadingOverlay, PhysicsParamsWidget)

    pendulum/                Pendulum animation mode
        canvas.py            Real-time pendulum rendering
        controls.py          Pendulum control panel
        view.py              Orchestration and simulation worker

    fractal/                 Fractal explorer mode
        canvas.py            FractalCanvas: QImage, pan/zoom, axes, legends
        controls.py          Time slider, colormap, resolution, inspect panel
        view.py              FractalView: orchestration, signal wiring
        bivariate.py         Torus colormaps for dual-angle display
        coloring.py          Univariate coloring (HSV LUT, angle-to-ARGB)
        compute.py           ComputeBackend protocol, grid builder
        _numpy_backend.py    Vectorized NumPy RK4
        _numba_backend.py    Numba JIT parallel RK4 (optional)
        cache.py             LRU fractal cache with memory tracking
        worker.py            QThread worker for background computation
        pendulum_diagram.py  Stick-figure pendulum widget (inspect tool)

    tests/                   pytest test suite (209 tests)
    docs/                    Living specs and architecture decision records
    requirements.txt         Python dependencies (numpy, scipy, PyQt6)
```

## References

### Lagrangian derivation

- [Diego Assencio -- Double pendulum: Lagrangian formulation](https://dassencio.org/33)
- [LSU Physics (Gabriela Gonzalez) -- Double Pendulum lecture notes (PDF)](https://www.phys.lsu.edu/faculty/gonzalez/Teaching/Phys7221/DoublePendulum.pdf)
- [Wikipedia -- Double pendulum](https://en.wikipedia.org/wiki/Double_pendulum)

### Equations of motion

- [myPhysicsLab -- Double Pendulum](https://www.myphysicslab.com/pendulum/double-pendulum-en.html)
- [Eric Weisstein's World of Physics -- Double Pendulum](https://scienceworld.wolfram.com/physics/DoublePendulum.html)
- [UC Berkeley -- The Double Pendulum](https://rotations.berkeley.edu/the-double-pendulum/)

### Chaos and Lyapunov exponents

- [Shinbrot et al. -- "Chaos in a double pendulum" (American Journal of Physics, 1992)](https://yorke.umd.edu/Yorke_papers_most_cited_and_post2000/1992-04-Wisdom_Shinbrot_AmerJPhys_double_pendulum.pdf)
- [SciELO -- "Deterministic chaos: A pedagogical review of the double pendulum case"](https://www.scielo.br/j/rbef/a/SsWk5qnzBgvmYB4hRtkbwqM/?lang=en)
- [Kyle Monette -- "Double Pendulum: Lagrangian Mechanics and Chaos" (PDF)](https://kylemonette.github.io/files/mccnny-2022.pdf)

### Numerical methods

- [SciPython -- The double pendulum](https://scipython.com/blog/the-double-pendulum/)

## License

MIT
