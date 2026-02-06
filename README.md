# Chaos Pendulum

An interactive double pendulum simulation for exploring chaotic dynamics and sensitive dependence on initial conditions.

## What is a double pendulum?

A double pendulum is one of the simplest physical systems that exhibits chaotic behavior. Two rigid arms connected by frictionless pivots follow deterministic equations of motion, yet tiny differences in starting position lead to wildly divergent trajectories. This project lets you see that divergence in real time.

## Features

- Real-time simulation of double pendulum dynamics using Lagrangian mechanics
- Interactive controls for initial angles, angular velocities, masses, and arm lengths
- Phase space visualization (angle vs. angular velocity)
- Trace overlay showing the trajectory of the outer bob
- Side-by-side comparison of nearby initial conditions to demonstrate chaotic sensitivity
- Energy conservation monitoring

## Getting started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/nicklavers/chaos-pendulum.git
cd chaos-pendulum
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

## Physics

The simulation derives equations of motion from the double pendulum Lagrangian:

```
L = T - V
```

where kinetic energy T and potential energy V depend on the angles, angular velocities, masses, and lengths of both arms. The resulting coupled second-order ODEs are integrated numerically using SciPy's adaptive Runge-Kutta solver (`solve_ivp` with RK45).

## Project structure

```
chaos-pendulum/
  main.py              # Entry point
  simulation.py        # Double pendulum equations of motion and integrator
  visualization.py     # Real-time rendering and phase space plots
  requirements.txt     # Python dependencies
  README.md
```

## License

MIT
