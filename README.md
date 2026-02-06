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

The coupled second-order ODEs are rewritten as four first-order ODEs by introducing omega1 and omega2 as independent state variables. This system is integrated using SciPy's `solve_ivp` with the RK45 (Dormand-Prince) adaptive step-size method, which adjusts the time step to maintain accuracy while keeping computation efficient.

### Why it's chaotic

The double pendulum is a Hamiltonian system -- total energy is conserved, and there is no dissipation. Despite being fully deterministic, the system exhibits sensitive dependence on initial conditions: two trajectories starting with an angular difference as small as 1e-9 radians will diverge exponentially, with the Lyapunov exponent characterizing the rate of separation. This is the hallmark of deterministic chaos, and the reason the long-term behavior of a double pendulum is effectively unpredictable.

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
