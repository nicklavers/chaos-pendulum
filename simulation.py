"""Double pendulum physics engine.

Implements the Lagrangian equations of motion for a double pendulum
and provides numerical integration via SciPy's solve_ivp (RK45).
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class DoublePendulumParams:
    """Physical parameters of the double pendulum system."""

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81


def derivatives(t, state, params):
    """Compute the four first-order ODEs for the double pendulum.

    State vector: [theta1, theta2, omega1, omega2]
    Returns: [d_theta1/dt, d_theta2/dt, d_omega1/dt, d_omega2/dt]
    """
    theta1, theta2, omega1, omega2 = state
    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

    delta = theta1 - theta2
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    denom = m1 + m2 - m2 * cos_delta**2

    alpha1 = (
        -m2 * l1 * omega1**2 * sin_delta * cos_delta
        - m2 * l2 * omega2**2 * sin_delta
        - (m1 + m2) * g * np.sin(theta1)
        + m2 * g * np.sin(theta2) * cos_delta
    ) / (l1 * denom)

    alpha2 = (
        (m1 + m2) * l1 * omega1**2 * sin_delta
        + (m1 + m2) * g * np.sin(theta1) * cos_delta
        + m2 * l2 * omega2**2 * sin_delta * cos_delta
        - (m1 + m2) * g * np.sin(theta2)
    ) / (l2 * denom)

    return [omega1, omega2, alpha1, alpha2]


def simulate(params, theta1_0, theta2_0, omega1_0=0.0, omega2_0=0.0,
             t_end=30.0, dt=0.005):
    """Run a full simulation and return uniformly-spaced results.

    Returns:
        t_array: 1D array of time values at uniform dt spacing
        state_array: 2D array of shape (len(t_array), 4)
    """
    t_eval = np.arange(0, t_end, dt)
    y0 = [theta1_0, theta2_0, omega1_0, omega2_0]

    sol = solve_ivp(
        fun=lambda t, y: derivatives(t, y, params),
        t_span=(0, t_end),
        y0=y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-14,
        atol=1e-14,
    )

    return sol.t, sol.y.T  # shape: (n_steps, 4)


def positions(state, params):
    """Convert a single state to Cartesian coordinates.

    Returns (x1, y1, x2, y2) where y points downward from the pivot.
    """
    theta1, theta2 = state[0], state[1]
    l1, l2 = params.l1, params.l2

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    return x1, y1, x2, y2


def total_energy(state, params):
    """Compute total mechanical energy (T + V) for a single state.

    Potential energy is measured from the pivot point (y=0).
    """
    theta1, theta2, omega1, omega2 = state
    m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

    # Kinetic energy
    T = (
        0.5 * (m1 + m2) * l1**2 * omega1**2
        + 0.5 * m2 * l2**2 * omega2**2
        + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )

    # Potential energy (from pivot)
    V = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)

    return T + V
