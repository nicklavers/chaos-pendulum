"""PyQt6 visualization for the double pendulum simulation.

Contains PendulumCanvas (QPainter rendering), ControlPanel (sliders and
playback controls), and MainWindow (wiring everything together).
"""

import itertools
import math
import random
from collections import deque

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QThread
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QComboBox, QGroupBox, QSplitter,
    QStatusBar, QCheckBox, QSpinBox,
)

from simulation import (
    DoublePendulumParams, simulate, positions, total_energy,
)


# ---------------------------------------------------------------------------
# PendulumCanvas
# ---------------------------------------------------------------------------

class PendulumCanvas(QWidget):
    """Custom widget that draws the double pendulum using QPainter."""

    TRAIL_LENGTH = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = DoublePendulumParams()
        self.states = [[0.0, 0.0, 0.0, 0.0]]
        self.trails = [deque(maxlen=self.TRAIL_LENGTH)]
        self.primary_index = 0
        self.show_axes = False
        self.show_velocity = False
        self.setMinimumSize(400, 400)

    @property
    def state(self):
        """Primary trajectory state (backwards compat)."""
        return self.states[self.primary_index]

    @property
    def trail(self):
        """Primary trajectory trail (backwards compat)."""
        return self.trails[self.primary_index]

    def set_states(self, states_list, params, append_trail=True):
        """Update all trajectory states on the canvas."""
        self.states = states_list
        self.params = params
        # Ensure we have the right number of trail deques
        while len(self.trails) < len(states_list):
            self.trails.append(deque(maxlen=self.TRAIL_LENGTH))
        if append_trail:
            for i, st in enumerate(states_list):
                _, _, x2, y2 = positions(st, params)
                self.trails[i].append((x2, y2))
        self.update()

    def set_state(self, state, params, append_trail=True):
        """Update single-trajectory state (backwards compat)."""
        self.set_states([state], params, append_trail)

    def clear_trails(self):
        self.trails = [deque(maxlen=self.TRAIL_LENGTH)]

    def clear_trail(self):
        self.clear_trails()

    def _to_pixel(self, x, y):
        """Convert physics coords to pixel coords.

        Physics: origin at pivot, y negative = up, positive = down.
        Pixel: origin at centre-top area of widget, y increases downward.
        """
        w, h = self.width(), self.height()
        total_length = self.params.l1 + self.params.l2
        # Leave a margin so the pendulum doesn't clip the edges
        scale = min(w, h) * 0.38 / max(total_length, 0.01)
        cx = w / 2
        cy = h * 0.35
        px = cx + x * scale
        py = cy - y * scale  # flip y: physics y-down becomes screen y-down
        return px, py

    def _draw_axes(self, painter):
        """Draw faint concentric distance rings and crosshairs from the pivot."""
        total_length = self.params.l1 + self.params.l2
        w, h = self.width(), self.height()
        scale = min(w, h) * 0.38 / max(total_length, 0.01)
        cx = w / 2
        cy = h * 0.35

        # Pick a nice step: 0.5m for short pendulums, 1m for longer
        step = 0.5 if total_length <= 3.0 else 1.0
        max_r = total_length + step  # one ring past max reach

        ring_pen = QPen(QColor(255, 255, 255, 30))
        ring_pen.setWidthF(1.0)
        ring_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(ring_pen)

        label_font = QFont()
        label_font.setPointSizeF(9)
        painter.setFont(label_font)

        r = step
        while r <= max_r:
            r_px = r * scale
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r_px, r_px)

            # Label on the right of each ring
            painter.setPen(QColor(255, 255, 255, 50))
            label = f"{r:.1f} m" if step < 1.0 else f"{r:.0f} m"
            painter.drawText(QPointF(cx + r_px + 3, cy - 2), label)
            r += step

        # Crosshair lines through pivot
        line_pen = QPen(QColor(255, 255, 255, 20))
        line_pen.setWidthF(1.0)
        painter.setPen(line_pen)
        painter.drawLine(QPointF(0, cy), QPointF(w, cy))
        painter.drawLine(QPointF(cx, 0), QPointF(cx, h))

    def _draw_arrow(self, painter, x0, y0, x1, y1, color):
        """Draw a single arrow from (x0,y0) to (x1,y1) in pixel coords."""
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 2:
            return
        pen = QPen(color)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # Arrowhead
        head_size = min(8, length * 0.35)
        angle = math.atan2(dy, dx)
        for sign in [1, -1]:
            a = angle + math.pi - sign * math.radians(25)
            hx = x1 + head_size * math.cos(a)
            hy = y1 + head_size * math.sin(a)
            painter.drawLine(QPointF(x1, y1), QPointF(hx, hy))

    def _draw_velocity_arrows(self, painter, bob1_px, bob2_px):
        """Draw velocity arrows at each bob showing initial angular velocities."""
        theta1, theta2, omega1, omega2 = self.state
        l1, l2 = self.params.l1, self.params.l2

        w, h = self.width(), self.height()
        total_length = l1 + l2
        scale = min(w, h) * 0.38 / max(total_length, 0.01)
        arrow_scale = scale * 0.12

        # Component from omega1: same for both bobs
        cvx1 = l1 * math.cos(theta1) * omega1
        cvy1 = l1 * math.sin(theta1) * omega1
        # Component from omega2: only affects bob2
        cvx2 = l2 * math.cos(theta2) * omega2
        cvy2 = l2 * math.sin(theta2) * omega2

        # To pixel deltas (flip y)
        dx1, dy1 = cvx1 * arrow_scale, -cvy1 * arrow_scale
        dx2, dy2 = cvx2 * arrow_scale, -cvy2 * arrow_scale

        bob1_color = QColor(255, 120, 80)
        bob2_color = QColor(80, 200, 255)
        proj_color = QColor(255, 220, 60)

        # --- Bob 1: single arrow (omega1 contribution only) ---
        bx1, by1 = bob1_px
        self._draw_arrow(painter, bx1, by1, bx1 + dx1, by1 + dy1, bob1_color)

        # --- Bob 2: decomposed velocity arrows ---
        bx2, by2 = bob2_px
        has_omega1 = math.hypot(dx1, dy1) >= 2
        has_omega2 = math.hypot(dx2, dy2) >= 2

        # Blue: omega2 contribution, rooted at bob2 (already tangential to arm l2)
        if has_omega2:
            self._draw_arrow(painter, bx2, by2, bx2 + dx2, by2 + dy2, bob2_color)

        # Orange: omega1 contribution, chained from tip of blue
        if has_omega1:
            self._draw_arrow(painter, bx2 + dx2, by2 + dy2,
                             bx2 + dx2 + dx1, by2 + dy2 + dy1, bob1_color)

        # Red: projection of omega1 contribution onto the tangential direction
        # (perpendicular to arm l2). omega2 is already tangential, so only
        # the omega1 component needs projecting.
        # Tangent direction in physics coords: (cos(theta2), sin(theta2))
        if has_omega1:
            proj_mag = cvx1 * math.cos(theta2) + cvy1 * math.sin(theta2)
            # In pixel coords (flip y for tangent)
            pdx = math.cos(theta2) * proj_mag * arrow_scale
            pdy = -math.sin(theta2) * proj_mag * arrow_scale

            if math.hypot(pdx, pdy) >= 2:
                # Red arrow from blue tip (same base as orange)
                base_x = bx2 + dx2
                base_y = by2 + dy2
                self._draw_arrow(painter, base_x, base_y,
                                 base_x + pdx, base_y + pdy, proj_color)

                # Dashed line from orange tip to red tip
                dash_pen = QPen(QColor(255, 255, 255, 80))
                dash_pen.setWidthF(1.0)
                dash_pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(QPointF(bx2 + dx2 + dx1, by2 + dy2 + dy1),
                                 QPointF(base_x + pdx, base_y + pdy))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(20, 20, 30))

        # --- Axes / distance rings ---
        if self.show_axes:
            self._draw_axes(painter)

        # Shared pivot point
        pivot_px = self._to_pixel(0, 0)

        # --- Secondary trajectories: translucent full pendulums ---
        if len(self.states) > 1:
            ghost_alpha = max(10, min(40, 800 // len(self.states)))
            bob_r1 = 6 + 4 * self.params.m1
            bob_r2 = 6 + 4 * self.params.m2
            arm_pen = QPen(QColor(200, 200, 200, ghost_alpha))
            arm_pen.setWidthF(2.0)
            for i, st in enumerate(self.states):
                if i == self.primary_index:
                    continue
                sx1, sy1, sx2, sy2 = positions(st, self.params)
                sb1 = self._to_pixel(sx1, sy1)
                sb2 = self._to_pixel(sx2, sy2)
                # Arms
                painter.setPen(arm_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(QPointF(*pivot_px), QPointF(*sb1))
                painter.drawLine(QPointF(*sb1), QPointF(*sb2))
                # Bobs
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(255, 120, 80, ghost_alpha)))
                painter.drawEllipse(QPointF(*sb1), bob_r1, bob_r1)
                painter.setBrush(QBrush(QColor(80, 200, 255, ghost_alpha)))
                painter.drawEllipse(QPointF(*sb2), bob_r2, bob_r2)

        # --- Primary trajectory ---
        primary = self.states[self.primary_index]
        x1, y1, x2, y2 = positions(primary, self.params)
        bob1_px = self._to_pixel(x1, y1)
        bob2_px = self._to_pixel(x2, y2)

        # --- Trail (primary only) ---
        primary_trail = self.trails[self.primary_index] if self.primary_index < len(self.trails) else deque()
        if len(primary_trail) > 1:
            trail_list = list(primary_trail)
            for i in range(1, len(trail_list)):
                alpha = int(255 * i / len(trail_list))
                pen = QPen(QColor(100, 200, 255, alpha))
                pen.setWidthF(1.5)
                painter.setPen(pen)
                px0, py0 = self._to_pixel(*trail_list[i - 1])
                px1, py1 = self._to_pixel(*trail_list[i])
                painter.drawLine(QPointF(px0, py0), QPointF(px1, py1))

        # --- Arms ---
        arm_pen = QPen(QColor(200, 200, 200))
        arm_pen.setWidthF(2.5)
        painter.setPen(arm_pen)
        painter.drawLine(QPointF(*pivot_px), QPointF(*bob1_px))
        painter.drawLine(QPointF(*bob1_px), QPointF(*bob2_px))

        # --- Pivot ---
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(180, 180, 180)))
        painter.drawEllipse(QPointF(*pivot_px), 5, 5)

        # --- Bobs ---
        bob_radius_1 = 6 + 4 * self.params.m1  # scale with mass
        bob_radius_2 = 6 + 4 * self.params.m2
        painter.setBrush(QBrush(QColor(255, 120, 80)))
        painter.drawEllipse(QPointF(*bob1_px), bob_radius_1, bob_radius_1)
        painter.setBrush(QBrush(QColor(80, 200, 255)))
        painter.drawEllipse(QPointF(*bob2_px), bob_radius_2, bob_radius_2)

        # --- Velocity arrows (initial conditions only, primary only) ---
        if self.show_velocity:
            self._draw_velocity_arrows(painter, bob1_px, bob2_px)

        painter.end()


# ---------------------------------------------------------------------------
# ControlPanel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    """Sliders for initial conditions, system parameters, and playback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True  # suppress signals during init
        self._init_ui()
        self._building = False

    # -- helpers for creating sliders --

    def _make_slider(self, minimum, maximum, value, resolution=100):
        """Create an integer QSlider that maps to float values.

        The slider range is [minimum*resolution, maximum*resolution].
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(minimum * resolution))
        slider.setMaximum(int(maximum * resolution))
        slider.setValue(int(value * resolution))
        slider.resolution = resolution  # stash for later
        return slider

    @staticmethod
    def _slider_value(slider):
        return slider.value() / slider.resolution

    def _make_log_slider(self, log_min, log_max, steps=1000):
        """Create a slider with log mapping. Position 0 = 0, 1..steps = log scale."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(steps)
        slider.setValue(0)
        slider.log_min = log_min
        slider.log_max = log_max
        slider.log_steps = steps
        slider.is_log = True
        return slider

    @staticmethod
    def _log_slider_value(slider):
        pos = slider.value()
        if pos == 0:
            return 0.0
        t = (pos - 1) / (slider.log_steps - 1)
        return slider.log_min * (slider.log_max / slider.log_min) ** t

    @staticmethod
    def _format_spread(value, unit):
        if value == 0:
            return f"0{unit}"
        if value < 0.01:
            return f"{value:.1e}{unit}"
        return f"{value:.3f}{unit}"

    def _add_param_row(self, layout, row, label_text, slider, unit=""):
        label = QLabel(label_text)
        value_label = QLabel()
        value_label.setMinimumWidth(55)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

        def _update(val, vl=value_label, sl=slider, u=unit):
            vl.setText(f"{self._slider_value(sl):.2f}{u}")
            if not self._building:
                self._on_param_changed()

        slider.valueChanged.connect(_update)
        _update(slider.value())
        return value_label

    def _add_spread_count_row(self, layout, row, spread_slider, count_spinbox, unit=""):
        """Add an indented spread slider + count spinbox row below an IC row."""
        spread_label = QLabel("  spread")
        spread_label.setStyleSheet("color: #888;")
        spread_value = QLabel()
        spread_value.setFixedWidth(80)
        spread_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        n_label = QLabel("n=")
        n_label.setStyleSheet("color: #888;")

        layout.addWidget(spread_label, row, 0)
        layout.addWidget(spread_slider, row, 1)
        layout.addWidget(spread_value, row, 2)
        layout.addWidget(n_label, row, 3)
        layout.addWidget(count_spinbox, row, 4)

        def _update_spread(val, vl=spread_value, sl=spread_slider, u=unit):
            if getattr(sl, 'is_log', False):
                v = self._log_slider_value(sl)
            else:
                v = self._slider_value(sl)
            vl.setText(self._format_spread(v, u))
            if not self._building:
                self._on_param_changed()

        spread_slider.valueChanged.connect(_update_spread)
        _update_spread(spread_slider.value())

        def _update_count(val):
            if not self._building:
                self._on_param_changed()

        count_spinbox.valueChanged.connect(_update_count)

    # -- UI construction --

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # --- Initial Conditions ---
        ic_group = QGroupBox("Initial Conditions")
        ic_layout = QGridLayout()
        ic_group.setLayout(ic_layout)

        self.theta1_slider = self._make_slider(-math.pi, math.pi, math.pi / 2)
        self.theta2_slider = self._make_slider(-math.pi, math.pi, math.pi / 2)
        self.omega1_slider = self._make_slider(-10, 10, 0)
        self.omega2_slider = self._make_slider(-10, 10, 0)

        # Spread sliders (log scale: 0 then 1e-10 .. max)
        self.theta1_spread = self._make_log_slider(1e-10, 2 * math.pi)
        self.theta2_spread = self._make_log_slider(1e-10, 2 * math.pi)
        self.omega1_spread = self._make_log_slider(1e-10, 20)
        self.omega2_spread = self._make_log_slider(1e-10, 20)

        # Count spinboxes
        self.theta1_count = QSpinBox(); self.theta1_count.setRange(1, 100); self.theta1_count.setValue(1)
        self.theta2_count = QSpinBox(); self.theta2_count.setRange(1, 100); self.theta2_count.setValue(1)
        self.omega1_count = QSpinBox(); self.omega1_count.setRange(1, 100); self.omega1_count.setValue(1)
        self.omega2_count = QSpinBox(); self.omega2_count.setRange(1, 100); self.omega2_count.setValue(1)

        # Row 0: theta1 center, Row 1: theta1 spread/count
        self._add_param_row(ic_layout, 0, "\u03b81\u2080", self.theta1_slider, " rad")
        self._add_spread_count_row(ic_layout, 1, self.theta1_spread, self.theta1_count, " rad")
        # Row 2: theta2 center, Row 3: theta2 spread/count
        self._add_param_row(ic_layout, 2, "\u03b82\u2080", self.theta2_slider, " rad")
        self._add_spread_count_row(ic_layout, 3, self.theta2_spread, self.theta2_count, " rad")
        # Row 4: omega1 center, Row 5: omega1 spread/count
        self._add_param_row(ic_layout, 4, "\u03c91\u2080", self.omega1_slider, " /s")
        self._add_spread_count_row(ic_layout, 5, self.omega1_spread, self.omega1_count, " /s")
        # Row 6: omega2 center, Row 7: omega2 spread/count
        self._add_param_row(ic_layout, 6, "\u03c92\u2080", self.omega2_slider, " /s")
        self._add_spread_count_row(ic_layout, 7, self.omega2_spread, self.omega2_count, " /s")

        main_layout.addWidget(ic_group)

        # --- System Parameters ---
        sys_group = QGroupBox("System Parameters")
        sys_layout = QGridLayout()
        sys_group.setLayout(sys_layout)

        self.m1_slider = self._make_slider(0.1, 5.0, 1.0)
        self.m2_slider = self._make_slider(0.1, 5.0, 1.0)
        self.l1_slider = self._make_slider(0.1, 3.0, 1.0)
        self.l2_slider = self._make_slider(0.1, 3.0, 1.0)

        self._add_param_row(sys_layout, 0, "m\u2081", self.m1_slider, " kg")
        self._add_param_row(sys_layout, 1, "m\u2082", self.m2_slider, " kg")
        self._add_param_row(sys_layout, 2, "l\u2081", self.l1_slider, " m")
        self._add_param_row(sys_layout, 3, "l\u2082", self.l2_slider, " m")

        main_layout.addWidget(sys_group)

        # --- Simulation ---
        sim_group = QGroupBox("Simulation")
        sim_layout = QGridLayout()
        sim_group.setLayout(sim_layout)

        self.t_end_slider = self._make_slider(5, 60, 30, resolution=1)
        self._add_param_row(sim_layout, 0, "Duration", self.t_end_slider, " s")

        # Max trajectories
        max_traj_label = QLabel("Max trajectories")
        self.max_trajectories_spin = QSpinBox()
        self.max_trajectories_spin.setRange(1, 5000)
        self.max_trajectories_spin.setValue(500)
        self.max_trajectories_spin.valueChanged.connect(
            lambda v: self._on_param_changed() if not self._building else None
        )
        sim_layout.addWidget(max_traj_label, 1, 0)
        sim_layout.addWidget(self.max_trajectories_spin, 1, 1)

        self.trajectory_count_label = QLabel("Trajectories: 1")
        self.trajectory_count_label.setStyleSheet("color: #aaa;")
        sim_layout.addWidget(self.trajectory_count_label, 2, 0, 1, 3)

        main_layout.addWidget(sim_group)

        # --- Timeline Scrubber ---
        tl_group = QGroupBox("Timeline")
        tl_layout = QVBoxLayout()
        tl_group.setLayout(tl_layout)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.setValue(0)
        self.timeline_label = QLabel("0.00 / 0.00 s")
        tl_layout.addWidget(self.timeline_slider)
        tl_layout.addWidget(self.timeline_label)

        main_layout.addWidget(tl_group)

        # --- Playback ---
        pb_group = QGroupBox("Playback")
        pb_layout = QHBoxLayout()
        pb_group.setLayout(pb_layout)

        self.play_btn = QPushButton("Play")
        self.reset_btn = QPushButton("Reset")
        self.speed_combo = QComboBox()
        for s in ["0.25x", "0.5x", "1x", "2x", "4x"]:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentIndex(2)  # default 1x

        self.axes_checkbox = QCheckBox("Show axes")

        pb_layout.addWidget(self.play_btn)
        pb_layout.addWidget(self.reset_btn)
        pb_layout.addWidget(QLabel("Speed:"))
        pb_layout.addWidget(self.speed_combo)
        pb_layout.addWidget(self.axes_checkbox)

        main_layout.addWidget(pb_group)
        main_layout.addStretch()

    # -- Public accessors --

    def get_initial_conditions(self):
        return (
            self._slider_value(self.theta1_slider),
            self._slider_value(self.theta2_slider),
            self._slider_value(self.omega1_slider),
            self._slider_value(self.omega2_slider),
        )

    def get_params(self):
        return DoublePendulumParams(
            m1=self._slider_value(self.m1_slider),
            m2=self._slider_value(self.m2_slider),
            l1=self._slider_value(self.l1_slider),
            l2=self._slider_value(self.l2_slider),
        )

    def get_spread_counts(self):
        """Return list of (spread, count) for each IC: theta1, theta2, omega1, omega2."""
        return [
            (self._log_slider_value(self.theta1_spread), self.theta1_count.value()),
            (self._log_slider_value(self.theta2_spread), self.theta2_count.value()),
            (self._log_slider_value(self.omega1_spread), self.omega1_count.value()),
            (self._log_slider_value(self.omega2_spread), self.omega2_count.value()),
        ]

    def get_max_trajectories(self):
        return self.max_trajectories_spin.value()

    def get_t_end(self):
        return self._slider_value(self.t_end_slider)

    def get_speed(self):
        text = self.speed_combo.currentText().replace("x", "")
        return float(text)

    # -- Callbacks (wired by MainWindow) --

    def _on_param_changed(self):
        """Called when any parameter slider changes. MainWindow overrides."""
        pass


# ---------------------------------------------------------------------------
# LoadingOverlay
# ---------------------------------------------------------------------------

class LoadingOverlay(QWidget):
    """Semi-transparent overlay with an orbital loading animation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.message = "Simulating..."
        self.t = 0.0
        self._timer = QTimer()
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self, message="Simulating..."):
        self.message = message
        self.t = 0.0
        if self.parentWidget():
            self.resize(self.parentWidget().size())
        self.show()
        self.raise_()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self.t += 0.016
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Dim background
        painter.fillRect(self.rect(), QColor(20, 20, 30, 180))

        cx, cy = w / 2, h / 2 - 15  # slightly above center for text room

        # --- Moon orbital parameters ---
        # Big moon: tighter orbit, steeper tilt
        big_orbit_a, big_orbit_b = 36, 22
        big_rot = math.radians(50)
        big_period = 2.0
        big_r = 7
        big_angle = 2 * math.pi * self.t / big_period

        big_lx = big_orbit_a * math.cos(big_angle)
        big_ly = big_orbit_b * math.sin(big_angle)
        big_sx = big_lx * math.cos(big_rot) - big_ly * math.sin(big_rot)
        big_sy = big_lx * math.sin(big_rot) + big_ly * math.cos(big_rot)
        big_behind = math.sin(big_angle) > 0

        # Small moon: wider orbit, nearly horizontal
        sm_orbit_a, sm_orbit_b = 52, 14
        sm_rot = math.radians(-5)
        sm_period = 3.2
        sm_r = 4
        sm_angle = 2 * math.pi * self.t / sm_period

        sm_lx = sm_orbit_a * math.cos(sm_angle)
        sm_ly = sm_orbit_b * math.sin(sm_angle)
        sm_sx = sm_lx * math.cos(sm_rot) - sm_ly * math.sin(sm_rot)
        sm_sy = sm_lx * math.sin(sm_rot) + sm_ly * math.cos(sm_rot)
        sm_behind = math.sin(sm_angle) > 0

        # Planet wobble (reacts to moon gravity)
        wobble_x = -(big_sx * 0.05 + sm_sx * 0.02)
        wobble_y = -(big_sy * 0.05 + sm_sy * 0.02)
        px, py = cx + wobble_x, cy + wobble_y

        # Planet geometry
        planet_r = 24
        ring_a, ring_b = 42, 10
        ring_angle_deg = 20  # cocked ring tilt

        ring_pen = QPen(QColor(190, 170, 110))
        ring_pen.setWidthF(5)
        ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        ring_rect = QRectF(-ring_a, -ring_b, 2 * ring_a, 2 * ring_b)

        # ---- Behind moons ----
        painter.setPen(Qt.PenStyle.NoPen)
        if big_behind:
            painter.setBrush(QBrush(QColor(160, 175, 195)))
            painter.drawEllipse(QPointF(px + big_sx, py + big_sy), big_r, big_r)
        if sm_behind:
            painter.setBrush(QBrush(QColor(170, 170, 180)))
            painter.drawEllipse(QPointF(px + sm_sx, py + sm_sy), sm_r, sm_r)

        # ---- Back ring arc (behind planet) ----
        painter.save()
        painter.translate(px, py)
        painter.rotate(ring_angle_deg)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(ring_rect, 0, 180 * 16)  # top half = behind
        painter.restore()

        # ---- Planet body ----
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(210, 160, 80)))
        painter.drawEllipse(QPointF(px, py), planet_r, planet_r)

        # Subtle bands for Saturn look
        clip = QPainterPath()
        clip.addEllipse(QPointF(px, py), planet_r, planet_r)
        painter.save()
        painter.setClipPath(clip)
        painter.setBrush(QBrush(QColor(190, 140, 60)))
        painter.drawRect(QRectF(px - planet_r - 1, py - 4, 2 * planet_r + 2, 7))
        painter.setBrush(QBrush(QColor(200, 155, 75)))
        painter.drawRect(QRectF(px - planet_r - 1, py + 8, 2 * planet_r + 2, 4))
        painter.restore()

        # ---- Front ring arc (in front of planet) ----
        painter.save()
        painter.translate(px, py)
        painter.rotate(ring_angle_deg)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(ring_rect, 180 * 16, 180 * 16)  # bottom half = in front
        painter.restore()

        # ---- Front moons ----
        painter.setPen(Qt.PenStyle.NoPen)
        if not big_behind:
            painter.setBrush(QBrush(QColor(160, 175, 195)))
            painter.drawEllipse(QPointF(px + big_sx, py + big_sy), big_r, big_r)
        if not sm_behind:
            painter.setBrush(QBrush(QColor(170, 170, 180)))
            painter.drawEllipse(QPointF(px + sm_sx, py + sm_sy), sm_r, sm_r)

        # ---- Text (bobs with amplified wobble) ----
        text_wobble_y = wobble_y * 3.0
        painter.setPen(QColor(255, 255, 255, 200))
        font = QFont("Chalkboard SE")
        font.setPointSizeF(16)
        font.setBold(True)
        painter.setFont(font)
        text_rect = QRectF(0, cy + 65 + text_wobble_y, w, 40)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self.message,
        )

        painter.end()


# ---------------------------------------------------------------------------
# SimWorker
# ---------------------------------------------------------------------------

class SimWorker(QThread):
    """Runs multi-trajectory simulation in a background thread."""

    def __init__(self, params, ic_list, t_end, dt):
        super().__init__()
        self.params = params
        self.ic_list = ic_list
        self.t_end = t_end
        self.dt = dt
        self.t_array = None
        self.state_arrays = []
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for theta1, theta2, omega1, omega2 in self.ic_list:
            if self._cancelled:
                return
            t, states = simulate(
                self.params, theta1, theta2, omega1, omega2, self.t_end, self.dt,
            )
            if self.t_array is None:
                self.t_array = t
            self.state_arrays.append(states)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Top-level window wiring simulation, canvas, and controls together."""

    FPS = 60
    DT = 0.005  # simulation time step

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chaos Pendulum")
        self.resize(1200, 750)

        # Widgets
        self.canvas = PendulumCanvas()
        self.controls = ControlPanel()

        # Layout via splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.time_label = QLabel()
        self.energy_label = QLabel()
        self.drift_label = QLabel()
        self.status_bar.addWidget(self.time_label)
        self.status_bar.addWidget(self.energy_label)
        self.status_bar.addWidget(self.drift_label)

        # Loading overlay (child of canvas so it covers the drawing area)
        self.loading_overlay = LoadingOverlay(self.canvas)

        # Simulation state
        self.t_array = None
        self.state_arrays = []  # list of (N, 4) arrays
        self.current_index = 0
        self.playing = False
        self.initial_energy = 0.0
        self._sim_stale = True  # True when params changed but not yet simulated
        self._sim_running = False
        self._sim_params_changed = False
        self._sim_on_complete = None
        self._sim_worker = None
        self._scrub_target = 0

        # Timer for animation
        self.timer = QTimer()
        self.timer.setInterval(int(1000 / self.FPS))
        self.timer.timeout.connect(self._on_timer)

        # Wire signals
        self.controls._on_param_changed = self._on_param_changed
        self.controls.play_btn.clicked.connect(self._toggle_play)
        self.controls.reset_btn.clicked.connect(self._reset)
        self.controls.timeline_slider.sliderMoved.connect(self._on_scrub)
        self.controls.timeline_slider.sliderPressed.connect(self._on_scrub_pressed)
        self.controls.axes_checkbox.toggled.connect(self._on_axes_toggled)

        # Show initial state preview
        self._update_preview()

    # -- Simulation --

    def _generate_initial_conditions(self):
        """Build list of IC tuples from center values, spreads, and counts."""
        centers = self.controls.get_initial_conditions()
        spread_counts = self.controls.get_spread_counts()
        max_traj = self.controls.get_max_trajectories()

        # Generate value arrays for each IC parameter
        value_sets = []
        for center, (spread, count) in zip(centers, spread_counts):
            if count == 1:
                value_sets.append([center])
            else:
                value_sets.append(
                    list(np.linspace(center - spread / 2, center + spread / 2, count))
                )

        full_product = list(itertools.product(*value_sets))
        total = len(full_product)

        # Ensure center IC is always first
        center_ic = tuple(centers)
        if center_ic in full_product:
            full_product.remove(center_ic)
        full_product.insert(0, center_ic)

        if total <= max_traj:
            ic_list = full_product
            sampled = False
        else:
            # Keep center (index 0), sample the rest
            rest = full_product[1:]
            sampled_rest = random.sample(rest, max_traj - 1)
            ic_list = [full_product[0]] + sampled_rest
            sampled = True

        # Update trajectory count label
        if sampled:
            self.controls.trajectory_count_label.setText(
                f"Trajectories: {max_traj} / {total} (sampled)"
            )
        else:
            self.controls.trajectory_count_label.setText(
                f"Trajectories: {total}"
            )

        return ic_list

    def _update_preview(self):
        """Show frame-0 positions from current ICs without running simulation."""
        ic_list = self._generate_initial_conditions()
        params = self.controls.get_params()

        # Each IC tuple is already the state at t=0 (with omega values)
        states_at_zero = [list(ic) for ic in ic_list]

        self.current_index = 0
        self.canvas.clear_trails()
        self.canvas.show_velocity = True
        self.canvas.set_states(states_at_zero, params, append_trail=False)

        # Reset timeline (max=0 prevents scrubbing before simulation)
        t_end = self.controls.get_t_end()
        self.controls.timeline_slider.setMaximum(0)
        self.controls.timeline_slider.setValue(0)
        self.controls.timeline_label.setText(f"0.00 / {t_end:.2f} s")

        # Energy at initial state
        self.initial_energy = total_energy(states_at_zero[0], params)
        energy = self.initial_energy
        self.time_label.setText("  t = 0.000 s  ")
        self.energy_label.setText(f"  E = {energy:.4f} J  ")
        self.drift_label.setText("  \u0394E = +0.000000 J  ")

        self._sim_stale = True

    def _start_simulation(self, on_complete=None):
        """Launch background simulation. Calls on_complete when done."""
        if self._sim_running:
            # Cancel previous run
            if self._sim_worker:
                self._sim_worker.cancel()
            return

        ic_list = self._generate_initial_conditions()
        params = self.controls.get_params()
        t_end = self.controls.get_t_end()

        self._sim_on_complete = on_complete
        self._sim_running = True
        self._sim_params_changed = False

        n = len(ic_list)
        msg = f"Simulating {n} trajectory..." if n == 1 else f"Simulating {n} trajectories..."
        self.controls.play_btn.setEnabled(False)
        self.loading_overlay.start(msg)

        self._sim_worker = SimWorker(params, ic_list, t_end, self.DT)
        self._sim_worker.finished.connect(self._on_simulation_finished)
        self._sim_worker.start()

    def _on_simulation_finished(self):
        """Handle completed (or cancelled) background simulation."""
        self.loading_overlay.stop()
        self.controls.play_btn.setEnabled(True)

        worker = self._sim_worker
        self._sim_worker = None
        self._sim_running = False

        if self._sim_params_changed or worker._cancelled:
            return  # results outdated, discard

        self.t_array = worker.t_array
        self.state_arrays = worker.state_arrays
        self.current_index = 0
        self.canvas.clear_trails()

        self.initial_energy = total_energy(self.state_arrays[0][0], self.controls.get_params())
        self.controls.timeline_slider.setMaximum(len(self.t_array) - 1)
        self.controls.timeline_slider.setValue(0)

        self._sim_stale = False

        if self._sim_on_complete:
            self._sim_on_complete()

    def _update_display(self):
        """Push current frame to the canvas and status bar."""
        params = self.controls.get_params()
        states_at_frame = [sa[self.current_index] for sa in self.state_arrays]
        primary_state = states_at_frame[0]
        t = self.t_array[self.current_index]
        t_end = self.t_array[-1]

        self.canvas.show_velocity = (self.current_index == 0 and not self.playing)
        self.canvas.set_states(states_at_frame, params)

        # Timeline label
        self.controls.timeline_label.setText(f"{t:.2f} / {t_end:.2f} s")

        # Status bar (primary trajectory only)
        energy = total_energy(primary_state, params)
        drift = energy - self.initial_energy
        self.time_label.setText(f"  t = {t:.3f} s  ")
        self.energy_label.setText(f"  E = {energy:.4f} J  ")
        self.drift_label.setText(f"  \u0394E = {drift:+.6f} J  ")

        # Keep timeline slider in sync (unless user is dragging it)
        if not self.controls.timeline_slider.isSliderDown():
            self.controls.timeline_slider.setValue(self.current_index)

    # -- Playback --

    def _start_playback(self):
        """Begin animation after simulation is ready."""
        self.playing = True
        self.timer.start()
        self.controls.play_btn.setText("Pause")

    def _toggle_play(self):
        if self._sim_running:
            return  # simulation in progress, ignore
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.controls.play_btn.setText("Play")
        elif self._sim_stale:
            self._start_simulation(on_complete=self._start_playback)
        else:
            self._start_playback()

    def _on_timer(self):
        """Advance the animation by the appropriate number of steps."""
        speed = self.controls.get_speed()
        steps = max(1, int(speed * self.FPS * self.DT / (1 / self.FPS)))
        # steps per frame = speed * (sim_time_per_real_second) / (frames_per_second ... )
        # Simpler: each real frame is 1/FPS seconds. At 1x we advance 1/FPS of sim-time.
        # Number of indices = speed / FPS / DT ... let's just do it directly:
        steps = max(1, round(speed / (self.FPS * self.DT)))

        self.current_index += steps
        if self.current_index >= len(self.t_array):
            self.current_index = len(self.t_array) - 1
            self._toggle_play()  # stop at end

        self._update_display()

    def _reset(self):
        """Reset to initial state."""
        if self.playing:
            self._toggle_play()
        self._update_preview()

    # -- Scrubbing --

    def _on_scrub_pressed(self):
        """Pause playback when user grabs the scrubber."""
        if self.playing:
            self._toggle_play()

    def _on_scrub(self, value):
        """Jump to the frame indicated by the timeline slider."""
        if self._sim_stale:
            self._scrub_target = value
            self._start_simulation(on_complete=self._do_scrub)
            return
        self._do_scrub_to(value)

    def _do_scrub(self):
        """Callback after simulation finishes for a pending scrub."""
        self._do_scrub_to(self._scrub_target)

    def _do_scrub_to(self, value):
        """Scrub to a specific frame (simulation must be ready)."""
        self.current_index = value
        # Rebuild primary trail up to current position
        self.canvas.clear_trails()
        while len(self.canvas.trails) < len(self.state_arrays):
            self.canvas.trails.append(deque(maxlen=PendulumCanvas.TRAIL_LENGTH))
        params = self.controls.get_params()
        trail_start = max(0, self.current_index - PendulumCanvas.TRAIL_LENGTH)
        for i in range(trail_start, self.current_index + 1):
            _, _, x2, y2 = positions(self.state_arrays[0][i], params)
            self.canvas.trails[0].append((x2, y2))
        self._update_display()

    # -- Axes toggle --

    def _on_axes_toggled(self, checked):
        self.canvas.show_axes = checked
        self.canvas.update()

    # -- Parameter changes --

    def _on_param_changed(self):
        """Called when any parameter or IC slider changes."""
        if self.playing:
            self._toggle_play()
        if self._sim_running:
            self._sim_params_changed = True
            if self._sim_worker:
                self._sim_worker.cancel()
        self._update_preview()
