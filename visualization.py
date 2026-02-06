"""PyQt6 visualization for the double pendulum simulation.

Contains PendulumCanvas (QPainter rendering), ControlPanel (sliders and
playback controls), and MainWindow (wiring everything together).
"""

import math
from collections import deque

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QComboBox, QGroupBox, QSplitter,
    QStatusBar,
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
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.trail = deque(maxlen=self.TRAIL_LENGTH)
        self.setMinimumSize(400, 400)

    def set_state(self, state, params, append_trail=True):
        """Update the state shown on the canvas."""
        self.state = state
        self.params = params
        if append_trail:
            _, _, x2, y2 = positions(state, params)
            self.trail.append((x2, y2))
        self.update()

    def clear_trail(self):
        self.trail.clear()

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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(20, 20, 30))

        x1, y1, x2, y2 = positions(self.state, self.params)
        pivot_px = self._to_pixel(0, 0)
        bob1_px = self._to_pixel(x1, y1)
        bob2_px = self._to_pixel(x2, y2)

        # --- Trail ---
        if len(self.trail) > 1:
            trail_list = list(self.trail)
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

        self._add_param_row(ic_layout, 0, "\u03b81\u2080", self.theta1_slider, " rad")
        self._add_param_row(ic_layout, 1, "\u03b82\u2080", self.theta2_slider, " rad")
        self._add_param_row(ic_layout, 2, "\u03c91\u2080", self.omega1_slider, " /s")
        self._add_param_row(ic_layout, 3, "\u03c92\u2080", self.omega2_slider, " /s")

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

        # --- Simulation Duration ---
        sim_group = QGroupBox("Simulation")
        sim_layout = QGridLayout()
        sim_group.setLayout(sim_layout)

        self.t_end_slider = self._make_slider(5, 60, 30, resolution=1)
        self._add_param_row(sim_layout, 0, "Duration", self.t_end_slider, " s")

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

        pb_layout.addWidget(self.play_btn)
        pb_layout.addWidget(self.reset_btn)
        pb_layout.addWidget(QLabel("Speed:"))
        pb_layout.addWidget(self.speed_combo)

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

        # Simulation state
        self.t_array = None
        self.state_array = None
        self.current_index = 0
        self.playing = False
        self.initial_energy = 0.0

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

        # Run initial simulation
        self._run_simulation()

    # -- Simulation --

    def _run_simulation(self):
        """(Re)compute the full trajectory with current parameters."""
        was_playing = self.playing
        if self.playing:
            self._toggle_play()

        params = self.controls.get_params()
        theta1_0, theta2_0, omega1_0, omega2_0 = self.controls.get_initial_conditions()
        t_end = self.controls.get_t_end()

        self.t_array, self.state_array = simulate(
            params, theta1_0, theta2_0, omega1_0, omega2_0, t_end, self.DT,
        )

        self.current_index = 0
        self.canvas.clear_trail()

        # Record initial energy for drift display
        self.initial_energy = total_energy(self.state_array[0], params)

        # Update timeline slider range
        self.controls.timeline_slider.setMaximum(len(self.t_array) - 1)
        self.controls.timeline_slider.setValue(0)

        self._update_display()

        if was_playing:
            self._toggle_play()

    def _update_display(self):
        """Push current frame to the canvas and status bar."""
        params = self.controls.get_params()
        state = self.state_array[self.current_index]
        t = self.t_array[self.current_index]
        t_end = self.t_array[-1]

        self.canvas.set_state(state, params)

        # Timeline label
        self.controls.timeline_label.setText(f"{t:.2f} / {t_end:.2f} s")

        # Status bar
        energy = total_energy(state, params)
        drift = energy - self.initial_energy
        self.time_label.setText(f"  t = {t:.3f} s  ")
        self.energy_label.setText(f"  E = {energy:.4f} J  ")
        self.drift_label.setText(f"  \u0394E = {drift:+.6f} J  ")

        # Keep timeline slider in sync (unless user is dragging it)
        if not self.controls.timeline_slider.isSliderDown():
            self.controls.timeline_slider.setValue(self.current_index)

    # -- Playback --

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.controls.play_btn.setText("Play")
        else:
            self.playing = True
            self.timer.start()
            self.controls.play_btn.setText("Pause")

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
        """Re-run simulation with current parameters."""
        self._run_simulation()

    # -- Scrubbing --

    def _on_scrub_pressed(self):
        """Pause playback when user grabs the scrubber."""
        if self.playing:
            self._toggle_play()

    def _on_scrub(self, value):
        """Jump to the frame indicated by the timeline slider."""
        self.current_index = value
        # Rebuild trail up to current position
        self.canvas.clear_trail()
        params = self.controls.get_params()
        trail_start = max(0, self.current_index - PendulumCanvas.TRAIL_LENGTH)
        for i in range(trail_start, self.current_index + 1):
            _, _, x2, y2 = positions(self.state_array[i], params)
            self.canvas.trail.append((x2, y2))
        self._update_display()

    # -- Parameter changes --

    def _on_param_changed(self):
        """Called when any parameter or IC slider changes."""
        self._run_simulation()
