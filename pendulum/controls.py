"""Pendulum control panel: sliders for ICs, parameters, and playback.

Extracted from visualization.py ControlPanel without behavioral changes.
Uses PhysicsParamsWidget from ui_common for the shared physics parameters.
"""

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QHBoxLayout,
    QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
    QCheckBox, QSpinBox,
)

from simulation import DoublePendulumParams
from ui_common import make_slider, slider_value, PhysicsParamsWidget


class PendulumControls(QWidget):
    """Sliders for initial conditions, system parameters, and playback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True
        self._init_ui()
        self._building = False

    # -- helpers --

    @staticmethod
    def _make_log_slider(log_min, log_max, steps=1000):
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
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

        def _update(val, vl=value_label, sl=slider, u=unit):
            vl.setText(f"{slider_value(sl):.2f}{u}")
            if not self._building:
                self._on_param_changed()

        slider.valueChanged.connect(_update)
        _update(slider.value())
        return value_label

    def _add_spread_count_row(self, layout, row, spread_slider, count_spinbox, unit=""):
        spread_label = QLabel("  spread")
        spread_label.setStyleSheet("color: #888;")
        spread_value = QLabel()
        spread_value.setFixedWidth(80)
        spread_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

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
                v = slider_value(sl)
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

        self.theta1_slider = make_slider(-math.pi, math.pi, math.pi / 2)
        self.theta2_slider = make_slider(-math.pi, math.pi, math.pi / 2)
        self.omega1_slider = make_slider(-10, 10, 0)
        self.omega2_slider = make_slider(-10, 10, 0)

        self.theta1_spread = self._make_log_slider(1e-10, 2 * math.pi)
        self.theta2_spread = self._make_log_slider(1e-10, 2 * math.pi)
        self.omega1_spread = self._make_log_slider(1e-10, 20)
        self.omega2_spread = self._make_log_slider(1e-10, 20)

        self.theta1_count = QSpinBox()
        self.theta1_count.setRange(1, 100)
        self.theta1_count.setValue(1)
        self.theta2_count = QSpinBox()
        self.theta2_count.setRange(1, 100)
        self.theta2_count.setValue(1)
        self.omega1_count = QSpinBox()
        self.omega1_count.setRange(1, 100)
        self.omega1_count.setValue(1)
        self.omega2_count = QSpinBox()
        self.omega2_count.setRange(1, 100)
        self.omega2_count.setValue(1)

        self._add_param_row(ic_layout, 0, "\u03b81\u2080", self.theta1_slider, " rad")
        self._add_spread_count_row(ic_layout, 1, self.theta1_spread, self.theta1_count, " rad")
        self._add_param_row(ic_layout, 2, "\u03b82\u2080", self.theta2_slider, " rad")
        self._add_spread_count_row(ic_layout, 3, self.theta2_spread, self.theta2_count, " rad")
        self._add_param_row(ic_layout, 4, "\u03c91\u2080", self.omega1_slider, " /s")
        self._add_spread_count_row(ic_layout, 5, self.omega1_spread, self.omega1_count, " /s")
        self._add_param_row(ic_layout, 6, "\u03c92\u2080", self.omega2_slider, " /s")
        self._add_spread_count_row(ic_layout, 7, self.omega2_spread, self.omega2_count, " /s")

        main_layout.addWidget(ic_group)

        # --- System Parameters ---
        sys_group = QGroupBox("System Parameters")
        sys_layout = QVBoxLayout()
        sys_group.setLayout(sys_layout)

        self.physics_params = PhysicsParamsWidget()
        # Wire physics param changes to our _on_param_changed
        for sl in [
            self.physics_params.m1_slider,
            self.physics_params.m2_slider,
            self.physics_params.l1_slider,
            self.physics_params.l2_slider,
        ]:
            sl.valueChanged.connect(
                lambda _val: self._on_param_changed() if not self._building else None
            )
        sys_layout.addWidget(self.physics_params)

        main_layout.addWidget(sys_group)

        # --- Simulation ---
        sim_group = QGroupBox("Simulation")
        sim_layout = QGridLayout()
        sim_group.setLayout(sim_layout)

        self.t_end_slider = make_slider(5, 60, 30, resolution=1)
        self._add_param_row(sim_layout, 0, "Duration", self.t_end_slider, " s")

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
        self.speed_combo.setCurrentIndex(2)

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
            slider_value(self.theta1_slider),
            slider_value(self.theta2_slider),
            slider_value(self.omega1_slider),
            slider_value(self.omega2_slider),
        )

    def get_params(self):
        return self.physics_params.get_params()

    def set_params(self, params):
        """Set physics parameters from a DoublePendulumParams."""
        self.physics_params.set_params(params)

    def get_spread_counts(self):
        return [
            (self._log_slider_value(self.theta1_spread), self.theta1_count.value()),
            (self._log_slider_value(self.theta2_spread), self.theta2_count.value()),
            (self._log_slider_value(self.omega1_spread), self.omega1_count.value()),
            (self._log_slider_value(self.omega2_spread), self.omega2_count.value()),
        ]

    def get_max_trajectories(self):
        return self.max_trajectories_spin.value()

    def get_t_end(self):
        return slider_value(self.t_end_slider)

    def get_speed(self):
        text = self.speed_combo.currentText().replace("x", "")
        return float(text)

    def get_initial_theta1(self):
        return slider_value(self.theta1_slider)

    def get_initial_theta2(self):
        return slider_value(self.theta2_slider)

    # -- Callbacks (wired by PendulumView or AppWindow) --

    def _on_param_changed(self):
        """Called when any parameter slider changes. Override in parent."""
        pass
