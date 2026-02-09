"""Pendulum view: orchestrates simulation, canvas, and controls.

Extracted from visualization.py MainWindow's pendulum-specific logic.
This is a QWidget suitable for embedding in a QStackedWidget.
"""

import itertools
import random
from collections import deque

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter, QLabel

from simulation import (
    DoublePendulumParams, simulate, positions, total_energy,
)
from pendulum.canvas import PendulumCanvas
from pendulum.controls import PendulumControls
from ui_common import LoadingOverlay


# ---------------------------------------------------------------------------
# SimWorker
# ---------------------------------------------------------------------------

class PendulumSimWorker(QThread):
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
                self.params, theta1, theta2, omega1, omega2,
                self.t_end, self.dt,
            )
            if self.t_array is None:
                self.t_array = t
            self.state_arrays.append(states)


# ---------------------------------------------------------------------------
# PendulumView
# ---------------------------------------------------------------------------

class PendulumView(QWidget):
    """Complete pendulum mode: canvas + controls + simulation wiring."""

    FPS = 60
    DT = 0.005

    def __init__(self, parent=None):
        super().__init__(parent)

        self.canvas = PendulumCanvas()
        self.controls = PendulumControls()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        # Loading overlay
        self.loading_overlay = LoadingOverlay(self.canvas)

        # Status bar labels (AppWindow will place these in a real status bar)
        self.time_label = QLabel()
        self.energy_label = QLabel()
        self.drift_label = QLabel()

        # Simulation state
        self.t_array = None
        self.state_arrays = []
        self.current_index = 0
        self.playing = False
        self.initial_energy = 0.0
        self._sim_stale = True
        self._sim_running = False
        self._sim_params_changed = False
        self._sim_on_complete = None
        self._sim_worker = None
        self._scrub_target = 0

        # Timer
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

        self._update_preview()

    # -- Public interface for mode switching --

    def activate(self, params=None, theta1=None, theta2=None):
        """Called when switching to pendulum mode."""
        if params is not None:
            self.controls.set_params(params)
        if theta1 is not None:
            self.controls.theta1_slider.setValue(
                int(theta1 * self.controls.theta1_slider.resolution)
            )
        if theta2 is not None:
            self.controls.theta2_slider.setValue(
                int(theta2 * self.controls.theta2_slider.resolution)
            )
        if params is not None or theta1 is not None or theta2 is not None:
            self._update_preview()

    def deactivate(self):
        """Called when switching away from pendulum mode."""
        if self.playing:
            self._toggle_play()

    def get_params(self):
        return self.controls.get_params()

    def get_initial_theta1(self):
        return self.controls.get_initial_theta1()

    def get_initial_theta2(self):
        return self.controls.get_initial_theta2()

    def set_initial_conditions(self, theta1, theta2):
        """Set ICs and trigger re-render. Called from Ctrl+click bridge."""
        self.controls.theta1_slider.setValue(
            int(theta1 * self.controls.theta1_slider.resolution)
        )
        self.controls.theta2_slider.setValue(
            int(theta2 * self.controls.theta2_slider.resolution)
        )
        self._update_preview()

    # -- Simulation --

    def _generate_initial_conditions(self):
        centers = self.controls.get_initial_conditions()
        spread_counts = self.controls.get_spread_counts()
        max_traj = self.controls.get_max_trajectories()

        value_sets = []
        for center, (spread, count) in zip(centers, spread_counts):
            if count == 1:
                value_sets.append([center])
            else:
                value_sets.append(
                    list(np.linspace(center - spread / 2,
                                     center + spread / 2, count))
                )

        full_product = list(itertools.product(*value_sets))
        total = len(full_product)

        center_ic = tuple(centers)
        if center_ic in full_product:
            full_product.remove(center_ic)
        full_product.insert(0, center_ic)

        if total <= max_traj:
            ic_list = full_product
            sampled = False
        else:
            rest = full_product[1:]
            sampled_rest = random.sample(rest, max_traj - 1)
            ic_list = [full_product[0]] + sampled_rest
            sampled = True

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
        ic_list = self._generate_initial_conditions()
        params = self.controls.get_params()
        states_at_zero = [list(ic) for ic in ic_list]

        self.current_index = 0
        self.canvas.clear_trails()
        self.canvas.show_velocity = True
        self.canvas.set_states(states_at_zero, params, append_trail=False)

        t_end = self.controls.get_t_end()
        self.controls.timeline_slider.setMaximum(0)
        self.controls.timeline_slider.setValue(0)
        self.controls.timeline_label.setText(f"0.00 / {t_end:.2f} s")

        self.initial_energy = total_energy(states_at_zero[0], params)
        energy = self.initial_energy
        self.time_label.setText("  t = 0.000 s  ")
        self.energy_label.setText(f"  E = {energy:.4f} J  ")
        self.drift_label.setText("  \u0394E = +0.000000 J  ")

        self._sim_stale = True

    def _start_simulation(self, on_complete=None):
        if self._sim_running:
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
        msg = (
            f"Simulating {n} trajectory..."
            if n == 1
            else f"Simulating {n} trajectories..."
        )
        self.controls.play_btn.setEnabled(False)
        self.loading_overlay.start(msg)

        self._sim_worker = PendulumSimWorker(params, ic_list, t_end, self.DT)
        self._sim_worker.finished.connect(self._on_simulation_finished)
        self._sim_worker.start()

    def _on_simulation_finished(self):
        self.loading_overlay.stop()
        self.controls.play_btn.setEnabled(True)

        worker = self._sim_worker
        self._sim_worker = None
        self._sim_running = False

        if self._sim_params_changed or worker._cancelled:
            return

        self.t_array = worker.t_array
        self.state_arrays = worker.state_arrays
        self.current_index = 0
        self.canvas.clear_trails()

        self.initial_energy = total_energy(
            self.state_arrays[0][0], self.controls.get_params()
        )
        self.controls.timeline_slider.setMaximum(len(self.t_array) - 1)
        self.controls.timeline_slider.setValue(0)

        self._sim_stale = False

        if self._sim_on_complete:
            self._sim_on_complete()

    def _update_display(self):
        params = self.controls.get_params()
        states_at_frame = [sa[self.current_index] for sa in self.state_arrays]
        primary_state = states_at_frame[0]
        t = self.t_array[self.current_index]
        t_end = self.t_array[-1]

        self.canvas.show_velocity = (
            self.current_index == 0 and not self.playing
        )
        self.canvas.set_states(states_at_frame, params)

        self.controls.timeline_label.setText(f"{t:.2f} / {t_end:.2f} s")

        energy = total_energy(primary_state, params)
        drift = energy - self.initial_energy
        self.time_label.setText(f"  t = {t:.3f} s  ")
        self.energy_label.setText(f"  E = {energy:.4f} J  ")
        self.drift_label.setText(f"  \u0394E = {drift:+.6f} J  ")

        if not self.controls.timeline_slider.isSliderDown():
            self.controls.timeline_slider.setValue(self.current_index)

    # -- Playback --

    def _start_playback(self):
        self.playing = True
        self.timer.start()
        self.controls.play_btn.setText("Pause")

    def _toggle_play(self):
        if self._sim_running:
            return
        if self.playing:
            self.playing = False
            self.timer.stop()
            self.controls.play_btn.setText("Play")
        elif self._sim_stale:
            self._start_simulation(on_complete=self._start_playback)
        else:
            self._start_playback()

    def _on_timer(self):
        speed = self.controls.get_speed()
        steps = max(1, round(speed / (self.FPS * self.DT)))

        self.current_index += steps
        if self.current_index >= len(self.t_array):
            self.current_index = len(self.t_array) - 1
            self._toggle_play()

        self._update_display()

    def _reset(self):
        if self.playing:
            self._toggle_play()
        self._update_preview()

    # -- Scrubbing --

    def _on_scrub_pressed(self):
        if self.playing:
            self._toggle_play()

    def _on_scrub(self, value):
        if self._sim_stale:
            self._scrub_target = value
            self._start_simulation(on_complete=self._do_scrub)
            return
        self._do_scrub_to(value)

    def _do_scrub(self):
        self._do_scrub_to(self._scrub_target)

    def _do_scrub_to(self, value):
        self.current_index = value
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
        if self.playing:
            self._toggle_play()
        if self._sim_running:
            self._sim_params_changed = True
            if self._sim_worker:
                self._sim_worker.cancel()
        self._update_preview()
