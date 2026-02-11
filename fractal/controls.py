"""Fractal controls: time slider, animation, colormap, resolution, physics params.

All controls for the fractal explorer mode, organized in grouped sections.
"""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
    QButtonGroup,
)

from fractal.compute import DEFAULT_N_SAMPLES
from fractal.coloring import COLORMAPS
from fractal.bivariate import TORUS_COLORMAPS
from fractal.winding import WINDING_COLORMAPS
from ui_common import PhysicsParamsWidget, slider_value


class FractalControls(QWidget):
    """Control panel for the fractal explorer mode."""

    # Signals
    time_index_changed = pyqtSignal(float)
    colormap_changed = pyqtSignal(str)
    resolution_changed = pyqtSignal(int)
    physics_changed = pyqtSignal()
    t_end_changed = pyqtSignal()
    zoom_out_clicked = pyqtSignal()
    tool_mode_changed = pyqtSignal(str)  # "zoom", "pan", or "inspect"
    angle_selection_changed = pyqtSignal(int)  # 0 = theta1, 1 = theta2, 2 = both
    torus_colormap_changed = pyqtSignal(str)
    display_mode_changed = pyqtSignal(str)        # "angle" or "basin"
    winding_colormap_changed = pyqtSignal(str)    # winding colormap name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True
        self._init_ui()
        self._building = False

        # Apply default display mode (basin) UI state
        self._apply_display_mode("basin")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # --- Time Slider ---
        self._time_group = QGroupBox("Time")
        time_group = self._time_group
        time_layout = QVBoxLayout()
        time_group.setLayout(time_layout)

        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum((DEFAULT_N_SAMPLES - 1) * 10)  # 10x for smooth scrub
        self.time_slider.setValue(0)
        self.time_slider.setTickPosition(QSlider.TickPosition.NoTicks)

        self.time_label = QLabel("t = 0.0 s")

        # Animation controls
        anim_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.speed_combo = QComboBox()
        for s in ["0.5x", "1x", "2x", "4x"]:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentIndex(1)  # default 1x

        anim_layout.addWidget(self.play_btn)
        anim_layout.addWidget(QLabel("Speed:"))
        anim_layout.addWidget(self.speed_combo)
        anim_layout.addStretch()

        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(self.time_label)
        time_layout.addLayout(anim_layout)

        main_layout.addWidget(time_group)

        # --- Navigation ---
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()
        nav_group.setLayout(nav_layout)

        # Tool mode toggle
        tool_row = QHBoxLayout()

        self.zoom_tool_btn = QPushButton("Zoom")
        self.zoom_tool_btn.setCheckable(True)
        self.zoom_tool_btn.setChecked(True)

        self.pan_tool_btn = QPushButton("Pan")
        self.pan_tool_btn.setCheckable(True)

        self.inspect_tool_btn = QPushButton("Inspect")
        self.inspect_tool_btn.setCheckable(True)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_group.addButton(self.zoom_tool_btn)
        self._tool_group.addButton(self.pan_tool_btn)
        self._tool_group.addButton(self.inspect_tool_btn)

        tool_row.addWidget(self.zoom_tool_btn)
        tool_row.addWidget(self.pan_tool_btn)
        tool_row.addWidget(self.inspect_tool_btn)
        tool_row.addStretch()

        self.zoom_out_btn = QPushButton("Zoom Out (2\u00d7)")
        tool_row.addWidget(self.zoom_out_btn)

        nav_layout.addLayout(tool_row)

        self.nav_hint = QLabel("Drag to zoom in")
        self.nav_hint.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        nav_layout.addWidget(self.nav_hint)

        main_layout.addWidget(nav_group)

        # --- Display ---
        display_group = QGroupBox("Display")
        display_layout = QGridLayout()
        display_group.setLayout(display_layout)

        # Display mode toggle: Angle vs Basin
        display_layout.addWidget(QLabel("Mode:"), 0, 0)
        mode_row = QHBoxLayout()
        self._angle_mode_btn = QPushButton("Angle")
        self._angle_mode_btn.setCheckable(True)
        self._basin_mode_btn = QPushButton("Basin")
        self._basin_mode_btn.setCheckable(True)
        self._basin_mode_btn.setChecked(True)

        self._display_mode_group = QButtonGroup(self)
        self._display_mode_group.setExclusive(True)
        self._display_mode_group.addButton(self._angle_mode_btn)
        self._display_mode_group.addButton(self._basin_mode_btn)

        mode_row.addWidget(self._angle_mode_btn)
        mode_row.addWidget(self._basin_mode_btn)
        mode_row.addStretch()

        mode_container = QWidget()
        mode_container.setLayout(mode_row)
        display_layout.addWidget(mode_container, 0, 1)

        display_layout.addWidget(QLabel("Colormap:"), 1, 0)
        self.colormap_combo = QComboBox()
        # Placeholder; _apply_display_mode() repopulates for the active mode
        for name in TORUS_COLORMAPS:
            self.colormap_combo.addItem(name)
        display_layout.addWidget(self.colormap_combo, 1, 1)

        display_layout.addWidget(QLabel("Resolution:"), 2, 0)
        self.resolution_combo = QComboBox()
        for res in [64, 128, 256, 512]:
            self.resolution_combo.addItem(f"{res}x{res}", res)
        self.resolution_combo.setCurrentIndex(2)  # default 256x256
        display_layout.addWidget(self.resolution_combo, 2, 1)

        self._angle_label = QLabel("Display angle:")
        display_layout.addWidget(self._angle_label, 3, 0)
        self.angle_combo = QComboBox()
        self.angle_combo.addItem("\u03b8\u2082 (bob 2)", 1)
        self.angle_combo.addItem("\u03b8\u2081 (bob 1)", 0)
        self.angle_combo.addItem("Both (\u03b8\u2081, \u03b8\u2082)", 2)
        self.angle_combo.setCurrentIndex(2)  # default: Both
        display_layout.addWidget(self.angle_combo, 3, 1)

        main_layout.addWidget(display_group)

        # --- Simulation ---
        self._sim_group = QGroupBox("Simulation")
        sim_group = self._sim_group
        sim_layout = QGridLayout()
        sim_group.setLayout(sim_layout)

        sim_layout.addWidget(QLabel("Duration:"), 0, 0)
        self.t_end_slider = QSlider(Qt.Orientation.Horizontal)
        self.t_end_slider.setMinimum(5)
        self.t_end_slider.setMaximum(60)
        self.t_end_slider.setValue(30)
        self.t_end_label = QLabel("30 s")
        sim_layout.addWidget(self.t_end_slider, 0, 1)
        sim_layout.addWidget(self.t_end_label, 0, 2)

        main_layout.addWidget(sim_group)

        # --- Physics Parameters (collapsed) ---
        physics_group = QGroupBox("Physics Parameters")
        physics_layout = QVBoxLayout()
        physics_group.setLayout(physics_layout)

        self.physics_warning = QLabel(
            "Changing physics will recompute the fractal"
        )
        self.physics_warning.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        physics_layout.addWidget(self.physics_warning)

        self.physics_params = PhysicsParamsWidget()
        physics_layout.addWidget(self.physics_params)

        main_layout.addWidget(physics_group)
        main_layout.addStretch()

        # --- Wire signals ---
        self.time_slider.valueChanged.connect(self._on_time_slider_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.angle_combo.currentIndexChanged.connect(self._on_angle_changed)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.t_end_slider.valueChanged.connect(self._on_t_end_changed)

        # Physics param changes
        for sl in [
            self.physics_params.m1_slider,
            self.physics_params.m2_slider,
            self.physics_params.l1_slider,
            self.physics_params.l2_slider,
            self.physics_params.friction_slider,
        ]:
            sl.valueChanged.connect(self._on_physics_changed)

        # Tool mode toggle
        self._tool_group.buttonClicked.connect(self._on_tool_changed)

        # Display mode toggle
        self._display_mode_group.buttonClicked.connect(self._on_display_mode_changed)

        # Zoom out button
        self.zoom_out_btn.clicked.connect(self._on_zoom_out_clicked)

        # Animation timer
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)  # ~30 fps for animation
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self.play_btn.toggled.connect(self._on_play_toggled)

    # -- Public accessors --

    def get_time_index(self) -> float:
        """Return the current time index as a float in [0, n_samples-1]."""
        return self.time_slider.value() / 10.0

    def get_t_end(self) -> float:
        return float(self.t_end_slider.value())

    def get_angle_index(self) -> int:
        """Return the currently selected angle index (0=theta1, 1=theta2, 2=both)."""
        return self.angle_combo.currentData()

    def get_resolution(self) -> int:
        return self.resolution_combo.currentData()

    def get_params(self):
        return self.physics_params.get_params()

    def set_params(self, params):
        self.physics_params.set_params(params)

    def get_speed(self) -> float:
        text = self.speed_combo.currentText().replace("x", "")
        return float(text)

    def update_time_label(self, t_end: float) -> None:
        """Update the time label based on current slider position."""
        frac = self.time_slider.value() / max(1, self.time_slider.maximum())
        t = frac * t_end
        self.time_label.setText(f"t = {t:.1f} s")

    def get_display_mode(self) -> str:
        """Return 'angle' or 'basin'."""
        if self._basin_mode_btn.isChecked():
            return "basin"
        return "angle"

    def set_friction(self, value: float) -> None:
        """Programmatically set the friction slider value."""
        self.physics_params.friction_slider.setValue(
            int(value * self.physics_params.friction_slider.resolution)
        )

    def get_friction(self) -> float:
        """Read current friction value from the slider."""
        return slider_value(self.physics_params.friction_slider)

    # -- Callbacks --

    def _on_time_slider_changed(self, value):
        time_index = value / 10.0
        self.time_index_changed.emit(time_index)
        t_end = self.get_t_end()
        self.update_time_label(t_end)

        # Scrubbing pauses animation
        if self.play_btn.isChecked() and self.time_slider.isSliderDown():
            self._anim_timer.stop()

    def _on_colormap_changed(self, name):
        if self._building:
            return
        if self.get_display_mode() == "basin":
            self.winding_colormap_changed.emit(name)
        elif self.angle_combo.currentData() == 2:
            self.torus_colormap_changed.emit(name)
        else:
            self.colormap_changed.emit(name)

    def _on_angle_changed(self, _index):
        if self._building:
            return
        angle_index = self.angle_combo.currentData()
        self.angle_selection_changed.emit(angle_index)
        self._update_colormap_options(angle_index)

    def _update_colormap_options(self, angle_index: int) -> None:
        """Swap colormap combo contents based on univariate vs bivariate mode."""
        self._building = True
        current_text = self.colormap_combo.currentText()
        self.colormap_combo.clear()

        if angle_index == 2:
            for name in TORUS_COLORMAPS:
                self.colormap_combo.addItem(name)
        else:
            for name in COLORMAPS:
                self.colormap_combo.addItem(name)

        # Try to restore previous selection
        idx = self.colormap_combo.findText(current_text)
        if idx >= 0:
            self.colormap_combo.setCurrentIndex(idx)

        self._building = False

    def _on_resolution_changed(self, _index):
        if not self._building:
            self.resolution_changed.emit(self.get_resolution())

    def _on_physics_changed(self, _value):
        if not self._building:
            self.physics_changed.emit()

    def _on_tool_changed(self, button):
        if button is self.zoom_tool_btn:
            self.tool_mode_changed.emit("zoom")
            self.nav_hint.setText("Drag to zoom in")
        elif button is self.pan_tool_btn:
            self.tool_mode_changed.emit("pan")
            self.nav_hint.setText("Drag to pan")
        elif button is self.inspect_tool_btn:
            self.tool_mode_changed.emit("inspect")
            self.nav_hint.setText("Hover to inspect, click to pin")

    def _on_zoom_out_clicked(self):
        self.zoom_out_clicked.emit()

    def _on_display_mode_changed(self, button):
        """Handle Angle/Basin toggle click."""
        if self._building:
            return
        mode = "basin" if button is self._basin_mode_btn else "angle"
        self._apply_display_mode(mode)
        self.display_mode_changed.emit(mode)

    def _apply_display_mode(self, mode: str) -> None:
        """Show/hide UI elements based on display mode."""
        self._building = True

        if mode == "basin":
            # Hide time group (no scrubbing in basin mode)
            self._time_group.setVisible(False)
            # Hide simulation group (t_end auto-computed)
            self._sim_group.setVisible(False)
            # Hide angle combo (irrelevant in basin mode)
            self.angle_combo.setVisible(False)
            self._angle_label.setVisible(False)
            # Auto-set friction to 0.38 if currently zero
            if self.get_friction() < 0.01:
                self.set_friction(0.38)
            # Swap colormap dropdown to winding colormaps
            current_text = self.colormap_combo.currentText()
            self.colormap_combo.clear()
            for name in WINDING_COLORMAPS:
                self.colormap_combo.addItem(name)
            idx = self.colormap_combo.findText(current_text)
            if idx >= 0:
                self.colormap_combo.setCurrentIndex(idx)
            else:
                # Default to Basin Hash on first switch to basin mode
                basin_idx = self.colormap_combo.findText("Basin Hash")
                if basin_idx >= 0:
                    self.colormap_combo.setCurrentIndex(basin_idx)
        else:
            # Show time group and simulation group
            self._time_group.setVisible(True)
            self._sim_group.setVisible(True)
            # Show angle combo
            self.angle_combo.setVisible(True)
            self._angle_label.setVisible(True)
            # Swap colormap dropdown back based on angle selection
            self._update_colormap_options(self.angle_combo.currentData())

        self._building = False

    def _on_t_end_changed(self, value):
        self.t_end_label.setText(f"{value} s")
        if not self._building:
            self.t_end_changed.emit()

    # -- Animation --

    def _on_play_toggled(self, checked):
        if checked:
            self.play_btn.setText("Pause")
            self._anim_timer.start()
        else:
            self.play_btn.setText("Play")
            self._anim_timer.stop()

    def _on_anim_tick(self):
        """Advance time slider by speed-adjusted increment."""
        speed = self.get_speed()
        # Advance by speed * 1 sample per ~1 second of real time
        # At 30fps timer, each tick is ~33ms = 0.033s
        increment = speed * 0.033 * 10  # 10 = slider resolution factor
        new_val = self.time_slider.value() + int(max(1, increment))

        if new_val > self.time_slider.maximum():
            new_val = 0  # loop

        self.time_slider.setValue(new_val)
