"""Fractal controls: time slider, animation, colormap, resolution, physics params.

All controls for the fractal explorer mode, organized in grouped sections.
"""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
    QButtonGroup,
)

from fractal.data_cube import DEFAULT_GRID
from fractal.winding import WINDING_COLORMAPS
from ui_common import PhysicsParamsWidget, slider_value, set_slider_value


class FractalControls(QWidget):
    """Control panel for the fractal explorer mode."""

    # Signals
    resolution_changed = pyqtSignal(int)
    physics_changed = pyqtSignal()
    physics_released = pyqtSignal()  # slider handle released (mouse-up)
    zoom_out_clicked = pyqtSignal()
    tool_mode_changed = pyqtSignal(str)  # "zoom", "pan", or "inspect"
    winding_colormap_changed = pyqtSignal(str)    # winding colormap name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._building = True
        self._init_ui()
        self._building = False

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

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

        display_layout.addWidget(QLabel("Colormap:"), 0, 0)
        self.colormap_combo = QComboBox()
        for name in WINDING_COLORMAPS:
            self.colormap_combo.addItem(name)
        display_layout.addWidget(self.colormap_combo, 0, 1)

        display_layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        for res in [64, 128, 256, 512]:
            self.resolution_combo.addItem(f"{res}x{res}", res)
        self.resolution_combo.setCurrentIndex(2)  # default 256x256
        display_layout.addWidget(self.resolution_combo, 1, 1)

        main_layout.addWidget(display_group)

        # --- Physics Parameters ---
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

        snap_values = {
            "m1": DEFAULT_GRID.m1_values,
            "m2": DEFAULT_GRID.m2_values,
            "l1": DEFAULT_GRID.l1_values,
            "l2": DEFAULT_GRID.l2_values,
            "friction": DEFAULT_GRID.friction_values,
        }
        self.physics_params = PhysicsParamsWidget(snap_values=snap_values)
        physics_layout.addWidget(self.physics_params)

        main_layout.addWidget(physics_group)
        main_layout.addStretch()

        # --- Wire signals ---
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)

        # Physics param changes
        for sl in [
            self.physics_params.m1_slider,
            self.physics_params.m2_slider,
            self.physics_params.l1_slider,
            self.physics_params.l2_slider,
            self.physics_params.friction_slider,
        ]:
            sl.valueChanged.connect(self._on_physics_changed)
            sl.sliderReleased.connect(self._on_physics_released)

        # Tool mode toggle
        self._tool_group.buttonClicked.connect(self._on_tool_changed)

        # Zoom out button
        self.zoom_out_btn.clicked.connect(self._on_zoom_out_clicked)

    # -- Public accessors --

    def get_resolution(self) -> int:
        return self.resolution_combo.currentData()

    def get_params(self):
        return self.physics_params.get_params()

    def set_params(self, params):
        self.physics_params.set_params(params)

    def set_friction(self, value: float) -> None:
        """Programmatically set the friction slider value."""
        set_slider_value(self.physics_params.friction_slider, value)

    def get_friction(self) -> float:
        """Read current friction value from the slider."""
        return slider_value(self.physics_params.friction_slider)

    # -- Callbacks --

    def _on_colormap_changed(self, name):
        if self._building:
            return
        self.winding_colormap_changed.emit(name)

    def _on_resolution_changed(self, _index):
        if not self._building:
            self.resolution_changed.emit(self.get_resolution())

    def _on_physics_changed(self, _value):
        if not self._building:
            self.physics_changed.emit()

    def _on_physics_released(self):
        if not self._building:
            self.physics_released.emit()

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
