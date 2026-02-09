"""App window: QStackedWidget with toolbar for mode switching.

Hosts both PendulumView and FractalView, handling parameter sync
at mode-switch time and the Ctrl+click bridge from fractal to pendulum.
"""

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QToolBar, QStatusBar, QLabel,
)

from pendulum.view import PendulumView
from fractal.view import FractalView

logger = logging.getLogger(__name__)


class AppWindow(QMainWindow):
    """Top-level window with mode switching between pendulum and fractal."""

    # Mode indices
    PENDULUM_MODE = 0
    FRACTAL_MODE = 1

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chaos Pendulum")
        self.resize(1200, 750)

        # --- Views ---
        self.pendulum_view = PendulumView()
        self.fractal_view = FractalView()

        # --- Stacked widget ---
        self.stack = QStackedWidget()
        self.stack.addWidget(self.pendulum_view)  # index 0
        self.stack.addWidget(self.fractal_view)    # index 1
        self.setCentralWidget(self.stack)

        # --- Toolbar ---
        toolbar = QToolBar("Mode")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._mode_group = QActionGroup(self)
        self._mode_group.setExclusive(True)

        self._pendulum_action = QAction("Pendulum", self)
        self._pendulum_action.setCheckable(True)
        self._pendulum_action.setChecked(True)
        self._pendulum_action.triggered.connect(
            lambda: self._switch_mode(self.PENDULUM_MODE)
        )
        self._mode_group.addAction(self._pendulum_action)
        toolbar.addAction(self._pendulum_action)

        self._fractal_action = QAction("Fractal Explorer", self)
        self._fractal_action.setCheckable(True)
        self._fractal_action.triggered.connect(
            lambda: self._switch_mode(self.FRACTAL_MODE)
        )
        self._mode_group.addAction(self._fractal_action)
        toolbar.addAction(self._fractal_action)

        # --- Status bar ---
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        # Pendulum status labels
        self._status_bar.addWidget(self.pendulum_view.time_label)
        self._status_bar.addWidget(self.pendulum_view.energy_label)
        self._status_bar.addWidget(self.pendulum_view.drift_label)

        # Fractal status labels
        self._fractal_coord_label = QLabel()
        self._fractal_res_label = QLabel()
        self._fractal_cache_label = QLabel()
        self._fractal_backend_label = QLabel()

        backend_name = type(self.fractal_view._backend).__name__
        self._fractal_backend_label.setText(f"  Backend: {backend_name}  ")

        self._status_bar.addWidget(self._fractal_coord_label)
        self._status_bar.addWidget(self._fractal_res_label)
        self._status_bar.addWidget(self._fractal_cache_label)
        self._status_bar.addWidget(self._fractal_backend_label)

        # Show appropriate status labels
        self._update_status_visibility()

        # --- Ctrl+click bridge ---
        self.fractal_view.canvas.ic_selected.connect(self._on_ic_selected)

        # --- Fractal canvas hover tracking ---
        self.fractal_view.canvas.setMouseTracking(True)
        # We'll update the coord label periodically via a paint event override
        # or more simply, connect the canvas mouseMoveEvent indirectly
        self._setup_hover_tracking()

    def _setup_hover_tracking(self):
        """Set up periodic hover coordinate updates."""
        from PyQt6.QtCore import QTimer
        self._hover_timer = QTimer()
        self._hover_timer.setInterval(100)  # 10 Hz updates
        self._hover_timer.timeout.connect(self._update_hover_coords)
        self._hover_timer.start()

    def _update_hover_coords(self):
        """Update the fractal coordinate label from canvas hover state."""
        if self.stack.currentIndex() != self.FRACTAL_MODE:
            return

        canvas = self.fractal_view.canvas
        t1 = canvas.hover_theta1
        t2 = canvas.hover_theta2
        if t1 is not None and t2 is not None:
            self._fractal_coord_label.setText(
                f"  \u03b81={t1:.3f}  \u03b82={t2:.3f}  "
            )

        # Update cache info
        cache = self.fractal_view._cache
        self._fractal_cache_label.setText(
            f"  Cache: {cache.memory_used_mb:.1f} MB  "
        )

    def _update_status_visibility(self):
        """Show/hide status labels based on current mode."""
        is_pendulum = self.stack.currentIndex() == self.PENDULUM_MODE

        self.pendulum_view.time_label.setVisible(is_pendulum)
        self.pendulum_view.energy_label.setVisible(is_pendulum)
        self.pendulum_view.drift_label.setVisible(is_pendulum)

        self._fractal_coord_label.setVisible(not is_pendulum)
        self._fractal_res_label.setVisible(not is_pendulum)
        self._fractal_cache_label.setVisible(not is_pendulum)
        self._fractal_backend_label.setVisible(not is_pendulum)

    def _switch_mode(self, mode: int) -> None:
        """Switch between pendulum and fractal modes with param sync."""
        current = self.stack.currentIndex()
        if current == mode:
            return

        if current == self.PENDULUM_MODE and mode == self.FRACTAL_MODE:
            # Pendulum -> Fractal
            self.pendulum_view.deactivate()
            params = self.pendulum_view.get_params()
            theta1 = self.pendulum_view.get_initial_theta1()
            theta2 = self.pendulum_view.get_initial_theta2()
            self.stack.setCurrentIndex(self.FRACTAL_MODE)
            self.fractal_view.activate(params, theta1, theta2)

        elif current == self.FRACTAL_MODE and mode == self.PENDULUM_MODE:
            # Fractal -> Pendulum
            self.fractal_view.deactivate()
            params = self.fractal_view.get_params()
            self.stack.setCurrentIndex(self.PENDULUM_MODE)
            self.pendulum_view.activate(params=params)

        self._update_status_visibility()
        logger.info("Switched to %s mode", "pendulum" if mode == 0 else "fractal")

    def _on_ic_selected(self, theta1: float, theta2: float) -> None:
        """Handle Ctrl+click bridge from fractal to pendulum mode."""
        logger.info(
            "IC selected from fractal: theta1=%.3f, theta2=%.3f",
            theta1, theta2,
        )
        params = self.fractal_view.get_params()
        self.fractal_view.deactivate()
        self._pendulum_action.setChecked(True)
        self.stack.setCurrentIndex(self.PENDULUM_MODE)
        self.pendulum_view.activate(params=params, theta1=theta1, theta2=theta2)
        self._update_status_visibility()
