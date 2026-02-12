"""InspectColumn: hover display + stacked multi-trajectory animation.

Shown to the left of the fractal canvas when the inspect tool is active.
Contains a hover display section at the top, a single stacked animation
widget showing all pinned trajectories, and a row of clickable indicator
circles below.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QFrame, QSlider,
)

from simulation import DoublePendulumParams
from fractal.pendulum_diagram import PendulumDiagram
from fractal.winding_circle import WindingCircle
from fractal.animated_diagram import MultiTrajectoryDiagram, TrajectoryInfo
from fractal.trajectory_indicator import TrajectoryIndicator
from fractal.winding import get_single_winding_color, WINDING_COLORMAPS


# Animation timer interval (ms) — ~30fps
ANIMATION_INTERVAL_MS = 33

# Subsampling: show every Nth step to keep animations a reasonable speed
# At dt=0.01 and 30fps, raw playback is 0.33x real-time. Skip frames to
# show ~2x real-time for snappy visual feedback.
FRAME_SUBSAMPLE = 6


@dataclass(frozen=True)
class PinnedTrajectory:
    """Immutable data for a single pinned trajectory.

    Attributes:
        row_id: Unique identifier.
        theta1_init: Initial theta1 angle (radians).
        theta2_init: Initial theta2 angle (radians).
        trajectory: (n_steps, 4) float32 subsampled state array.
        n1: Winding number for theta1.
        n2: Winding number for theta2.
        color_rgb: Basin color as (R, G, B) tuple.
    """

    row_id: str
    theta1_init: float
    theta2_init: float
    trajectory: np.ndarray
    n1: int
    n2: int
    color_rgb: tuple[int, int, int]


class InspectColumn(QWidget):
    """Stacked animation column for inspect mode.

    Layout:
        - Header row: "Pinned Trajectories" + "Clear All"
        - Hover section: initial diagram + winding circle (basin) or at-t (angle)
        - MultiTrajectoryDiagram: single stacked animation widget
        - Indicator row: horizontal row of TrajectoryIndicator circles

    Signals:
        row_removed(row_id): Emitted when an indicator X is clicked.
        all_cleared(): Emitted when Clear All is clicked.
    """

    row_removed = pyqtSignal(str)
    all_cleared = pyqtSignal()
    indicator_hovered = pyqtSignal(str)
    indicator_unhovered = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pinned: dict[str, PinnedTrajectory] = {}
        self._insertion_order: tuple[str, ...] = ()
        self._primary_id: str | None = None
        self._indicators: dict[str, TrajectoryIndicator] = {}
        self._basin_mode = False
        self._winding_colormap_name = "Basin Hash"
        self._current_params = DoublePendulumParams()
        self._dt = 0.01
        self._t_end = 10.0
        self._scrub_playing = False

        self._init_ui()
        self._init_timer()
        self._init_scrub_signals()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # --- Header ---
        header_row = QHBoxLayout()
        title = QLabel("Pinned Trajectories")
        title.setStyleSheet("color: #ccc; font-weight: bold; font-size: 12px;")
        header_row.addWidget(title)
        header_row.addStretch()

        self._clear_all_btn = QPushButton("Clear All")
        self._clear_all_btn.setStyleSheet(
            "QPushButton { color: #888; font-size: 11px; }"
            "QPushButton:hover { color: #ff6666; }"
        )
        self._clear_all_btn.clicked.connect(self._on_clear_all)
        self._clear_all_btn.setEnabled(False)
        header_row.addWidget(self._clear_all_btn)

        main_layout.addLayout(header_row)

        # --- Hover section ---
        self._hover_group = QGroupBox("Hover")
        self._hover_group.setStyleSheet(
            "QGroupBox { color: #999; font-size: 11px; border: 1px solid #3a3a4a; "
            "border-radius: 4px; margin-top: 8px; padding-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        hover_layout = QHBoxLayout()
        hover_layout.setSpacing(4)
        self._hover_group.setLayout(hover_layout)

        # Hover: initial diagram (always shown)
        self._hover_initial_diagram = PendulumDiagram()
        self._hover_initial_diagram.set_label("Initial (t=0)")
        self._hover_initial_diagram.setFixedSize(110, 110)
        hover_layout.addWidget(self._hover_initial_diagram)

        # Hover: winding circle (basin mode)
        self._hover_winding_circle = WindingCircle()
        self._hover_winding_circle.set_label("Winding")
        self._hover_winding_circle.setFixedSize(110, 110)
        hover_layout.addWidget(self._hover_winding_circle)

        # Angle labels under the hover section
        hover_info_layout = QVBoxLayout()
        self._hover_angles_label = QLabel(
            "\u03b8\u2081=\u2014, \u03b8\u2082=\u2014"
        )
        self._hover_angles_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._hover_angles_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hover_info_layout.addWidget(self._hover_angles_label)

        main_layout.addWidget(self._hover_group)
        main_layout.addLayout(hover_info_layout)

        # --- Multi-trajectory animation ---
        self._anim_diagram = MultiTrajectoryDiagram()
        self._anim_diagram.set_label("Trajectories")
        self._anim_diagram.setMinimumSize(200, 200)
        main_layout.addWidget(self._anim_diagram, stretch=1)

        # --- Scrub controls ---
        scrub_frame = QFrame()
        scrub_frame.setStyleSheet("QFrame { border: none; }")
        scrub_layout = QVBoxLayout(scrub_frame)
        scrub_layout.setContentsMargins(4, 0, 4, 0)
        scrub_layout.setSpacing(2)

        self._scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self._scrub_slider.setMinimum(0)
        self._scrub_slider.setMaximum(0)
        self._scrub_slider.setValue(0)
        self._scrub_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._scrub_slider.setEnabled(False)
        scrub_layout.addWidget(self._scrub_slider)

        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(6)

        self._play_btn = QPushButton("▶")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(32)
        self._play_btn.setStyleSheet(
            "QPushButton { color: #ccc; font-size: 11px; padding: 2px 4px; }"
            "QPushButton:checked { color: #ff9944; }"
        )
        self._play_btn.setEnabled(False)
        scrub_row.addWidget(self._play_btn)

        self._scrub_time_label = QLabel("t = 0.0 s")
        self._scrub_time_label.setStyleSheet(
            "color: #aaa; font-size: 11px;"
        )
        scrub_row.addWidget(self._scrub_time_label)
        scrub_row.addStretch()

        scrub_layout.addLayout(scrub_row)
        main_layout.addWidget(scrub_frame)

        # --- Indicator row ---
        self._indicator_frame = QFrame()
        self._indicator_frame.setStyleSheet(
            "QFrame { border: none; }"
        )
        self._indicator_layout = QHBoxLayout(self._indicator_frame)
        self._indicator_layout.setContentsMargins(2, 2, 2, 2)
        self._indicator_layout.setSpacing(4)
        self._indicator_layout.addStretch()  # center the circles

        main_layout.addWidget(self._indicator_frame)

    def _init_timer(self) -> None:
        """Create the master animation timer."""
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(ANIMATION_INTERVAL_MS)
        self._anim_timer.timeout.connect(self._on_anim_tick)

    def _init_scrub_signals(self) -> None:
        """Wire the scrub slider and play/pause button."""
        self._scrub_slider.valueChanged.connect(self._on_scrub_slider_changed)
        self._scrub_slider.sliderPressed.connect(self._on_scrub_pressed)
        self._scrub_slider.sliderReleased.connect(self._on_scrub_released)
        self._play_btn.toggled.connect(self._on_play_toggled)

    # --- Basin mode toggle ---

    def set_basin_mode(self, basin: bool) -> None:
        """Set basin mode. Always True now (angle mode removed from UI)."""
        self._basin_mode = basin

    # --- Hover updates ---

    def update_hover_basin(
        self,
        theta1: float,
        theta2: float,
        n1: int,
        n2: int,
        colormap_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        params: DoublePendulumParams,
    ) -> None:
        """Update the hover display for basin mode."""
        self._hover_initial_diagram.set_params(params)
        self._hover_initial_diagram.set_state(theta1, theta2)

        self._hover_winding_circle.set_winding(n1, n2, colormap_fn)

        self._hover_angles_label.setText(
            f"\u03b8\u2081={math.degrees(theta1):.1f}\u00b0, "
            f"\u03b8\u2082={math.degrees(theta2):.1f}\u00b0"
        )

    def set_hover_params(self, params: DoublePendulumParams) -> None:
        """Update physics params on the hover diagram."""
        self._hover_initial_diagram.set_params(params)

    # --- Pinned trajectory management ---

    def add_row(
        self,
        row_id: str,
        theta1: float,
        theta2: float,
        trajectory: np.ndarray,
        params: DoublePendulumParams,
        n1: int = 0,
        n2: int = 0,
    ) -> None:
        """Add a pinned trajectory.

        Args:
            row_id: Unique identifier for this trajectory.
            theta1: Initial theta1 angle.
            theta2: Initial theta2 angle.
            trajectory: (n_steps, 4) float32 state array.
            params: Physics parameters.
            n1: Winding number for theta1.
            n2: Winding number for theta2.
        """
        if row_id in self._pinned:
            return  # Avoid duplicates

        self._current_params = params

        # Subsample trajectory
        subsampled = trajectory[::FRAME_SUBSAMPLE]

        # Look up basin color
        color_rgb = self._lookup_basin_color(n1, n2)

        # Create immutable pinned trajectory
        pinned = PinnedTrajectory(
            row_id=row_id,
            theta1_init=theta1,
            theta2_init=theta2,
            trajectory=subsampled,
            n1=n1,
            n2=n2,
            color_rgb=color_rgb,
        )
        self._pinned = {**self._pinned, row_id: pinned}
        self._insertion_order = (*self._insertion_order, row_id)

        # First trajectory becomes primary
        if self._primary_id is None:
            self._primary_id = row_id

        # Create indicator widget
        indicator = TrajectoryIndicator(
            row_id=row_id,
            n1=n1,
            n2=n2,
            theta1_init=theta1,
            theta2_init=theta2,
            color_rgb=color_rgb,
        )
        indicator.clicked.connect(self._on_indicator_clicked)
        indicator.remove_clicked.connect(self._on_indicator_remove)
        indicator.hovered.connect(self._on_indicator_hovered)
        indicator.hovered.connect(self.indicator_hovered)
        indicator.unhovered.connect(self._on_indicator_unhovered)
        indicator.unhovered.connect(self.indicator_unhovered)
        self._indicators = {**self._indicators, row_id: indicator}

        # Insert before the trailing stretch
        idx = self._indicator_layout.count() - 1
        self._indicator_layout.insertWidget(idx, indicator)

        self._clear_all_btn.setEnabled(True)

        # Rebuild the stacked animation
        self._rebuild_animation()
        self._update_indicator_highlights()

        # Start timer and auto-play if first trajectory
        if len(self._pinned) == 1:
            self._play_btn.setChecked(True)
            self._anim_timer.start()

    def remove_row(self, row_id: str) -> None:
        """Remove a pinned trajectory by ID."""
        if row_id not in self._pinned:
            return

        # Exit freeze-frame if the removed trajectory was being hovered
        self._anim_diagram.exit_freeze_frame()

        # Remove from pinned dict (immutable rebuild)
        self._pinned = {
            k: v for k, v in self._pinned.items() if k != row_id
        }
        self._insertion_order = tuple(
            rid for rid in self._insertion_order if rid != row_id
        )

        # Remove indicator widget
        indicator = self._indicators.get(row_id)
        if indicator is not None:
            self._indicator_layout.removeWidget(indicator)
            indicator.deleteLater()
            self._indicators = {
                k: v for k, v in self._indicators.items() if k != row_id
            }

        # Update primary
        if self._primary_id == row_id:
            if self._insertion_order:
                self._primary_id = self._insertion_order[0]
            else:
                self._primary_id = None

        # Rebuild animation
        self._rebuild_animation()
        self._update_indicator_highlights()

        if not self._pinned:
            self._anim_timer.stop()
            self._play_btn.setChecked(False)
            self._clear_all_btn.setEnabled(False)

    def clear_all(self) -> None:
        """Remove all pinned trajectories."""
        # Exit freeze-frame before clearing
        self._anim_diagram.exit_freeze_frame()

        # Remove all indicator widgets
        for indicator in self._indicators.values():
            self._indicator_layout.removeWidget(indicator)
            indicator.deleteLater()

        self._pinned = {}
        self._insertion_order = ()
        self._indicators = {}
        self._primary_id = None

        self._anim_timer.stop()
        self._clear_all_btn.setEnabled(False)

        # Reset scrub controls
        self._play_btn.setChecked(False)
        self._scrub_slider.blockSignals(True)
        self._scrub_slider.setValue(0)
        self._scrub_slider.setMaximum(0)
        self._scrub_slider.blockSignals(False)
        self._scrub_slider.setEnabled(False)
        self._play_btn.setEnabled(False)
        self._scrub_time_label.setText("t = 0.0 s")

        # Clear the animation diagram
        self._anim_diagram.set_trajectories((), self._current_params)

    def set_winding_colormap(self, name: str) -> None:
        """Update the winding colormap and recolor all trajectories.

        Args:
            name: Colormap name (key into WINDING_COLORMAPS).
        """
        self._winding_colormap_name = name

        # Rebuild all PinnedTrajectory with new colors (immutable)
        rebuilt: dict[str, PinnedTrajectory] = {}
        for row_id, pt in self._pinned.items():
            new_color = self._lookup_basin_color(pt.n1, pt.n2)
            rebuilt[row_id] = PinnedTrajectory(
                row_id=pt.row_id,
                theta1_init=pt.theta1_init,
                theta2_init=pt.theta2_init,
                trajectory=pt.trajectory,
                n1=pt.n1,
                n2=pt.n2,
                color_rgb=new_color,
            )
        self._pinned = rebuilt

        # Update indicator colors
        for row_id, indicator in self._indicators.items():
            pt = self._pinned.get(row_id)
            if pt is not None:
                indicator.set_color(pt.color_rgb)

        # Rebuild animation with new colors
        self._rebuild_animation()

    def update_winding(self, row_id: str, n1: int, n2: int) -> None:
        """Update winding numbers for a pinned trajectory (definition toggle).

        Immutably rebuilds the PinnedTrajectory with new n1/n2 and color,
        updates the indicator widget, and rebuilds the animation.

        Args:
            row_id: Trajectory identifier.
            n1: New winding number for theta1.
            n2: New winding number for theta2.
        """
        pt = self._pinned.get(row_id)
        if pt is None:
            return

        new_color = self._lookup_basin_color(n1, n2)
        rebuilt_pt = PinnedTrajectory(
            row_id=pt.row_id,
            theta1_init=pt.theta1_init,
            theta2_init=pt.theta2_init,
            trajectory=pt.trajectory,
            n1=n1,
            n2=n2,
            color_rgb=new_color,
        )
        self._pinned = {**self._pinned, row_id: rebuilt_pt}

        # Update indicator
        indicator = self._indicators.get(row_id)
        if indicator is not None:
            indicator.set_winding(n1, n2)
            indicator.set_color(new_color)

        # Rebuild animation with updated colors
        self._rebuild_animation()

    def get_pinned(self) -> dict[str, PinnedTrajectory]:
        """Return the current pinned trajectories (read-only view)."""
        return dict(self._pinned)

    def get_marker_colors(self) -> dict[str, tuple[int, int, int]]:
        """Return {row_id: (R, G, B)} for all pinned trajectories."""
        return {
            row_id: pt.color_rgb
            for row_id, pt in self._pinned.items()
        }

    # --- Animation & scrub ---

    def set_time_params(self, t_end: float, dt: float) -> None:
        """Update simulation time parameters for scrub label.

        Args:
            t_end: Simulation end time (seconds).
            dt: Time step (seconds).
        """
        self._t_end = t_end
        self._dt = dt
        self._anim_diagram.set_dt_per_frame(FRAME_SUBSAMPLE * dt)
        self._update_scrub_time_label()

    def _on_anim_tick(self) -> None:
        """Advance the stacked animation by one frame and sync slider."""
        self._anim_diagram.advance_frame()
        self._sync_slider_to_frame()

    def _sync_slider_to_frame(self) -> None:
        """Update slider position to match current animation frame."""
        max_frames = self._anim_diagram.max_frames
        if max_frames <= 0:
            return

        # Block signals to avoid recursive slider → set_frame → slider loop
        self._scrub_slider.blockSignals(True)
        self._scrub_slider.setValue(self._anim_diagram.frame_idx)
        self._scrub_slider.blockSignals(False)

        self._update_scrub_time_label()

    def _update_scrub_time_label(self) -> None:
        """Update the time label to show current simulation time."""
        frame_idx = self._anim_diagram.frame_idx
        # Each animation frame = FRAME_SUBSAMPLE * dt of simulation time
        t_sim = frame_idx * FRAME_SUBSAMPLE * self._dt
        self._scrub_time_label.setText(f"t = {t_sim:.1f} s")

    def _on_scrub_slider_changed(self, value: int) -> None:
        """Scrub slider moved: jump the animation to that frame."""
        self._anim_diagram.set_frame(value)
        self._update_scrub_time_label()

    def _on_scrub_pressed(self) -> None:
        """User started scrubbing: pause auto-advance if playing."""
        if self._play_btn.isChecked():
            self._anim_timer.stop()

    def _on_scrub_released(self) -> None:
        """User released scrub slider: resume auto-advance if playing."""
        if self._play_btn.isChecked() and not self._anim_diagram.is_frozen:
            self._anim_timer.start()

    def _on_play_toggled(self, checked: bool) -> None:
        """Play/pause button toggled."""
        if not self._pinned:
            return

        if checked:
            self._play_btn.setText("⏸")
            self._scrub_playing = True
            if not self._anim_diagram.is_frozen:
                self._anim_timer.start()
        else:
            self._play_btn.setText("▶")
            self._scrub_playing = False
            self._anim_timer.stop()

    def _update_scrub_range(self) -> None:
        """Update slider range from current animation max_frames."""
        max_frames = self._anim_diagram.max_frames
        has_frames = max_frames > 0
        self._scrub_slider.setMaximum(max(0, max_frames - 1))
        self._scrub_slider.setEnabled(has_frames)
        self._play_btn.setEnabled(has_frames)

    # --- Internal helpers ---

    def _lookup_basin_color(self, n1: int, n2: int) -> tuple[int, int, int]:
        """Look up basin color as (R, G, B) from current winding colormap.

        Args:
            n1: Winding number for theta1.
            n2: Winding number for theta2.

        Returns:
            (R, G, B) tuple of uint8 values.
        """
        colormap_fn = WINDING_COLORMAPS.get(self._winding_colormap_name)
        if colormap_fn is None:
            return (128, 128, 128)

        b, g, r, _a = get_single_winding_color(n1, n2, colormap_fn)
        return (r, g, b)

    def _rebuild_animation(self) -> None:
        """Rebuild the MultiTrajectoryDiagram from current pinned data."""
        if not self._pinned:
            self._anim_diagram.set_trajectories((), self._current_params)
            self._update_scrub_range()
            return

        # Build TrajectoryInfo tuple in insertion order
        traj_infos: list[TrajectoryInfo] = []
        primary_index = 0
        for i, row_id in enumerate(self._insertion_order):
            pt = self._pinned.get(row_id)
            if pt is None:
                continue
            if row_id == self._primary_id:
                primary_index = len(traj_infos)
            traj_infos.append(TrajectoryInfo(
                trajectory=pt.trajectory,
                color_rgb=pt.color_rgb,
            ))

        self._anim_diagram.set_trajectories(
            tuple(traj_infos), self._current_params,
        )
        self._anim_diagram.set_primary(primary_index)
        self._update_scrub_range()

    def _update_indicator_highlights(self) -> None:
        """Update which indicator has the primary highlight."""
        for row_id, indicator in self._indicators.items():
            indicator.set_highlighted(row_id == self._primary_id)

    # --- Freeze-frame hover ---

    def _on_indicator_hovered(self, row_id: str) -> None:
        """Indicator mouse-enter: show freeze-frame for this trajectory."""
        pt = self._pinned.get(row_id)
        if pt is None:
            return

        tinfo = TrajectoryInfo(
            trajectory=pt.trajectory,
            color_rgb=pt.color_rgb,
        )
        self._anim_diagram.enter_freeze_frame(tinfo)
        self._anim_timer.stop()

    def _on_indicator_unhovered(self, row_id: str) -> None:
        """Indicator mouse-leave: restore normal animation."""
        self._anim_diagram.exit_freeze_frame()
        if self._scrub_playing:
            self._anim_timer.start()

    # --- Callbacks ---

    def _on_indicator_clicked(self, row_id: str) -> None:
        """Handle indicator click: foreground that trajectory."""
        if row_id not in self._pinned:
            return
        self._primary_id = row_id
        self._rebuild_animation()
        self._update_indicator_highlights()

    def _on_indicator_remove(self, row_id: str) -> None:
        """Handle indicator X button: remove trajectory."""
        self.remove_row(row_id)
        self.row_removed.emit(row_id)

    def _on_clear_all(self) -> None:
        """Handle Clear All button click."""
        self.clear_all()
        self.all_cleared.emit()
