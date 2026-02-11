"""Fractal view: orchestrates canvas, controls, worker, cache, and inspect column.

This is the main coordinator for fractal mode. It:
- Listens to viewport changes from the canvas
- Checks the cache for existing results
- Launches FractalWorker for cache misses
- Feeds completed data to the canvas for display
- Manages the inspect column (hover display + pinned trajectories)
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter

from simulation import DoublePendulumParams
from fractal.canvas import FractalCanvas
from fractal.controls import FractalControls
from fractal.inspect_column import InspectColumn
from fractal.cache import FractalCache, CacheKey
from fractal.compute import (
    FractalViewport, FractalTask,
    get_default_backend, get_progressive_levels, DEFAULT_N_SAMPLES,
)
from fractal.coloring import interpolate_angle
from fractal._numpy_backend import rk4_single_trajectory
from fractal.winding import extract_winding_numbers, WINDING_COLORMAPS
from fractal.worker import FractalWorker
from ui_common import LoadingOverlay

logger = logging.getLogger(__name__)

# Default RK4 step size
DEFAULT_DT = 0.01


class FractalView(QWidget):
    """Complete fractal mode: canvas + controls + inspect column + compute."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Backend
        self._backend = get_default_backend()
        self._progressive_levels = get_progressive_levels(self._backend)

        # Cache (adaptive budget based on backend)
        backend_name = type(self._backend).__name__
        if backend_name == "JaxBackend":
            cache_budget = 64 * 1024 * 1024
        elif backend_name == "NumbaBackend":
            cache_budget = 128 * 1024 * 1024
        else:
            cache_budget = 512 * 1024 * 1024
        self._cache = FractalCache(max_bytes=cache_budget)

        # UI: 3-pane layout [InspectColumn | Canvas | Controls]
        self.inspect_column = InspectColumn()
        self.canvas = FractalCanvas()
        self.controls = FractalControls()

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self.inspect_column)
        self._splitter.addWidget(self.canvas)
        self._splitter.addWidget(self.controls)
        self._splitter.setStretchFactor(0, 1)  # inspect column
        self._splitter.setStretchFactor(1, 3)  # canvas
        self._splitter.setStretchFactor(2, 2)  # controls

        # Start with inspect column hidden
        self.inspect_column.setVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._splitter)

        # Loading overlay
        self.loading_overlay = LoadingOverlay(self.canvas)

        # Worker state
        self._worker: FractalWorker | None = None
        self._retiring_workers: list[FractalWorker] = []
        self._current_params: DoublePendulumParams | None = None

        # Basin display mode
        self._basin_mode = False
        self._current_final_velocities: np.ndarray | None = None

        # Cached final state for basin-mode hover lookup
        self._basin_final_state: np.ndarray | None = None

        # Status labels (AppWindow will read these)
        self._status_text = ""

        # Wire signals
        self.canvas.viewport_changed.connect(self._on_viewport_changed)
        self.controls.time_index_changed.connect(self._on_time_index_changed)
        self.controls.colormap_changed.connect(self._on_colormap_changed)
        self.controls.angle_selection_changed.connect(self._on_angle_changed)
        self.controls.torus_colormap_changed.connect(self._on_torus_colormap_changed)
        self.controls.resolution_changed.connect(self._on_resolution_changed)
        self.controls.physics_changed.connect(self._on_physics_changed)
        self.controls.t_end_changed.connect(self._on_t_end_changed)
        self.controls.zoom_out_clicked.connect(self._on_zoom_out)
        self.controls.tool_mode_changed.connect(self._on_tool_mode_changed)
        self.controls.display_mode_changed.connect(self._on_display_mode_changed)
        self.controls.winding_colormap_changed.connect(self._on_winding_colormap_changed)
        self.canvas.hover_updated.connect(self._on_hover_updated)

        # Inspect column signals
        self.canvas.trajectory_pinned.connect(self._on_trajectory_pinned)
        self.inspect_column.row_removed.connect(self.canvas.remove_marker)
        self.inspect_column.all_cleared.connect(self.canvas.clear_markers)

    # -- Public interface for mode switching --

    def activate(
        self,
        params: DoublePendulumParams | None = None,
        center_theta1: float = 0.0,
        center_theta2: float = 0.0,
    ) -> None:
        """Called when switching to fractal mode."""
        if params is not None:
            self.controls.set_params(params)
            self._current_params = params

        self.canvas.set_viewport(center_theta1, center_theta2)

        # Trigger initial computation
        viewport = self.canvas.get_viewport()
        self._start_computation(viewport)

    def deactivate(self) -> None:
        """Called when switching away from fractal mode."""
        self._cancel_worker()
        # Wait for all retiring workers to finish (blocking, but
        # only happens on mode switch so a brief freeze is acceptable)
        for worker in self._retiring_workers:
            worker.wait(3000)
        self._retiring_workers.clear()

    def get_params(self) -> DoublePendulumParams:
        return self.controls.get_params()

    # -- Computation pipeline --

    def _start_computation(self, viewport: FractalViewport) -> None:
        """Start the progressive computation pipeline for a viewport."""
        params = self.controls.get_params()
        self._current_params = params

        # Determine t_end: auto-scaled in basin mode, user-set in angle mode
        if self._basin_mode:
            mu = max(params.friction, 0.01)
            t_end = min(5.0 / mu, 500.0)
        else:
            t_end = self.controls.get_t_end()

        # Check cache for the full-resolution result
        full_key = CacheKey.from_viewport(viewport, params)
        cached = self._cache.get(full_key)
        if cached is not None:
            logger.debug("Cache hit for %s", full_key)
            if self._basin_mode:
                self._display_basin(cached)
            else:
                self.canvas.display(cached, self.controls.get_time_index())
            return

        # Cache miss: start progressive pipeline
        self._cancel_worker()

        # Update canvas resolution
        self.canvas.set_resolution(viewport.resolution)

        task = FractalTask(
            params=params,
            viewport=viewport,
            t_end=t_end,
            dt=DEFAULT_DT,
            n_samples=DEFAULT_N_SAMPLES,
            basin=self._basin_mode,
        )

        self._worker = FractalWorker(task, self._backend, self._progressive_levels)
        self._worker.level_complete.connect(self._on_level_complete)
        self._worker.progress.connect(self._on_progress)
        self._worker.all_complete.connect(self._on_all_complete)
        self._worker.finished.connect(self._on_worker_finished)

        self.loading_overlay.start("Computing fractal...")
        self._worker.start()

    def _cancel_worker(self) -> None:
        """Cancel any running worker.

        Moves the old worker to a retirement list so it won't be
        garbage-collected while the thread is still running. The worker
        cleans itself up via its finished signal.
        """
        if self._worker is None:
            return

        old_worker = self._worker
        self._worker = None

        # Disconnect signals so stale results don't arrive
        try:
            old_worker.level_complete.disconnect(self._on_level_complete)
            old_worker.progress.disconnect(self._on_progress)
            old_worker.all_complete.disconnect(self._on_all_complete)
            old_worker.finished.disconnect(self._on_worker_finished)
        except TypeError:
            pass  # Already disconnected

        if old_worker.isRunning():
            old_worker.cancel()
            # Keep a reference so the QThread isn't destroyed while running
            self._retiring_workers.append(old_worker)
            old_worker.finished.connect(lambda w=old_worker: self._cleanup_retired(w))

    def _cleanup_retired(self, worker: FractalWorker) -> None:
        """Remove a retired worker after its thread has fully stopped."""
        try:
            self._retiring_workers.remove(worker)
        except ValueError:
            pass  # Already removed
        logger.debug(
            "Retired worker cleaned up, %d still retiring",
            len(self._retiring_workers),
        )

    # -- Worker signal handlers --

    def _on_level_complete(
        self,
        resolution: int,
        data: np.ndarray,
        final_velocities: np.ndarray | None,
    ) -> None:
        """Handle a completed progressive level.

        In basin mode, data is (N, 4) final state.
        In angle mode, data is (N, 2, n_samples) snapshots.
        """
        params = self.controls.get_params()
        viewport = self.canvas.get_viewport()

        # Build a viewport at this resolution for caching
        level_viewport = FractalViewport(
            center_theta1=viewport.center_theta1,
            center_theta2=viewport.center_theta2,
            span_theta1=viewport.span_theta1,
            span_theta2=viewport.span_theta2,
            resolution=resolution,
        )
        key = CacheKey.from_viewport(level_viewport, params)
        self._cache.put(key, data)

        # Display this level: basin vs angle mode
        if self._basin_mode:
            self._display_basin(data)
        else:
            self._current_final_velocities = final_velocities
            self.canvas.display(data, self.controls.get_time_index())

        logger.debug(
            "Level %dx%d complete, cache: %.1f MB",
            resolution, resolution, self._cache.memory_used_mb,
        )

    def _on_progress(self, steps_done: int, total_steps: int) -> None:
        """Update loading overlay with progress."""
        if total_steps > 0:
            pct = int(100 * steps_done / total_steps)
            self.loading_overlay.message = f"Computing fractal... {pct}%"

    def _on_all_complete(self) -> None:
        """All progressive levels finished."""
        self.loading_overlay.stop()
        self.canvas.activate_pending_ghost()

    def _on_worker_finished(self) -> None:
        """Worker thread has exited (may be due to cancellation)."""
        self.loading_overlay.stop()
        # Only clear reference if this is still the current worker.
        # Retired workers are cleaned up via _cleanup_retired.
        sender = self.sender()
        if sender is self._worker:
            self._worker = None

    # -- UI signal handlers --

    def _on_viewport_changed(self, viewport: FractalViewport) -> None:
        """Canvas emitted a new viewport (rectangle zoom or zoom-out)."""
        self._start_computation(viewport)

    def _on_zoom_out(self) -> None:
        """Zoom out button clicked: delegate to the canvas."""
        self.canvas.zoom_out()

    def _on_time_index_changed(self, time_index: float) -> None:
        """Time slider moved: update display from current snapshots."""
        self.canvas.set_time_index(time_index)
        self.controls.update_time_label(self.controls.get_t_end())

    def _on_colormap_changed(self, name: str) -> None:
        """Colormap dropdown changed."""
        self.canvas.set_colormap(name)

    def _on_angle_changed(self, angle_index: int) -> None:
        """Angle display selection changed."""
        self.canvas.set_angle_index(angle_index)

    def _on_torus_colormap_changed(self, name: str) -> None:
        """Torus colormap dropdown changed."""
        self.canvas.set_torus_colormap(name)

    def _on_resolution_changed(self, resolution: int) -> None:
        """Resolution dropdown changed: recompute at new resolution."""
        self.canvas.set_resolution(resolution)
        # Update progressive levels for the new max resolution
        self._progressive_levels = [
            lev for lev in get_progressive_levels(self._backend)
            if lev <= resolution
        ]
        if resolution not in self._progressive_levels:
            self._progressive_levels.append(resolution)
        viewport = self.canvas.get_viewport()
        new_viewport = FractalViewport(
            center_theta1=viewport.center_theta1,
            center_theta2=viewport.center_theta2,
            span_theta1=viewport.span_theta1,
            span_theta2=viewport.span_theta2,
            resolution=resolution,
        )
        self._start_computation(new_viewport)

    def _on_physics_changed(self) -> None:
        """Physics parameters changed: invalidate cache and recompute."""
        old_params = self._current_params
        new_params = self.controls.get_params()

        if old_params is not None:
            from fractal.cache import _params_hash
            self._cache.invalidate_params(_params_hash(old_params))

        self._current_params = new_params
        viewport = self.canvas.get_viewport()
        self._start_computation(viewport)

    def _on_t_end_changed(self) -> None:
        """Simulation duration changed: invalidate all cache and recompute."""
        self._cache.clear()
        viewport = self.canvas.get_viewport()
        self._start_computation(viewport)

    def _on_tool_mode_changed(self, mode: str) -> None:
        """Tool mode changed: show/hide inspect column, set canvas mode."""
        self.canvas.set_tool_mode(mode)
        show_inspect = (mode == "inspect")
        self.inspect_column.setVisible(show_inspect)

    def _on_display_mode_changed(self, mode: str) -> None:
        """Handle Angle/Basin toggle from controls."""
        self._basin_mode = (mode == "basin")
        self.canvas.set_basin_mode(self._basin_mode)
        self.inspect_column.set_basin_mode(self._basin_mode)
        self._cache.clear()

        # Clear pinned trajectories on mode switch
        self.inspect_column.clear_all()
        self.canvas.clear_markers()

        viewport = self.canvas.get_viewport()
        self._start_computation(viewport)

    def _on_winding_colormap_changed(self, name: str) -> None:
        """Winding colormap dropdown changed."""
        self.canvas.set_winding_colormap(name)
        self.inspect_column.set_winding_colormap(name)

    def _display_basin(self, final_state: np.ndarray) -> None:
        """Display the final-state winding number image.

        Args:
            final_state: (N, 4) float32 [theta1, theta2, omega1, omega2].
        """
        self._basin_final_state = final_state
        theta1_final = final_state[:, 0].astype(np.float32)
        theta2_final = final_state[:, 1].astype(np.float32)
        self.canvas.display_basin_final(theta1_final, theta2_final)

    # -- Inspect tool: hover + click-to-pin --

    def _on_hover_updated(self, theta1: float, theta2: float) -> None:
        """Inspect tool: look up data for hovered point and update column."""
        params = self.controls.get_params()

        if self._basin_mode:
            self._on_hover_basin(theta1, theta2, params)
        else:
            self._on_hover_angle(theta1, theta2, params)

    def _on_hover_basin(
        self, theta1: float, theta2: float, params: DoublePendulumParams,
    ) -> None:
        """Basin mode hover: show initial diagram + winding circle."""
        # Look up winding numbers for the hovered point from the final state
        final_state = self._basin_final_state
        if final_state is None:
            return

        viewport = self.canvas.get_viewport()
        res = int(np.sqrt(final_state.shape[0]))

        # Convert hovered physics coords to grid indices
        half_span1 = viewport.span_theta1 / 2
        half_span2 = viewport.span_theta2 / 2
        vmin1 = viewport.center_theta1 - half_span1
        vmin2 = viewport.center_theta2 - half_span2

        nx = (theta1 - vmin1) / viewport.span_theta1
        ny = (theta2 - vmin2) / viewport.span_theta2

        col = int(max(0, min(res - 1, round(nx * (res - 1)))))
        row = int(max(0, min(res - 1, round(ny * (res - 1)))))
        flat_idx = row * res + col

        # Extract winding numbers for this trajectory
        theta1_final = float(final_state[flat_idx, 0])
        theta2_final = float(final_state[flat_idx, 1])
        n1_arr, n2_arr = extract_winding_numbers(
            np.array([theta1_final], dtype=np.float32),
            np.array([theta2_final], dtype=np.float32),
        )
        n1, n2 = int(n1_arr[0]), int(n2_arr[0])

        # Get current winding colormap function
        colormap_name = self.canvas._winding_colormap_name
        colormap_fn = WINDING_COLORMAPS.get(colormap_name)
        if colormap_fn is None:
            return

        self.inspect_column.update_hover_basin(
            theta1, theta2, n1, n2, colormap_fn, params,
        )

    def _on_hover_angle(
        self, theta1: float, theta2: float, params: DoublePendulumParams,
    ) -> None:
        """Angle mode hover: show initial diagram + at-t diagram."""
        snapshots = self.canvas._current_snapshots
        if snapshots is None:
            return

        viewport = self.canvas.get_viewport()
        res = int(np.sqrt(snapshots.shape[0]))

        # Convert hovered physics coords to grid indices
        half_span1 = viewport.span_theta1 / 2
        half_span2 = viewport.span_theta2 / 2
        vmin1 = viewport.center_theta1 - half_span1
        vmin2 = viewport.center_theta2 - half_span2

        nx = (theta1 - vmin1) / viewport.span_theta1
        ny = (theta2 - vmin2) / viewport.span_theta2

        col = int(max(0, min(res - 1, round(nx * (res - 1)))))
        row = int(max(0, min(res - 1, round(ny * (res - 1)))))
        flat_idx = row * res + col

        # Look up angles at current time
        time_index = self.controls.get_time_index()
        theta1_at_t = float(
            interpolate_angle(snapshots[:, 0, :], time_index)[flat_idx]
        )
        theta2_at_t = float(
            interpolate_angle(snapshots[:, 1, :], time_index)[flat_idx]
        )

        # Compute t_value from slider position
        t_end = self.controls.get_t_end()
        slider_val = self.controls.time_slider.value()
        slider_max = max(1, self.controls.time_slider.maximum())
        t_value = (slider_val / slider_max) * t_end

        self.inspect_column.update_hover_angle(
            theta1, theta2,
            theta1_at_t, theta2_at_t,
            t_value, params,
        )

    def _on_trajectory_pinned(
        self, row_id: str, theta1: float, theta2: float,
    ) -> None:
        """Canvas click in inspect mode: compute trajectory and add row."""
        params = self.controls.get_params()

        # Compute t_end for the trajectory
        if self._basin_mode:
            mu = max(params.friction, 0.01)
            t_end = min(5.0 / mu, 500.0)
        else:
            t_end = self.controls.get_t_end()

        # Compute the single trajectory via RK4
        trajectory = rk4_single_trajectory(
            params, theta1, theta2, t_end, DEFAULT_DT,
        )

        # Extract winding numbers from the final state
        final_state = trajectory[-1]
        n1_arr, n2_arr = extract_winding_numbers(
            np.array([float(final_state[0])], dtype=np.float32),
            np.array([float(final_state[1])], dtype=np.float32),
        )
        n1, n2 = int(n1_arr[0]), int(n2_arr[0])

        # Pass time params and add row to the inspect column with winding numbers
        self.inspect_column.set_time_params(t_end, DEFAULT_DT)
        self.inspect_column.add_row(
            row_id, theta1, theta2, trajectory, params, n1, n2,
        )
