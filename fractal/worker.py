"""Fractal worker: QThread for background compute with progressive levels.

Runs the vectorized RK4 computation in a background thread, emitting
signals for each completed resolution level and progress updates.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from fractal.compute import (
    FractalTask, FractalViewport, ComputeBackend,
    build_initial_conditions,
)

logger = logging.getLogger(__name__)


class FractalWorker(QThread):
    """Background worker for fractal computation.

    Computes one resolution level at a time, emitting level_complete
    for each. Supports cancellation between levels and within the
    RK4 loop (via cancel_check callback).
    """

    level_complete = pyqtSignal(int, object)  # resolution, snapshots (np.ndarray)
    progress = pyqtSignal(int, int)            # steps_done, total_steps
    all_complete = pyqtSignal()

    def __init__(
        self,
        task: FractalTask,
        backend: ComputeBackend,
        progressive_levels: list[int],
    ):
        super().__init__()
        self._task = task
        self._backend = backend
        self._progressive_levels = progressive_levels
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation. Takes effect within ~100 RK4 steps."""
        self._cancelled = True

    def _cancel_check(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        task = self._task

        for level_res in self._progressive_levels:
            if self._cancelled:
                return

            # Build viewport at this resolution level
            level_viewport = FractalViewport(
                center_theta1=task.viewport.center_theta1,
                center_theta2=task.viewport.center_theta2,
                span_theta1=task.viewport.span_theta1,
                span_theta2=task.viewport.span_theta2,
                resolution=level_res,
            )

            ics = build_initial_conditions(level_viewport)

            logger.debug(
                "Computing level %dx%d (%d trajectories)",
                level_res, level_res, ics.shape[0],
            )

            try:
                snapshots = self._backend.simulate_batch(
                    params=task.params,
                    initial_conditions=ics,
                    t_end=task.t_end,
                    dt=task.dt,
                    n_samples=task.n_samples,
                    cancel_check=self._cancel_check,
                    progress_callback=lambda done, total: self.progress.emit(done, total),
                )
            except Exception:
                logger.exception("Fractal computation failed at level %d", level_res)
                return

            if self._cancelled:
                return

            self.level_complete.emit(level_res, snapshots)

        if not self._cancelled:
            self.all_complete.emit()
