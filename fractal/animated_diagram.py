"""MultiTrajectoryDiagram: stacked pendulum animation with basin-colored bobs.

Plays back multiple trajectories simultaneously in a single widget:
- Primary trajectory at full opacity with bob2 fading trail
- Ghost trajectories at reduced alpha
- Both bobs colored per-trajectory from basin colormap
- 1-second pause at initial state before animation starts
- Shorter trajectories freeze at their final frame
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPaintEvent, QColor, QPen, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget

from simulation import DoublePendulumParams, positions, total_energy
from fractal.compute import saddle_energy


# Drawing constants
PIVOT_COLOR = QColor(140, 140, 140)
ARM_COLOR = QColor(220, 220, 220)
GHOST_ARM_COLOR = QColor(120, 120, 140)
BACKGROUND_COLOR = QColor(30, 30, 45)
BOB_RADIUS = 6
PIVOT_RADIUS = 4
ARM_WIDTH = 2
GHOST_ARM_WIDTH = 1

# Trail settings (only for primary trajectory)
TRAIL_MAX_LENGTH = 60
TRAIL_WIDTH = 2

# Freeze-frame settings
FREEZE_TRACE_WIDTH = 2
FREEZE_TRACE_ACTIVE_ALPHA = 200    # pre-basin-capture trace opacity
FREEZE_TRACE_SETTLED_ALPHA = 40    # post-basin-capture trace opacity
FREEZE_TRACE_TRANSITION = 20       # frames over which alpha blends between the two
FREEZE_ARM_ALPHA_MIN = 50
FREEZE_ARM_ALPHA_MAX = 200
FREEZE_KEYFRAME_COUNT = 5
FREEZE_BOB_RADIUS = 5

# Basin-settle animation truncation
SETTLE_BUFFER_SECONDS = 5.0  # show this much animation after all trajectories settle

# Animation timing
PAUSE_FRAMES = 30  # ~1 second at 30fps


@dataclass(frozen=True)
class TrajectoryInfo:
    """Immutable data for a single trajectory in the stacked animation.

    Attributes:
        trajectory: (n_steps, 4) float32 subsampled state array.
        color_rgb: Basin colormap color for this trajectory's bobs.
    """

    trajectory: np.ndarray
    color_rgb: tuple[int, int, int]


class MultiTrajectoryDiagram(QWidget):
    """Stacked pendulum animation with ghost IC pattern.

    All trajectories play simultaneously. The primary trajectory
    is drawn at full opacity with a bob2 trail; others are ghosts
    at reduced alpha.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 180)

        self._params = DoublePendulumParams()
        self._trajectories: tuple[TrajectoryInfo, ...] = ()
        self._primary_index = 0
        self._frame_idx = 0
        self._pause_remaining = PAUSE_FRAMES
        self._label = ""

        # Trail: deque of recent (px_x, px_y) for primary bob2
        self._trail: deque[tuple[float, float]] = deque(maxlen=TRAIL_MAX_LENGTH)

        # Freeze-frame: when set, replaces normal animation with static view
        self._freeze_frame: TrajectoryInfo | None = None

        # Effective max frames (truncated when all trajectories have settled)
        self._dt_per_frame = 0.06  # default: FRAME_SUBSAMPLE(6) * dt(0.01)
        self._effective_max: int | None = None

    @property
    def n_trajectories(self) -> int:
        """Number of trajectories loaded."""
        return len(self._trajectories)

    @property
    def is_frozen(self) -> bool:
        """Whether a freeze-frame is currently active."""
        return self._freeze_frame is not None

    def enter_freeze_frame(self, trajectory_info: TrajectoryInfo) -> None:
        """Activate freeze-frame mode for a single trajectory.

        Replaces the normal animation with a static view showing the
        full bob2 trace and pendulum arms at keyframe positions.

        Args:
            trajectory_info: The trajectory to display in freeze-frame.
        """
        self._freeze_frame = trajectory_info
        self.update()

    def exit_freeze_frame(self) -> None:
        """Deactivate freeze-frame mode, restoring normal animation."""
        self._freeze_frame = None
        self.update()

    def set_dt_per_frame(self, dt_per_frame: float) -> None:
        """Set the simulation time per animation frame.

        Args:
            dt_per_frame: Seconds of simulation time per subsampled frame.
        """
        self._dt_per_frame = dt_per_frame

    @property
    def max_frames(self) -> int:
        """Maximum frame count, respecting settle-based truncation."""
        if not self._trajectories:
            return 0
        actual = max(t.trajectory.shape[0] for t in self._trajectories)
        if self._effective_max is not None:
            return min(actual, self._effective_max)
        return actual

    @property
    def frame_idx(self) -> int:
        """Current frame index."""
        return self._frame_idx

    @property
    def is_paused_at_start(self) -> bool:
        """Whether the animation is in its initial pause phase."""
        return self._pause_remaining > 0

    def set_trajectories(
        self,
        trajectories: tuple[TrajectoryInfo, ...],
        params: DoublePendulumParams,
    ) -> None:
        """Replace all trajectories (immutable swap).

        Args:
            trajectories: Tuple of TrajectoryInfo for each pinned trajectory.
            params: Physics parameters (arm lengths for drawing).
        """
        self._trajectories = trajectories
        self._params = params
        self._frame_idx = 0
        self._pause_remaining = PAUSE_FRAMES
        self._trail.clear()
        self._freeze_frame = None
        self._effective_max = self._compute_effective_max()
        self._primary_index = min(self._primary_index, max(0, len(trajectories) - 1))
        self.update()

    def set_primary(self, index: int) -> None:
        """Set which trajectory is the primary (foreground).

        Args:
            index: Index into the trajectories tuple.
        """
        if 0 <= index < len(self._trajectories):
            self._primary_index = index
            self._trail.clear()
            self.update()

    def advance_frame(self) -> None:
        """Advance animation by one frame (called by master timer).

        Handles the 1-second initial pause before trajectory playback.
        """
        if not self._trajectories:
            return

        # Initial pause phase
        if self._pause_remaining > 0:
            self._pause_remaining = self._pause_remaining - 1
            self.update()
            return

        # Advance frame
        self._frame_idx = self._frame_idx + 1

        max_n = self.max_frames
        if max_n == 0:
            return

        # Loop: reset frame and trail, restart pause
        if self._frame_idx >= max_n:
            self._frame_idx = 0
            self._pause_remaining = PAUSE_FRAMES
            self._trail.clear()
            self.update()
            return

        # Add primary bob2 position to trail
        if self._primary_index < len(self._trajectories):
            tinfo = self._trajectories[self._primary_index]
            n = tinfo.trajectory.shape[0]
            fi = min(self._frame_idx, n - 1)
            state = tinfo.trajectory[fi]
            b2_px, b2_py = self._bob2_pixel_coords(
                float(state[0]), float(state[1]),
            )
            self._trail.append((b2_px, b2_py))

        self.update()

    def set_frame(self, frame_idx: int) -> None:
        """Jump to a specific frame (for external scrubbing).

        Skips the initial pause and positions the animation directly
        at the given frame. Clears the trail since scrubbing is
        non-sequential.

        Args:
            frame_idx: Target frame index (clamped to valid range).
        """
        if not self._trajectories:
            return

        max_n = self.max_frames
        if max_n == 0:
            return

        self._frame_idx = max(0, min(frame_idx, max_n - 1))
        self._pause_remaining = 0
        self._trail.clear()
        self.update()

    def set_label(self, text: str) -> None:
        """Set the label shown above the diagram."""
        self._label = text
        self.update()

    def clear_trail(self) -> None:
        """Clear the bob2 trail."""
        self._trail.clear()

    # --- Layout computation ---

    def _layout_metrics(self) -> tuple[float, float, float, int]:
        """Compute layout: (pivot_px, pivot_py, scale, label_height).

        Returns:
            (pivot_px, pivot_py, scale, label_height).
        """
        w = self.width()
        h = self.height()

        label_height = 0
        if self._label:
            fm = QFontMetrics(QFont("Helvetica", 11))
            label_height = fm.height() + 4

        total_length = self._params.l1 + self._params.l2
        usable_h = h - label_height - 2 * BOB_RADIUS - 8
        usable_w = w - 2 * BOB_RADIUS - 8
        usable = min(usable_w, usable_h)
        scale = usable / (2.2 * total_length)

        pivot_px = w / 2
        pivot_py = label_height + (h - label_height) / 2

        return pivot_px, pivot_py, scale, label_height

    def _bob2_pixel_coords(
        self, theta1: float, theta2: float,
    ) -> tuple[float, float]:
        """Convert angles to bob2 pixel coordinates."""
        pivot_px, pivot_py, scale, _ = self._layout_metrics()

        state = [theta1, theta2, 0.0, 0.0]
        _x1, _y1, x2, y2 = positions(state, self._params)

        b2_px = pivot_px + x2 * scale
        b2_py = pivot_py - y2 * scale
        return b2_px, b2_py

    # --- Painting ---

    def paintEvent(self, event: QPaintEvent) -> None:
        """Dispatch to freeze-frame or normal rendering."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        w = self.width()

        # Draw label at top
        if self._label:
            font = QFont("Helvetica", 11)
            painter.setFont(font)
            painter.setPen(QColor(180, 180, 190))
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(self._label)
            painter.drawText(int((w - tw) / 2), fm.ascent() + 2, self._label)

        if self._freeze_frame is not None:
            self._paint_freeze_frame(painter, self._freeze_frame)
        else:
            self._paint_normal(painter)

        painter.end()

    def _paint_normal(self, painter: QPainter) -> None:
        """Draw all trajectories: ghosts first, then primary on top."""
        if not self._trajectories:
            return

        pivot_px, pivot_py, scale, _ = self._layout_metrics()

        # Ghost alpha: inversely proportional to trajectory count
        n_traj = len(self._trajectories)
        ghost_alpha = max(30, min(90, 1200 // max(1, n_traj)))

        # Determine frame index for each trajectory
        # During pause: frame_idx=0, so show initial state
        fi = self._frame_idx if self._pause_remaining <= 0 else 0

        # Draw ghosts first (non-primary), then primary on top
        for pass_idx in range(2):
            for i, tinfo in enumerate(self._trajectories):
                is_primary = (i == self._primary_index)
                if pass_idx == 0 and is_primary:
                    continue  # Draw primary on second pass
                if pass_idx == 1 and not is_primary:
                    continue

                n = tinfo.trajectory.shape[0]
                if n == 0:
                    continue

                # Shorter trajectories freeze at final frame
                local_fi = min(fi, n - 1)
                state = tinfo.trajectory[local_fi]
                theta1, theta2 = float(state[0]), float(state[1])

                x1, y1, x2, y2 = positions(
                    [theta1, theta2, 0.0, 0.0], self._params,
                )

                b1_px = pivot_px + x1 * scale
                b1_py = pivot_py - y1 * scale
                b2_px = pivot_px + x2 * scale
                b2_py = pivot_py - y2 * scale

                r, g, b = tinfo.color_rgb
                alpha = 255 if is_primary else ghost_alpha

                if is_primary:
                    # Draw primary trail before arms/bobs
                    self._draw_trail(painter, r, g, b)

                    # Arms at full opacity
                    arm_pen = QPen(ARM_COLOR)
                    arm_pen.setWidth(ARM_WIDTH)
                    painter.setPen(arm_pen)
                else:
                    # Ghost arms: dimmed
                    arm_pen = QPen(QColor(
                        GHOST_ARM_COLOR.red(),
                        GHOST_ARM_COLOR.green(),
                        GHOST_ARM_COLOR.blue(),
                        ghost_alpha,
                    ))
                    arm_pen.setWidth(GHOST_ARM_WIDTH)
                    painter.setPen(arm_pen)

                painter.drawLine(
                    int(pivot_px), int(pivot_py), int(b1_px), int(b1_py),
                )
                painter.drawLine(
                    int(b1_px), int(b1_py), int(b2_px), int(b2_py),
                )

                # Draw bobs colored by basin color
                painter.setPen(Qt.PenStyle.NoPen)
                bob_color = QColor(r, g, b, alpha)
                painter.setBrush(bob_color)
                painter.drawEllipse(
                    int(b1_px - BOB_RADIUS), int(b1_py - BOB_RADIUS),
                    BOB_RADIUS * 2, BOB_RADIUS * 2,
                )
                painter.drawEllipse(
                    int(b2_px - BOB_RADIUS), int(b2_py - BOB_RADIUS),
                    BOB_RADIUS * 2, BOB_RADIUS * 2,
                )

        # Draw pivot on top of everything
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PIVOT_COLOR)
        painter.drawEllipse(
            int(pivot_px - PIVOT_RADIUS), int(pivot_py - PIVOT_RADIUS),
            PIVOT_RADIUS * 2, PIVOT_RADIUS * 2,
        )

    # --- Freeze-frame rendering ---

    @staticmethod
    def _compute_keyframe_indices(n_frames: int) -> tuple[int, ...]:
        """Compute evenly-spaced keyframe indices for freeze-frame display.

        Returns ~FREEZE_KEYFRAME_COUNT indices including first and last frame.

        Args:
            n_frames: Total number of frames in the trajectory.

        Returns:
            Tuple of frame indices, deduplicated and sorted.
        """
        if n_frames <= 0:
            return ()
        if n_frames == 1:
            return (0,)

        count = min(FREEZE_KEYFRAME_COUNT, n_frames)
        last = n_frames - 1
        indices: list[int] = []
        for i in range(count):
            idx = round(i * last / (count - 1))
            indices.append(idx)

        # Deduplicate while preserving order
        seen: set[int] = set()
        unique: list[int] = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)

        return tuple(unique)

    def _paint_freeze_frame(
        self, painter: QPainter, tinfo: TrajectoryInfo,
    ) -> None:
        """Render freeze-frame: full bob2 trace + keyframe arms.

        Args:
            painter: Active QPainter (already has background + label drawn).
            tinfo: The single trajectory to display.
        """
        n = tinfo.trajectory.shape[0]
        if n == 0:
            return

        pivot_px, pivot_py, scale, _ = self._layout_metrics()

        # 1. Draw full bob2 trace with alpha fade
        self._draw_full_trace(painter, tinfo, pivot_px, pivot_py, scale)

        # 2. Draw keyframe arms + bobs
        keyframes = self._compute_keyframe_indices(n)
        self._draw_keyframe_arms(
            painter, tinfo, keyframes, pivot_px, pivot_py, scale,
        )

        # 3. Draw pivot on top
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PIVOT_COLOR)
        painter.drawEllipse(
            int(pivot_px - PIVOT_RADIUS), int(pivot_py - PIVOT_RADIUS),
            PIVOT_RADIUS * 2, PIVOT_RADIUS * 2,
        )

    def _find_settled_index(self, tinfo: TrajectoryInfo) -> int | None:
        """Find the first frame where energy drops below the saddle threshold.

        Returns None if friction is zero or energy never drops below threshold.
        """
        if self._params.friction <= 0:
            return None

        threshold = saddle_energy(self._params)
        n = tinfo.trajectory.shape[0]

        for i in range(n):
            state = tinfo.trajectory[i]
            e = total_energy(
                [float(state[0]), float(state[1]),
                 float(state[2]), float(state[3])],
                self._params,
            )
            if e < threshold:
                return i

        return None

    def _compute_effective_max(self) -> int | None:
        """Compute effective max frames based on when all trajectories settle.

        If every trajectory has dropped below the saddle energy threshold,
        the animation is truncated to the latest settle point plus a
        SETTLE_BUFFER_SECONDS buffer. Returns None if any trajectory
        never settles (no truncation).
        """
        if not self._trajectories:
            return None

        buffer_frames = math.ceil(SETTLE_BUFFER_SECONDS / self._dt_per_frame)

        latest_settle = 0
        for tinfo in self._trajectories:
            idx = self._find_settled_index(tinfo)
            if idx is None:
                return None  # at least one never settles
            latest_settle = max(latest_settle, idx)

        return latest_settle + buffer_frames

    def _draw_full_trace(
        self,
        painter: QPainter,
        tinfo: TrajectoryInfo,
        pivot_px: float,
        pivot_py: float,
        scale: float,
    ) -> None:
        """Draw the full bob2 path with two-tier alpha.

        Pre-settling segments are drawn at high opacity; post-settling
        segments at low opacity, with a short transition region in between.
        """
        n = tinfo.trajectory.shape[0]
        if n <= 1:
            return

        r, g, b = tinfo.color_rgb
        settled_idx = self._find_settled_index(tinfo)

        # Precompute alpha for each segment index
        half_t = FREEZE_TRACE_TRANSITION / 2

        s_prev = tinfo.trajectory[0]
        _, _, x2_prev, y2_prev = positions(
            [float(s_prev[0]), float(s_prev[1]), 0.0, 0.0], self._params,
        )
        px_prev = pivot_px + x2_prev * scale
        py_prev = pivot_py - y2_prev * scale

        for i in range(1, n):
            alpha = self._trace_alpha_at(i, settled_idx, half_t)

            pen = QPen(QColor(r, g, b, alpha))
            pen.setWidth(FREEZE_TRACE_WIDTH)
            painter.setPen(pen)

            s_curr = tinfo.trajectory[i]
            _, _, x2_curr, y2_curr = positions(
                [float(s_curr[0]), float(s_curr[1]), 0.0, 0.0], self._params,
            )
            px_curr = pivot_px + x2_curr * scale
            py_curr = pivot_py - y2_curr * scale

            painter.drawLine(
                int(px_prev), int(py_prev), int(px_curr), int(py_curr),
            )

            px_prev, py_prev = px_curr, py_curr

    @staticmethod
    def _trace_alpha_at(
        i: int,
        settled_idx: int | None,
        half_transition: float,
    ) -> int:
        """Compute trace alpha for segment *i* given the settled index.

        Before settling: FREEZE_TRACE_ACTIVE_ALPHA.
        After settling: FREEZE_TRACE_SETTLED_ALPHA.
        Transition region blends linearly between the two.
        If settled_idx is None (no settling detected), returns active alpha.
        """
        if settled_idx is None:
            return FREEZE_TRACE_ACTIVE_ALPHA

        if half_transition <= 0:
            if i < settled_idx:
                return FREEZE_TRACE_ACTIVE_ALPHA
            return FREEZE_TRACE_SETTLED_ALPHA

        dist = i - settled_idx
        if dist <= -half_transition:
            return FREEZE_TRACE_ACTIVE_ALPHA
        if dist >= half_transition:
            return FREEZE_TRACE_SETTLED_ALPHA

        # Linear blend within transition region
        frac = (dist + half_transition) / (2 * half_transition)
        return int(
            FREEZE_TRACE_ACTIVE_ALPHA
            + frac * (FREEZE_TRACE_SETTLED_ALPHA - FREEZE_TRACE_ACTIVE_ALPHA)
        )

    def _draw_keyframe_arms(
        self,
        painter: QPainter,
        tinfo: TrajectoryInfo,
        keyframe_indices: tuple[int, ...],
        pivot_px: float,
        pivot_py: float,
        scale: float,
    ) -> None:
        """Draw pendulum arms and bobs at each keyframe with alpha ramp.

        Earlier keyframes are dimmer; later keyframes are brighter.
        """
        n_kf = len(keyframe_indices)
        if n_kf == 0:
            return

        r, g, b = tinfo.color_rgb

        for ki, fi in enumerate(keyframe_indices):
            # Alpha ramp across keyframes
            frac = ki / max(1, n_kf - 1)
            arm_alpha = int(
                FREEZE_ARM_ALPHA_MIN
                + frac * (FREEZE_ARM_ALPHA_MAX - FREEZE_ARM_ALPHA_MIN)
            )

            state = tinfo.trajectory[fi]
            theta1, theta2 = float(state[0]), float(state[1])
            x1, y1, x2, y2 = positions(
                [theta1, theta2, 0.0, 0.0], self._params,
            )

            b1_px = pivot_px + x1 * scale
            b1_py = pivot_py - y1 * scale
            b2_px = pivot_px + x2 * scale
            b2_py = pivot_py - y2 * scale

            # Arms in gray with alpha ramp
            arm_pen = QPen(QColor(220, 220, 220, arm_alpha))
            arm_pen.setWidth(ARM_WIDTH)
            painter.setPen(arm_pen)
            painter.drawLine(
                int(pivot_px), int(pivot_py), int(b1_px), int(b1_py),
            )
            painter.drawLine(
                int(b1_px), int(b1_py), int(b2_px), int(b2_py),
            )

            # Bobs in basin color with matching alpha
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(r, g, b, arm_alpha))
            painter.drawEllipse(
                int(b1_px - FREEZE_BOB_RADIUS),
                int(b1_py - FREEZE_BOB_RADIUS),
                FREEZE_BOB_RADIUS * 2, FREEZE_BOB_RADIUS * 2,
            )
            painter.drawEllipse(
                int(b2_px - FREEZE_BOB_RADIUS),
                int(b2_py - FREEZE_BOB_RADIUS),
                FREEZE_BOB_RADIUS * 2, FREEZE_BOB_RADIUS * 2,
            )

    def _draw_trail(
        self, painter: QPainter, r: int, g: int, b: int,
    ) -> None:
        """Draw the fading polyline trail for the primary trajectory."""
        trail_len = len(self._trail)
        if trail_len <= 1:
            return

        for i in range(1, trail_len):
            alpha = int(255 * i / trail_len)
            trail_pen = QPen(QColor(r, g, b, alpha))
            trail_pen.setWidth(TRAIL_WIDTH)
            painter.setPen(trail_pen)

            x_prev, y_prev = self._trail[i - 1]
            x_curr, y_curr = self._trail[i]
            painter.drawLine(
                int(x_prev), int(y_prev),
                int(x_curr), int(y_curr),
            )
