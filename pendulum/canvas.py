"""Pendulum canvas: QPainter rendering of the double pendulum.

Extracted from visualization.py without behavioral changes.
"""

import math
from collections import deque

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt6.QtWidgets import QWidget

from simulation import DoublePendulumParams, positions


class PendulumCanvas(QWidget):
    """Custom widget that draws the double pendulum using QPainter."""

    TRAIL_LENGTH = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = DoublePendulumParams()
        self.states = [[0.0, 0.0, 0.0, 0.0]]
        self.trails = [deque(maxlen=self.TRAIL_LENGTH)]
        self.primary_index = 0
        self.show_axes = False
        self.show_velocity = False
        self.setMinimumSize(400, 400)

    @property
    def state(self):
        """Primary trajectory state (backwards compat)."""
        return self.states[self.primary_index]

    @property
    def trail(self):
        """Primary trajectory trail (backwards compat)."""
        return self.trails[self.primary_index]

    def set_states(self, states_list, params, append_trail=True):
        """Update all trajectory states on the canvas."""
        self.states = states_list
        self.params = params
        while len(self.trails) < len(states_list):
            self.trails.append(deque(maxlen=self.TRAIL_LENGTH))
        if append_trail:
            for i, st in enumerate(states_list):
                _, _, x2, y2 = positions(st, params)
                self.trails[i].append((x2, y2))
        self.update()

    def set_state(self, state, params, append_trail=True):
        """Update single-trajectory state (backwards compat)."""
        self.set_states([state], params, append_trail)

    def clear_trails(self):
        self.trails = [deque(maxlen=self.TRAIL_LENGTH)]

    def clear_trail(self):
        self.clear_trails()

    def _to_pixel(self, x, y):
        """Convert physics coords to pixel coords."""
        w, h = self.width(), self.height()
        total_length = self.params.l1 + self.params.l2
        scale = min(w, h) * 0.38 / max(total_length, 0.01)
        cx = w / 2
        cy = h * 0.35
        px = cx + x * scale
        py = cy - y * scale
        return px, py

    def _draw_axes(self, painter):
        """Draw faint concentric distance rings and crosshairs from the pivot."""
        total_length = self.params.l1 + self.params.l2
        w, h = self.width(), self.height()
        scale = min(w, h) * 0.38 / max(total_length, 0.01)
        cx = w / 2
        cy = h * 0.35

        step = 0.5 if total_length <= 3.0 else 1.0
        max_r = total_length + step

        ring_pen = QPen(QColor(255, 255, 255, 30))
        ring_pen.setWidthF(1.0)
        ring_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(ring_pen)

        label_font = QFont()
        label_font.setPointSizeF(9)
        painter.setFont(label_font)

        r = step
        while r <= max_r:
            r_px = r * scale
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), r_px, r_px)

            painter.setPen(QColor(255, 255, 255, 50))
            label = f"{r:.1f} m" if step < 1.0 else f"{r:.0f} m"
            painter.drawText(QPointF(cx + r_px + 3, cy - 2), label)
            r += step

        line_pen = QPen(QColor(255, 255, 255, 20))
        line_pen.setWidthF(1.0)
        painter.setPen(line_pen)
        painter.drawLine(QPointF(0, cy), QPointF(w, cy))
        painter.drawLine(QPointF(cx, 0), QPointF(cx, h))

    def _draw_arrow(self, painter, x0, y0, x1, y1, color):
        """Draw a single arrow from (x0,y0) to (x1,y1) in pixel coords."""
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 2:
            return
        pen = QPen(color)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        head_size = min(8, length * 0.35)
        angle = math.atan2(dy, dx)
        for sign in [1, -1]:
            a = angle + math.pi - sign * math.radians(25)
            hx = x1 + head_size * math.cos(a)
            hy = y1 + head_size * math.sin(a)
            painter.drawLine(QPointF(x1, y1), QPointF(hx, hy))

    def _draw_velocity_arrows(self, painter, bob1_px, bob2_px):
        """Draw velocity arrows at each bob showing initial angular velocities."""
        theta1, theta2, omega1, omega2 = self.state
        l1, l2 = self.params.l1, self.params.l2

        w, h = self.width(), self.height()
        total_length = l1 + l2
        scale = min(w, h) * 0.38 / max(total_length, 0.01)
        arrow_scale = scale * 0.12

        cvx1 = l1 * math.cos(theta1) * omega1
        cvy1 = l1 * math.sin(theta1) * omega1
        cvx2 = l2 * math.cos(theta2) * omega2
        cvy2 = l2 * math.sin(theta2) * omega2

        dx1, dy1 = cvx1 * arrow_scale, -cvy1 * arrow_scale
        dx2, dy2 = cvx2 * arrow_scale, -cvy2 * arrow_scale

        bob1_color = QColor(255, 120, 80)
        bob2_color = QColor(80, 200, 255)
        proj_color = QColor(255, 220, 60)

        bx1, by1 = bob1_px
        self._draw_arrow(painter, bx1, by1, bx1 + dx1, by1 + dy1, bob1_color)

        bx2, by2 = bob2_px
        has_omega1 = math.hypot(dx1, dy1) >= 2
        has_omega2 = math.hypot(dx2, dy2) >= 2

        if has_omega2:
            self._draw_arrow(painter, bx2, by2, bx2 + dx2, by2 + dy2, bob2_color)

        if has_omega1:
            self._draw_arrow(painter, bx2 + dx2, by2 + dy2,
                             bx2 + dx2 + dx1, by2 + dy2 + dy1, bob1_color)

        if has_omega1:
            proj_mag = cvx1 * math.cos(theta2) + cvy1 * math.sin(theta2)
            pdx = math.cos(theta2) * proj_mag * arrow_scale
            pdy = -math.sin(theta2) * proj_mag * arrow_scale

            if math.hypot(pdx, pdy) >= 2:
                base_x = bx2 + dx2
                base_y = by2 + dy2
                self._draw_arrow(painter, base_x, base_y,
                                 base_x + pdx, base_y + pdy, proj_color)

                dash_pen = QPen(QColor(255, 255, 255, 80))
                dash_pen.setWidthF(1.0)
                dash_pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(QPointF(bx2 + dx2 + dx1, by2 + dy2 + dy1),
                                 QPointF(base_x + pdx, base_y + pdy))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(20, 20, 30))

        if self.show_axes:
            self._draw_axes(painter)

        pivot_px = self._to_pixel(0, 0)

        # Secondary trajectories
        if len(self.states) > 1:
            ghost_alpha = max(10, min(40, 800 // len(self.states)))
            bob_r1 = 6 + 4 * self.params.m1
            bob_r2 = 6 + 4 * self.params.m2
            arm_pen = QPen(QColor(200, 200, 200, ghost_alpha))
            arm_pen.setWidthF(2.0)
            for i, st in enumerate(self.states):
                if i == self.primary_index:
                    continue
                sx1, sy1, sx2, sy2 = positions(st, self.params)
                sb1 = self._to_pixel(sx1, sy1)
                sb2 = self._to_pixel(sx2, sy2)
                painter.setPen(arm_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(QPointF(*pivot_px), QPointF(*sb1))
                painter.drawLine(QPointF(*sb1), QPointF(*sb2))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(255, 120, 80, ghost_alpha)))
                painter.drawEllipse(QPointF(*sb1), bob_r1, bob_r1)
                painter.setBrush(QBrush(QColor(80, 200, 255, ghost_alpha)))
                painter.drawEllipse(QPointF(*sb2), bob_r2, bob_r2)

        # Primary trajectory
        primary = self.states[self.primary_index]
        x1, y1, x2, y2 = positions(primary, self.params)
        bob1_px = self._to_pixel(x1, y1)
        bob2_px = self._to_pixel(x2, y2)

        # Trail
        primary_trail = (
            self.trails[self.primary_index]
            if self.primary_index < len(self.trails)
            else deque()
        )
        if len(primary_trail) > 1:
            trail_list = list(primary_trail)
            for i in range(1, len(trail_list)):
                alpha = int(255 * i / len(trail_list))
                pen = QPen(QColor(100, 200, 255, alpha))
                pen.setWidthF(1.5)
                painter.setPen(pen)
                px0, py0 = self._to_pixel(*trail_list[i - 1])
                px1, py1 = self._to_pixel(*trail_list[i])
                painter.drawLine(QPointF(px0, py0), QPointF(px1, py1))

        # Arms
        arm_pen = QPen(QColor(200, 200, 200))
        arm_pen.setWidthF(2.5)
        painter.setPen(arm_pen)
        painter.drawLine(QPointF(*pivot_px), QPointF(*bob1_px))
        painter.drawLine(QPointF(*bob1_px), QPointF(*bob2_px))

        # Pivot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(180, 180, 180)))
        painter.drawEllipse(QPointF(*pivot_px), 5, 5)

        # Bobs
        bob_radius_1 = 6 + 4 * self.params.m1
        bob_radius_2 = 6 + 4 * self.params.m2
        painter.setBrush(QBrush(QColor(255, 120, 80)))
        painter.drawEllipse(QPointF(*bob1_px), bob_radius_1, bob_radius_1)
        painter.setBrush(QBrush(QColor(80, 200, 255)))
        painter.drawEllipse(QPointF(*bob2_px), bob_radius_2, bob_radius_2)

        # Velocity arrows
        if self.show_velocity:
            self._draw_velocity_arrows(painter, bob1_px, bob2_px)

        painter.end()
