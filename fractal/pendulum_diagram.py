"""Pendulum diagram: small stick-figure widget for the inspect tool.

Draws a double-pendulum at given (theta1, theta2) angles with
coloured bobs and a label. Used in pairs (initial / at time t)
in the inspect panel.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget

from simulation import DoublePendulumParams, positions


# Drawing constants
PIVOT_COLOR = QColor(140, 140, 140)
ARM_COLOR = QColor(220, 220, 220)
BOB1_COLOR = QColor(255, 120, 80)   # orange
BOB2_COLOR = QColor(80, 200, 255)   # cyan
BACKGROUND_COLOR = QColor(30, 30, 45)
BOB_RADIUS = 6
PIVOT_RADIUS = 4
ARM_WIDTH = 2


class PendulumDiagram(QWidget):
    """Small widget that draws a double-pendulum stick figure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(120, 120)

        self._theta1 = 0.0
        self._theta2 = 0.0
        self._params = DoublePendulumParams()
        self._label = ""

    def set_state(self, theta1: float, theta2: float) -> None:
        """Update the pendulum angles and repaint."""
        self._theta1 = theta1
        self._theta2 = theta2
        self.update()

    def set_params(self, params: DoublePendulumParams) -> None:
        """Update the physics parameters (arm lengths affect drawing)."""
        self._params = params
        self.update()

    def set_label(self, text: str) -> None:
        """Set the label shown above the diagram."""
        self._label = text
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the pendulum stick figure."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), BACKGROUND_COLOR)

        w = self.width()
        h = self.height()

        # Draw label at top
        label_height = 0
        if self._label:
            font = QFont("Helvetica", 11)
            painter.setFont(font)
            painter.setPen(QColor(180, 180, 190))
            fm = QFontMetrics(font)
            label_height = fm.height() + 4
            tw = fm.horizontalAdvance(self._label)
            painter.drawText(int((w - tw) / 2), fm.ascent() + 2, self._label)

        # Compute pendulum positions
        state = [self._theta1, self._theta2, 0.0, 0.0]
        x1, y1, x2, y2 = positions(state, self._params)

        # Scale to widget: total arm length determines the scale
        total_length = self._params.l1 + self._params.l2
        usable_h = h - label_height - 2 * BOB_RADIUS - 8
        usable_w = w - 2 * BOB_RADIUS - 8
        usable = min(usable_w, usable_h)
        scale = usable / (2.2 * total_length)  # 2.2 gives margin

        # Pivot at center of usable area (pendulum can extend in all directions)
        pivot_px = w / 2
        pivot_py = label_height + (h - label_height) / 2

        # Convert physics coords to pixel (y points down in physics)
        # positions() returns y pointing down from pivot already
        b1_px = pivot_px + x1 * scale
        b1_py = pivot_py - y1 * scale  # negate because Qt y is down, physics y is down
        b2_px = pivot_px + x2 * scale
        b2_py = pivot_py - y2 * scale

        # Draw arms
        arm_pen = QPen(ARM_COLOR)
        arm_pen.setWidth(ARM_WIDTH)
        painter.setPen(arm_pen)
        painter.drawLine(int(pivot_px), int(pivot_py), int(b1_px), int(b1_py))
        painter.drawLine(int(b1_px), int(b1_py), int(b2_px), int(b2_py))

        # Draw pivot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PIVOT_COLOR)
        painter.drawEllipse(
            int(pivot_px - PIVOT_RADIUS),
            int(pivot_py - PIVOT_RADIUS),
            PIVOT_RADIUS * 2,
            PIVOT_RADIUS * 2,
        )

        # Draw bob 1
        painter.setBrush(BOB1_COLOR)
        painter.drawEllipse(
            int(b1_px - BOB_RADIUS),
            int(b1_py - BOB_RADIUS),
            BOB_RADIUS * 2,
            BOB_RADIUS * 2,
        )

        # Draw bob 2
        painter.setBrush(BOB2_COLOR)
        painter.drawEllipse(
            int(b2_px - BOB_RADIUS),
            int(b2_py - BOB_RADIUS),
            BOB_RADIUS * 2,
            BOB_RADIUS * 2,
        )

        painter.end()
