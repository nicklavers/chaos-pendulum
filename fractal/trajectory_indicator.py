"""TrajectoryIndicator: small clickable circle for the indicator row.

Displays a filled circle colored by the trajectory's basin color,
with winding numbers (n1, n2) inside. Initial angles shown as tiny
text below. Click to foreground; hover X button to remove.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QFontMetrics, QPainter, QPen, QMouseEvent, QPaintEvent,
)
from PyQt6.QtWidgets import QWidget


# Layout constants
CIRCLE_DIAMETER = 44
HIGHLIGHT_BORDER = 3
HOVER_BORDER = 2
REMOVE_BTN_SIZE = 14


class TrajectoryIndicator(QWidget):
    """Small colored circle indicator for a pinned trajectory.

    Signals:
        clicked(row_id): Emitted when the circle is clicked (foreground).
        remove_clicked(row_id): Emitted when the X button is clicked.
    """

    clicked = pyqtSignal(str)
    remove_clicked = pyqtSignal(str)

    def __init__(
        self,
        row_id: str,
        n1: int,
        n2: int,
        theta1_init: float,
        theta2_init: float,
        color_rgb: tuple[int, int, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._row_id = row_id
        self._n1 = n1
        self._n2 = n2
        self._theta1_init = theta1_init
        self._theta2_init = theta2_init
        self._color_rgb = color_rgb
        self._highlighted = False
        self._hovered = False

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._init_ui()

    @property
    def row_id(self) -> str:
        """Unique identifier for this trajectory."""
        return self._row_id

    def set_highlighted(self, highlighted: bool) -> None:
        """Toggle thick white border when this is the primary trajectory."""
        self._highlighted = highlighted
        self.update()

    def set_color(self, color_rgb: tuple[int, int, int]) -> None:
        """Update the fill color (for colormap changes)."""
        self._color_rgb = color_rgb
        self.update()

    def sizeHint(self) -> QSize:
        """Preferred size: circle + text below."""
        return QSize(CIRCLE_DIAMETER + 8, CIRCLE_DIAMETER + 28)

    def minimumSizeHint(self) -> QSize:
        """Minimum size."""
        return QSize(CIRCLE_DIAMETER + 4, CIRCLE_DIAMETER + 24)

    # --- Painting ---

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the colored circle with winding numbers, and angle text below."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()

        # Circle center
        cx = w / 2
        cy = CIRCLE_DIAMETER / 2 + 2
        r = CIRCLE_DIAMETER / 2

        # Fill circle with basin color
        fill_color = QColor(*self._color_rgb)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill_color)
        painter.drawEllipse(int(cx - r), int(cy - r), CIRCLE_DIAMETER, CIRCLE_DIAMETER)

        # Highlight border (white) when primary
        if self._highlighted:
            border_pen = QPen(QColor(255, 255, 255), HIGHLIGHT_BORDER)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(cx - r + 1), int(cy - r + 1),
                CIRCLE_DIAMETER - 2, CIRCLE_DIAMETER - 2,
            )
        elif self._hovered:
            border_pen = QPen(QColor(200, 200, 200, 120), HOVER_BORDER)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                int(cx - r + 1), int(cy - r + 1),
                CIRCLE_DIAMETER - 2, CIRCLE_DIAMETER - 2,
            )

        # Winding numbers inside circle — pick contrasting text color
        brightness = (
            0.299 * self._color_rgb[0]
            + 0.587 * self._color_rgb[1]
            + 0.114 * self._color_rgb[2]
        )
        text_color = QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)

        winding_font = QFont("Helvetica", 8, QFont.Weight.Bold)
        painter.setFont(winding_font)
        painter.setPen(text_color)
        fm = QFontMetrics(winding_font)

        line1 = f"n\u2081={self._n1}"
        line2 = f"n\u2082={self._n2}"
        tw1 = fm.horizontalAdvance(line1)
        tw2 = fm.horizontalAdvance(line2)
        line_h = fm.height()

        # Two lines centered vertically in circle
        text_y_start = cy - line_h + fm.ascent() - 1
        painter.drawText(int(cx - tw1 / 2), int(text_y_start), line1)
        painter.drawText(int(cx - tw2 / 2), int(text_y_start + line_h), line2)

        # Initial angles as tiny text below the circle
        angle_font = QFont("Helvetica", 7)
        painter.setFont(angle_font)
        painter.setPen(QColor(170, 170, 170))
        afm = QFontMetrics(angle_font)

        deg1 = math.degrees(self._theta1_init)
        deg2 = math.degrees(self._theta2_init)
        angle_text = f"{deg1:.0f}\u00b0, {deg2:.0f}\u00b0"
        atw = afm.horizontalAdvance(angle_text)
        angle_y = int(cy + r + 4 + afm.ascent())
        painter.drawText(int(cx - atw / 2), angle_y, angle_text)

        # X button (top-right of circle, shown on hover)
        if self._hovered:
            xr = REMOVE_BTN_SIZE / 2
            xc_x = cx + r - 4
            xc_y = cy - r + 4
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(60, 60, 60, 200))
            painter.drawEllipse(
                int(xc_x - xr), int(xc_y - xr),
                REMOVE_BTN_SIZE, REMOVE_BTN_SIZE,
            )
            x_font = QFont("Helvetica", 8, QFont.Weight.Bold)
            painter.setFont(x_font)
            painter.setPen(QColor(255, 100, 100))
            xfm = QFontMetrics(x_font)
            xtw = xfm.horizontalAdvance("\u2715")
            painter.drawText(
                int(xc_x - xtw / 2), int(xc_y + xfm.ascent() / 2 - 1), "\u2715",
            )

        painter.end()

    # --- Mouse events ---

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle click: check if on X button or circle."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        w = self.width()
        cx = w / 2
        cy = CIRCLE_DIAMETER / 2 + 2
        r = CIRCLE_DIAMETER / 2

        # Check if click is on the X button area (top-right)
        mx, my = event.position().x(), event.position().y()
        xc_x = cx + r - 4
        xc_y = cy - r + 4
        dist_to_x = math.sqrt((mx - xc_x) ** 2 + (my - xc_y) ** 2)
        if dist_to_x <= REMOVE_BTN_SIZE / 2 + 2:
            self.remove_clicked.emit(self._row_id)
            return

        # Otherwise, foreground click
        self.clicked.emit(self._row_id)

    def enterEvent(self, event) -> None:
        """Mouse entered: show hover effects."""
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        """Mouse left: hide hover effects."""
        self._hovered = False
        self.update()

    def _init_ui(self) -> None:
        """Set fixed width and minimum height."""
        self.setFixedWidth(CIRCLE_DIAMETER + 8)
        self.setMinimumHeight(CIRCLE_DIAMETER + 24)
