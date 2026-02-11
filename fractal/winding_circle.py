"""WindingCircle: small widget showing winding numbers inside a colored circle.

Displays the (n1, n2) winding pair as text inside a filled circle whose
color matches the active winding colormap. Used in the inspect column
to replace the "At t" diagram in basin mode.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget

from fractal.winding import get_single_winding_color


# Drawing constants
BACKGROUND_COLOR = QColor(30, 30, 45)
CIRCLE_RADIUS_FRAC = 0.38  # fraction of min(width, height)
TEXT_COLOR_LIGHT = QColor(240, 240, 240)
TEXT_COLOR_DARK = QColor(30, 30, 30)
LABEL_COLOR = QColor(180, 180, 190)


def _perceived_brightness(r: int, g: int, b: int) -> float:
    """Compute perceived brightness (0-255) for text color selection."""
    return 0.299 * r + 0.587 * g + 0.114 * b


class WindingCircle(QWidget):
    """Small widget that draws a colored circle with winding numbers inside."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(120, 120)

        self._n1: int | None = None
        self._n2: int | None = None
        self._fill_color = BACKGROUND_COLOR
        self._label = "Winding"

    def set_winding(
        self,
        n1: int,
        n2: int,
        colormap_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """Update the displayed winding numbers and fill color.

        Args:
            n1: Winding number for theta1.
            n2: Winding number for theta2.
            colormap_fn: Winding colormap function (for color lookup).
        """
        self._n1 = n1
        self._n2 = n2
        b, g, r, _a = get_single_winding_color(n1, n2, colormap_fn)
        self._fill_color = QColor(r, g, b)
        self.update()

    def clear_winding(self) -> None:
        """Reset to blank state."""
        self._n1 = None
        self._n2 = None
        self._fill_color = BACKGROUND_COLOR
        self.update()

    def set_label(self, text: str) -> None:
        """Set the label shown above the circle."""
        self._label = text
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the colored circle with winding numbers."""
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
            painter.setPen(LABEL_COLOR)
            fm = QFontMetrics(font)
            label_height = fm.height() + 4
            tw = fm.horizontalAdvance(self._label)
            painter.drawText(int((w - tw) / 2), fm.ascent() + 2, self._label)

        # Compute circle geometry
        usable_h = h - label_height - 8
        usable_w = w - 8
        radius = int(min(usable_w, usable_h) * CIRCLE_RADIUS_FRAC)
        cx = w / 2
        cy = label_height + (h - label_height) / 2

        # Draw filled circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._fill_color)
        painter.drawEllipse(
            int(cx - radius), int(cy - radius),
            radius * 2, radius * 2,
        )

        # Draw winding text inside
        if self._n1 is not None and self._n2 is not None:
            # Choose text color based on fill brightness
            r = self._fill_color.red()
            g = self._fill_color.green()
            b = self._fill_color.blue()
            brightness = _perceived_brightness(r, g, b)
            text_color = TEXT_COLOR_DARK if brightness > 128 else TEXT_COLOR_LIGHT

            painter.setPen(text_color)

            # Draw n1, n2 on two lines
            text_font = QFont("Helvetica", 13, QFont.Weight.Bold)
            painter.setFont(text_font)
            fm = QFontMetrics(text_font)

            line1 = f"n\u2081={self._n1}"
            line2 = f"n\u2082={self._n2}"

            tw1 = fm.horizontalAdvance(line1)
            tw2 = fm.horizontalAdvance(line2)
            line_h = fm.height()
            total_h = 2 * line_h

            y_start = cy - total_h / 2 + fm.ascent()
            painter.drawText(int(cx - tw1 / 2), int(y_start), line1)
            painter.drawText(int(cx - tw2 / 2), int(y_start + line_h), line2)

        painter.end()
