"""WindingCircle: Venn diagram showing winding numbers with tapered arcs.

Displays a vertical Venn diagram (two linked rings) whose tapered arcs
encode the winding count for each pendulum arm. Used in the inspect
column hover section for basin mode.

Draw order: gray bg circles, then top arcs (behind), then bottom arcs
(in front), then digits on top of everything.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget

from fractal.arrow_arc import draw_tapered_arcs
from fractal.winding import get_single_winding_color


# Drawing constants
BACKGROUND_COLOR = QColor(30, 30, 45)
NEUTRAL_BG = QColor(55, 55, 70)
OUTLINE_COLOR = QColor(20, 20, 30)
OVERLAP_FRAC = 0.35
LABEL_COLOR = QColor(180, 180, 190)
DIGIT_FONT_SIZE = 13
DIGIT_OFFSET_FRAC = 0.15


def _format_winding(n: int) -> str:
    """Format winding number with + prefix for positive values."""
    if n > 0:
        return f"+{n}"
    return str(n)


class WindingCircle(QWidget):
    """Venn diagram widget showing winding numbers with tapered arcs."""

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
        """Set the label shown above the Venn diagram."""
        self._label = text
        self.update()

    def paintEvent(self, event) -> None:
        """Draw the Venn diagram with tapered arcs and winding digits.

        Multi-pass draw order:
        1. Gray bg circles (behind everything)
        2. Top circle arcs (behind bottom's arcs)
        3. Bottom circle arcs (in front of top's)
        4. Digits on top of everything
        """
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

        if self._n1 is None or self._n2 is None:
            painter.end()
            return

        # Compute sub-circle geometry to fit within available space
        usable_h = h - label_height - 8
        usable_w = w - 8
        cx = w / 2.0

        # Each sub-circle diameter: fit two circles with overlap vertically
        # total_span = 2*d - overlap = 2*d - 0.35*d = 1.65*d
        max_d_from_h = usable_h / (2.0 - OVERLAP_FRAC)
        max_d_from_w = usable_w
        d = min(max_d_from_h, max_d_from_w)
        radius = d / 2.0
        overlap = d * OVERLAP_FRAC

        total_span = 2 * d - overlap
        venn_top = label_height + (h - label_height - total_span) / 2.0
        cy_top = venn_top + radius
        cy_bottom = cy_top + d - overlap

        basin_color = self._fill_color
        digit_offset = d * DIGIT_OFFSET_FRAC

        # Pass 1: Gray background circles
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(NEUTRAL_BG)
        painter.drawEllipse(QRectF(cx - radius, cy_bottom - radius, d, d))
        painter.drawEllipse(QRectF(cx - radius, cy_top - radius, d, d))

        # Pass 2: Top circle arcs (drawn first = behind bottom's)
        draw_tapered_arcs(
            painter, cx, cy_top, radius, self._n1,
            basin_color, OUTLINE_COLOR,
        )

        # Pass 3: Bottom circle arcs (drawn second = in front of top's)
        draw_tapered_arcs(
            painter, cx, cy_bottom, radius, self._n2,
            basin_color, OUTLINE_COLOR,
        )

        # Pass 4: Digits on top of everything (outlined text)
        self._draw_digit(
            painter, cx, cy_top - digit_offset, self._n1,
            basin_color, OUTLINE_COLOR,
        )
        self._draw_digit(
            painter, cx, cy_bottom + digit_offset, self._n2,
            basin_color, OUTLINE_COLOR,
        )

        painter.end()

    @staticmethod
    def _draw_digit(
        painter: QPainter,
        cx: float,
        cy: float,
        n: int,
        basin_color: QColor,
        outline_color: QColor,
    ) -> None:
        """Draw a winding number digit with outline at (cx, cy)."""
        font = QFont("Helvetica", DIGIT_FONT_SIZE, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        digit = _format_winding(n)
        tw = fm.horizontalAdvance(digit)
        text_x = int(cx - tw / 2)
        text_y = int(cy + fm.ascent() / 2 - 1)

        # Outline (4 offsets)
        painter.setPen(outline_color)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            painter.drawText(text_x + dx, text_y + dy, digit)

        # Fill
        painter.setPen(basin_color)
        painter.drawText(text_x, text_y, digit)
