"""TrajectoryIndicator: vertical Venn diagram winding number indicator.

Displays two linked rings (vertical Venn diagram) where each ring's
tapered arcs encode the winding number: |n| arcs each spanning 360/|n|
degrees, fat at the tail and tapering to a point. Top ring = arm 1 (n1),
bottom ring = arm 2 (n2). Click to foreground; hover X button to remove.

Draw order: gray bg circles, then top arcs (behind), then bottom arcs
(in front), then digits on top of everything.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, QSize, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen,
    QMouseEvent, QPaintEvent,
)
from PyQt6.QtWidgets import QWidget

from fractal.arrow_arc import draw_tapered_arcs


# Layout constants
CIRCLE_DIAMETER = 32
CIRCLE_RADIUS = CIRCLE_DIAMETER / 2
OVERLAP_FRAC = 0.35
OVERLAP_PX = int(CIRCLE_DIAMETER * OVERLAP_FRAC)  # ~11px
VENN_SPAN = 2 * CIRCLE_DIAMETER - OVERLAP_PX

HIGHLIGHT_BORDER = 3
HOVER_BORDER = 2
REMOVE_BTN_SIZE = 14
TOP_PADDING = 2

WIDGET_WIDTH = CIRCLE_DIAMETER + 8   # 40
MIN_HEIGHT = VENN_SPAN + 28          # ~81

# Neutral gray background for the circles
NEUTRAL_BG = QColor(55, 55, 70)
OUTLINE_COLOR = QColor(20, 20, 30)

# Digit rendering
DIGIT_FONT_SIZE = 13
DIGIT_OFFSET_FRAC = 0.15  # fraction of diameter; smaller = closer to center


def _format_winding(n: int) -> str:
    """Format winding number with + prefix for positive values."""
    if n > 0:
        return f"+{n}"
    return str(n)


class TrajectoryIndicator(QWidget):
    """Vertical Venn diagram indicator for a pinned trajectory.

    Two linked rings where tapered arcs encode winding numbers.
    Basin-colored arcs with dark outline on neutral gray background.

    Signals:
        clicked(row_id): Emitted when the indicator is clicked (foreground).
        remove_clicked(row_id): Emitted when the X button is clicked.
        hovered(row_id): Emitted on mouse enter.
        unhovered(row_id): Emitted on mouse leave.
    """

    clicked = pyqtSignal(str)
    remove_clicked = pyqtSignal(str)
    hovered = pyqtSignal(str)
    unhovered = pyqtSignal(str)

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

    def set_winding(self, n1: int, n2: int) -> None:
        """Update winding numbers (for definition toggle changes)."""
        self._n1 = n1
        self._n2 = n2
        self.update()

    def sizeHint(self) -> QSize:
        """Preferred size: Venn diagram + text below."""
        return QSize(WIDGET_WIDTH, MIN_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        """Minimum size."""
        return QSize(WIDGET_WIDTH - 4, MIN_HEIGHT - 4)

    # --- Geometry helpers ---

    def _circle_centers(self) -> tuple[float, float, float]:
        """Return (cx, cy_top, cy_bottom) for the two circles."""
        cx = self.width() / 2.0
        cy_top = TOP_PADDING + CIRCLE_RADIUS
        cy_bottom = cy_top + CIRCLE_DIAMETER - OVERLAP_PX
        return cx, cy_top, cy_bottom

    # --- Painting ---

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the Venn diagram with tapered arcs and winding digits.

        Multi-pass draw order:
        1. Gray bg circles (behind everything)
        2. Top circle arcs (behind bottom's arcs)
        3. Bottom circle arcs (in front of top's)
        4. Digits on top of everything
        5. Highlight/hover border
        6. Angle text and X button
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cx, cy_top, cy_bottom = self._circle_centers()
        r = CIRCLE_RADIUS
        basin_color = QColor(*self._color_rgb)
        digit_offset = CIRCLE_DIAMETER * DIGIT_OFFSET_FRAC

        # Pass 1: Gray background circles
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(NEUTRAL_BG)
        painter.drawEllipse(
            QRectF(cx - r, cy_bottom - r, CIRCLE_DIAMETER, CIRCLE_DIAMETER),
        )
        painter.drawEllipse(
            QRectF(cx - r, cy_top - r, CIRCLE_DIAMETER, CIRCLE_DIAMETER),
        )

        # Pass 2: Top circle arcs (drawn first = behind bottom's)
        draw_tapered_arcs(
            painter, cx, cy_top, r, self._n1, basin_color, OUTLINE_COLOR,
        )

        # Pass 3: Bottom circle arcs (drawn second = in front of top's)
        draw_tapered_arcs(
            painter, cx, cy_bottom, r, self._n2, basin_color, OUTLINE_COLOR,
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

        # Pass 5: Highlight / hover border around unified figure-8 contour
        if self._highlighted or self._hovered:
            border_path = QPainterPath()
            inset = 1.0
            d = CIRCLE_DIAMETER - 2 * inset
            border_path.addEllipse(
                QRectF(cx - r + inset, cy_top - r + inset, d, d),
            )
            border_path.addEllipse(
                QRectF(cx - r + inset, cy_bottom - r + inset, d, d),
            )
            unified = border_path.simplified()

            width = HIGHLIGHT_BORDER if self._highlighted else HOVER_BORDER
            alpha = 255 if self._highlighted else 120
            border_pen = QPen(QColor(255, 255, 255, alpha), width)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(unified)

        # Pass 6a: Initial angles as tiny text below the bottom circle
        angle_font = QFont("Helvetica", 7)
        painter.setFont(angle_font)
        painter.setPen(QColor(170, 170, 170))
        afm = QFontMetrics(angle_font)

        deg1 = math.degrees(self._theta1_init)
        deg2 = math.degrees(self._theta2_init)
        angle_text = f"{deg1:.0f}\u00b0, {deg2:.0f}\u00b0"
        atw = afm.horizontalAdvance(angle_text)
        angle_y = int(cy_bottom + r + 4 + afm.ascent())
        painter.drawText(int(cx - atw / 2), angle_y, angle_text)

        # Pass 6b: X button (top-right of top circle, on hover)
        if self._hovered:
            self._draw_remove_button(painter, cx, cy_top, r)

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

    @staticmethod
    def _draw_remove_button(
        painter: QPainter,
        cx: float,
        cy_top: float,
        r: float,
    ) -> None:
        """Draw the X remove button at top-right of top circle."""
        xr = REMOVE_BTN_SIZE / 2
        xc_x = cx + r - 4
        xc_y = cy_top - r + 4
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
            int(xc_x - xtw / 2), int(xc_y + xfm.ascent() / 2 - 1),
            "\u2715",
        )

    # --- Mouse events ---

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle click: check if on X button or inside either circle."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        cx, cy_top, cy_bottom = self._circle_centers()
        r = CIRCLE_RADIUS
        mx, my = event.position().x(), event.position().y()

        # Check X button area (top-right of top circle)
        xc_x = cx + r - 4
        xc_y = cy_top - r + 4
        dist_to_x = math.sqrt((mx - xc_x) ** 2 + (my - xc_y) ** 2)
        if dist_to_x <= REMOVE_BTN_SIZE / 2 + 2:
            self.remove_clicked.emit(self._row_id)
            return

        # Click anywhere on the widget foregrounds this trajectory
        self.clicked.emit(self._row_id)

    def enterEvent(self, event) -> None:
        """Mouse entered: show hover effects."""
        self._hovered = True
        self.update()
        self.hovered.emit(self._row_id)

    def leaveEvent(self, event) -> None:
        """Mouse left: hide hover effects."""
        self._hovered = False
        self.update()
        self.unhovered.emit(self._row_id)

    def _init_ui(self) -> None:
        """Set fixed width and minimum height."""
        self.setFixedWidth(WIDGET_WIDTH)
        self.setMinimumHeight(MIN_HEIGHT - 4)
