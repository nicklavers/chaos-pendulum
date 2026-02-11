"""Tapered arc geometry for winding number indicators.

Pure functions for computing and drawing tapered arc segments that
visually encode winding numbers: |n| arcs per circle, each spanning
360/|n| degrees, fat at the tail and tapering to a point. No separate
arrowheads — the taper itself conveys direction.

Convention: 0 degrees = 12 o'clock (top), positive = clockwise.
Direction: positive n = CW rotation, negative n = CCW rotation.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF


# Drawing constants
TAIL_WIDTH = 6.5
DOTTED_STROKE_WIDTH = 1.5
ARC_GAP_DEGREES = 4.0
SINGLE_GAP_DEGREES = 40.0
ARC_INSET = 4.0
OUTLINE_WIDTH = 1.5
TAPER_STEPS = 32


class TaperedArc(NamedTuple):
    """A single tapered arc segment.

    Attributes:
        start_deg: Start angle in degrees (0=top, positive=clockwise).
        span_deg: Angular span in degrees (always positive).
        clockwise: True if arc direction is clockwise (negative winding).
    """

    start_deg: float
    span_deg: float
    clockwise: bool


def compute_tapered_arcs(n: int) -> list[TaperedArc]:
    """Compute tapered arc segments for a given winding number.

    Args:
        n: Winding number. |n| determines count; sign determines direction.
           Positive = CW (clockwise=True), negative = CCW (clockwise=False).
           n=0 returns empty list (caller should draw dotted circle).

    Returns:
        List of TaperedArc segments evenly distributed around the circle,
        with a jaunty rotation offset so all winding numbers start their
        first arc at the same angle as n=1.
    """
    if n == 0:
        return []

    count = abs(n)
    clockwise = n > 0
    gap = SINGLE_GAP_DEGREES if count == 1 else ARC_GAP_DEGREES
    span = (360.0 - count * gap) / count

    # Rotation offset: n=1 starts at single_gap/2 off vertical.
    # For n>=2, add offset so all share that jaunty angle.
    rotation_offset = (SINGLE_GAP_DEGREES - gap) / 2.0

    arcs: list[TaperedArc] = []
    for i in range(count):
        start = i * (span + gap) + gap / 2.0 + rotation_offset
        arcs.append(TaperedArc(start_deg=start, span_deg=span, clockwise=clockwise))
    return arcs


def build_tapered_arc_polygon(
    cx: float,
    cy: float,
    arc_r: float,
    start_deg: float,
    span_deg: float,
    clockwise: bool,
    tail_width: float = TAIL_WIDTH,
    steps: int = TAPER_STEPS,
) -> QPolygonF:
    """Build a filled polygon tracing a tapered arc.

    The polygon is fat/squared at the tail (t=0) and tapers linearly
    to a point at the tip (t=1). Convention: 0 deg = top, positive = CW.

    Args:
        cx: Circle center x.
        cy: Circle center y.
        arc_r: Arc radius (center-line of the tapered shape).
        start_deg: Starting angle in degrees.
        span_deg: Angular span in degrees (always positive).
        clockwise: True if arc travels clockwise.
        tail_width: Full width at the squared tail end.
        steps: Number of polygon subdivision steps.

    Returns:
        QPolygonF suitable for QPainter.drawPolygon().
    """
    outer_points: list[QPointF] = []
    inner_points: list[QPointF] = []

    for i in range(steps + 1):
        t = i / steps

        if clockwise:
            angle_deg = start_deg + t * span_deg
        else:
            angle_deg = start_deg - t * span_deg

        angle_rad = math.radians(angle_deg)

        px = cx + arc_r * math.sin(angle_rad)
        py = cy - arc_r * math.cos(angle_rad)

        norm_x = math.sin(angle_rad)
        norm_y = -math.cos(angle_rad)

        half_w = tail_width * (1.0 - t)

        outer_points.append(QPointF(px + norm_x * half_w, py + norm_y * half_w))
        inner_points.append(QPointF(px - norm_x * half_w, py - norm_y * half_w))

    polygon = QPolygonF()
    for pt in outer_points:
        polygon.append(pt)
    for pt in reversed(inner_points):
        polygon.append(pt)

    return polygon


def draw_tapered_arcs(
    painter: QPainter,
    cx: float,
    cy: float,
    radius: float,
    n: int,
    fill_color: QColor,
    outline_color: QColor,
) -> None:
    """Draw tapered arcs (or dotted circle for n=0) inside a circle.

    Args:
        painter: Active QPainter with antialiasing enabled.
        cx: Circle center x.
        cy: Circle center y.
        radius: Circle radius (arcs drawn inset from this).
        n: Winding number (0 for dotted circle).
        fill_color: Basin color for the tapered arc fill.
        outline_color: Dark outline color around each arc polygon.
    """
    arc_r = radius - ARC_INSET

    if n == 0:
        _draw_dotted_circle(painter, cx, cy, arc_r, fill_color)
        return

    arcs = compute_tapered_arcs(n)
    for arc in arcs:
        polygon = build_tapered_arc_polygon(
            cx, cy, arc_r,
            arc.start_deg, arc.span_deg, arc.clockwise,
        )
        painter.setPen(QPen(outline_color, OUTLINE_WIDTH))
        painter.setBrush(fill_color)
        painter.drawPolygon(polygon)


def _draw_dotted_circle(
    painter: QPainter,
    cx: float,
    cy: float,
    arc_r: float,
    color: QColor,
) -> None:
    """Draw a thin dotted circle outline for n=0."""
    pen = QPen(color, DOTTED_STROKE_WIDTH)
    pen.setStyle(Qt.PenStyle.DotLine)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(QRectF(cx - arc_r, cy - arc_r, arc_r * 2, arc_r * 2))
