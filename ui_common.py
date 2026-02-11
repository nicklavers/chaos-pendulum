"""Shared UI widgets used by both pendulum and fractal modes.

Contains LoadingOverlay, PhysicsParamsWidget, and reusable slider helpers.
"""

import math

from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QGroupBox,
    QSlider, QLabel,
)

from simulation import DoublePendulumParams


# ---------------------------------------------------------------------------
# Slider helpers
# ---------------------------------------------------------------------------

def make_slider(minimum, maximum, value, resolution=100):
    """Create an integer QSlider that maps to float values.

    The slider range is [minimum*resolution, maximum*resolution].
    """
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(int(minimum * resolution))
    slider.setMaximum(int(maximum * resolution))
    slider.setValue(int(value * resolution))
    slider.resolution = resolution
    return slider


def slider_value(slider):
    """Read the float value from a slider created by make_slider."""
    return slider.value() / slider.resolution


# ---------------------------------------------------------------------------
# PhysicsParamsWidget
# ---------------------------------------------------------------------------

class PhysicsParamsWidget(QWidget):
    """Grouped sliders for the shared physics parameters (m1, m2, l1, l2, mu).

    Emits no signals itself; call get_params() to read current values.
    The parent can connect slider.valueChanged to detect changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.m1_slider = make_slider(0.1, 5.0, 1.0)
        self.m2_slider = make_slider(0.1, 5.0, 1.4)
        self.l1_slider = make_slider(0.1, 3.0, 1.0)
        self.l2_slider = make_slider(0.1, 3.0, 0.3)
        self.friction_slider = make_slider(0.0, 5.0, 0.38)

        self._add_row(layout, 0, "m\u2081", self.m1_slider, " kg")
        self._add_row(layout, 1, "m\u2082", self.m2_slider, " kg")
        self._add_row(layout, 2, "l\u2081", self.l1_slider, " m")
        self._add_row(layout, 3, "l\u2082", self.l2_slider, " m")
        self._add_row(layout, 4, "\u03bc", self.friction_slider)

    def _add_row(self, layout, row, label_text, slider, unit=""):
        label = QLabel(label_text)
        value_label = QLabel()
        value_label.setMinimumWidth(55)
        value_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

        def _update(_val, vl=value_label, sl=slider, u=unit):
            vl.setText(f"{slider_value(sl):.2f}{u}")

        slider.valueChanged.connect(_update)
        _update(slider.value())

    def get_params(self):
        """Return a DoublePendulumParams from the current slider values."""
        return DoublePendulumParams(
            m1=slider_value(self.m1_slider),
            m2=slider_value(self.m2_slider),
            l1=slider_value(self.l1_slider),
            l2=slider_value(self.l2_slider),
            friction=slider_value(self.friction_slider),
        )

    def set_params(self, params):
        """Set slider positions from a DoublePendulumParams."""
        self.m1_slider.setValue(int(params.m1 * self.m1_slider.resolution))
        self.m2_slider.setValue(int(params.m2 * self.m2_slider.resolution))
        self.l1_slider.setValue(int(params.l1 * self.l1_slider.resolution))
        self.l2_slider.setValue(int(params.l2 * self.l2_slider.resolution))
        self.friction_slider.setValue(
            int(params.friction * self.friction_slider.resolution)
        )


# ---------------------------------------------------------------------------
# LoadingOverlay
# ---------------------------------------------------------------------------

class LoadingOverlay(QWidget):
    """Semi-transparent overlay with an orbital loading animation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.message = "Simulating..."
        self.t = 0.0
        self._timer = QTimer()
        self._timer.setInterval(16)  # ~60 fps
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self, message="Simulating..."):
        self.message = message
        self.t = 0.0
        if self.parentWidget():
            self.resize(self.parentWidget().size())
        self.show()
        self.raise_()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        self.t += 0.016
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Dim background
        painter.fillRect(self.rect(), QColor(20, 20, 30, 180))

        cx, cy = w / 2, h / 2 - 15

        # --- Moon orbital parameters ---
        big_orbit_a, big_orbit_b = 36, 22
        big_rot = math.radians(50)
        big_period = 2.0
        big_r = 7
        big_angle = 2 * math.pi * self.t / big_period

        big_lx = big_orbit_a * math.cos(big_angle)
        big_ly = big_orbit_b * math.sin(big_angle)
        big_sx = big_lx * math.cos(big_rot) - big_ly * math.sin(big_rot)
        big_sy = big_lx * math.sin(big_rot) + big_ly * math.cos(big_rot)
        big_behind = math.sin(big_angle) > 0

        sm_orbit_a, sm_orbit_b = 52, 14
        sm_rot = math.radians(-5)
        sm_period = 3.2
        sm_r = 4
        sm_angle = 2 * math.pi * self.t / sm_period

        sm_lx = sm_orbit_a * math.cos(sm_angle)
        sm_ly = sm_orbit_b * math.sin(sm_angle)
        sm_sx = sm_lx * math.cos(sm_rot) - sm_ly * math.sin(sm_rot)
        sm_sy = sm_lx * math.sin(sm_rot) + sm_ly * math.cos(sm_rot)
        sm_behind = math.sin(sm_angle) > 0

        # Planet wobble
        wobble_x = -(big_sx * 0.05 + sm_sx * 0.02)
        wobble_y = -(big_sy * 0.05 + sm_sy * 0.02)
        px, py = cx + wobble_x, cy + wobble_y

        planet_r = 24
        ring_a, ring_b = 42, 10
        ring_angle_deg = 20

        ring_pen = QPen(QColor(190, 170, 110))
        ring_pen.setWidthF(5)
        ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        ring_rect = QRectF(-ring_a, -ring_b, 2 * ring_a, 2 * ring_b)

        # Behind moons
        painter.setPen(Qt.PenStyle.NoPen)
        if big_behind:
            painter.setBrush(QBrush(QColor(160, 175, 195)))
            painter.drawEllipse(QPointF(px + big_sx, py + big_sy), big_r, big_r)
        if sm_behind:
            painter.setBrush(QBrush(QColor(170, 170, 180)))
            painter.drawEllipse(QPointF(px + sm_sx, py + sm_sy), sm_r, sm_r)

        # Back ring arc
        painter.save()
        painter.translate(px, py)
        painter.rotate(ring_angle_deg)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(ring_rect, 0, 180 * 16)
        painter.restore()

        # Planet body
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(210, 160, 80)))
        painter.drawEllipse(QPointF(px, py), planet_r, planet_r)

        # Bands
        clip = QPainterPath()
        clip.addEllipse(QPointF(px, py), planet_r, planet_r)
        painter.save()
        painter.setClipPath(clip)
        painter.setBrush(QBrush(QColor(190, 140, 60)))
        painter.drawRect(QRectF(px - planet_r - 1, py - 4, 2 * planet_r + 2, 7))
        painter.setBrush(QBrush(QColor(200, 155, 75)))
        painter.drawRect(QRectF(px - planet_r - 1, py + 8, 2 * planet_r + 2, 4))
        painter.restore()

        # Front ring arc
        painter.save()
        painter.translate(px, py)
        painter.rotate(ring_angle_deg)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(ring_rect, 180 * 16, 180 * 16)
        painter.restore()

        # Front moons
        painter.setPen(Qt.PenStyle.NoPen)
        if not big_behind:
            painter.setBrush(QBrush(QColor(160, 175, 195)))
            painter.drawEllipse(QPointF(px + big_sx, py + big_sy), big_r, big_r)
        if not sm_behind:
            painter.setBrush(QBrush(QColor(170, 170, 180)))
            painter.drawEllipse(QPointF(px + sm_sx, py + sm_sy), sm_r, sm_r)

        # Text
        text_wobble_y = wobble_y * 3.0
        painter.setPen(QColor(255, 255, 255, 200))
        font = QFont("Chalkboard SE")
        font.setPointSizeF(16)
        font.setBold(True)
        painter.setFont(font)
        text_rect = QRectF(0, cy + 65 + text_wobble_y, w, 40)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self.message,
        )

        painter.end()
