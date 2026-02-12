"""Fractal canvas: QImage display with rectangle-select zoom and ghost overlay.

Displays the fractal image. Zoom-in is done by clicking and dragging a
fixed-aspect-ratio selection rectangle. Zoom-out is triggered externally
(button in controls), and draws a fading ghost rectangle showing the
previous viewport region.
"""

from __future__ import annotations

import math
import uuid

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QFont, QFontMetrics, QImage,
    QCursor, QPixmap, QPainterPath, QPolygonF,
)
from PyQt6.QtWidgets import QWidget

from fractal.compute import FractalViewport
from fractal.coloring import numpy_to_qimage
from fractal.winding import (
    winding_to_argb, build_winding_legend, WINDING_COLORMAPS,
    winding_basin_hash,
)


# Zoom limits
MIN_SPAN = 0.001       # radians
MAX_SPAN = 2 * math.pi  # radians
ZOOM_OUT_FACTOR = 2.0  # how much to zoom out per button press

# Tool modes
TOOL_ZOOM = "zoom"
TOOL_PAN = "pan"
TOOL_INSPECT = "inspect"

# Axis label layout
AXIS_MARGIN_LEFT = 72      # pixels reserved for θ₂ labels
AXIS_MARGIN_BOTTOM = 52    # pixels reserved for θ₁ labels
AXIS_LABEL_COLOR = QColor(160, 160, 170)
AXIS_LINE_COLOR = QColor(255, 255, 255, 50)
PI_LINE_COLOR = QColor(0, 0, 0)
TICK_LENGTH = 5

# Ghost rectangle animation
GHOST_FADE_MS = 2000       # total fade duration
GHOST_FADE_TICK_MS = 33    # ~30 fps fade animation
GHOST_INITIAL_ALPHA = 220  # starting opacity (0-255)


# Inspect cursor
CURSOR_SIZE = 24         # pixmap side length
CURSOR_HOTSPOT_X = 3     # tip position within pixmap
CURSOR_HOTSPOT_Y = 1
CURSOR_DEFAULT_COLOR = QColor(180, 180, 190)


def _build_inspect_cursor(fill: QColor) -> QCursor:
    """Build a small arrow cursor with a black outline and the given fill.

    The arrow points up-left, with the tip at (CURSOR_HOTSPOT_X, CURSOR_HOTSPOT_Y).

    Args:
        fill: Interior fill color for the arrow.

    Returns:
        QCursor with the custom pixmap and correct hotspot.
    """
    pixmap = QPixmap(CURSOR_SIZE, CURSOR_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    # Arrow polygon: tip at top-left, body extends down-right
    arrow = QPolygonF([
        QPointF(3, 1),    # tip
        QPointF(3, 17),   # left edge bottom
        QPointF(7, 13),   # notch
        QPointF(12, 20),  # tail bottom-right
        QPointF(15, 18),  # tail top-right
        QPointF(9, 11),   # notch inner
        QPointF(15, 5),   # right wing
        QPointF(3, 1),    # close
    ])

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Black outline
    outline_pen = QPen(QColor(0, 0, 0), 1.5)
    outline_pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(outline_pen)
    painter.setBrush(fill)
    painter.drawPolygon(arrow)

    painter.end()
    return QCursor(pixmap, CURSOR_HOTSPOT_X, CURSOR_HOTSPOT_Y)


def render_cube_to_qimage(cube_slice, colormap_fn) -> QImage:
    """Render a CubeSlice to a QImage using the given winding colormap.

    Extracts the rendering logic from display_basin_from_cube into a
    reusable function for use in pan background etc.

    Args:
        cube_slice: CubeSlice with n1 (R,R) int8, n2 (R,R) int8,
            brightness (R,R) uint8.
        colormap_fn: Winding colormap function mapping (n1, n2) → BGRA.

    Returns:
        QImage with the rendered basin preview.
    """
    n1_2d = cube_slice.n1
    n2_2d = cube_slice.n2
    brightness_2d = cube_slice.brightness
    R = n1_2d.shape[0]

    # Flatten for colormap lookup
    n1_flat = n1_2d.ravel().astype(np.int32)
    n2_flat = n2_2d.ravel().astype(np.int32)

    # Apply winding colormap → (N, 4) uint8 BGRA
    pixels = colormap_fn(n1_flat, n2_flat)

    # Apply pre-baked brightness: scale [0, 255] → [0.3, 1.0] float
    brightness_flat = brightness_2d.ravel().astype(np.float32) / 255.0
    brightness_scaled = np.float32(0.3) + np.float32(0.7) * brightness_flat

    # Modulate BGR channels (leave alpha alone)
    result = pixels.copy()
    bgr = result[:, :3].astype(np.float32) * brightness_scaled[:, np.newaxis]
    result[:, :3] = bgr.astype(np.uint8)

    # Reshape to image
    argb = result.reshape(R, R, 4)
    return numpy_to_qimage(argb)


def _format_angle(rad: float) -> str:
    """Format a radian value as a human-readable label.

    Uses multiples of pi where close, otherwise plain degrees.
    """
    # Check common pi fractions
    for numer, denom, label in [
        (0, 1, "0"),
        (1, 4, "\u03c0/4"),
        (1, 3, "\u03c0/3"),
        (1, 2, "\u03c0/2"),
        (2, 3, "2\u03c0/3"),
        (3, 4, "3\u03c0/4"),
        (1, 1, "\u03c0"),
        (5, 4, "5\u03c0/4"),
        (4, 3, "4\u03c0/3"),
        (3, 2, "3\u03c0/2"),
        (5, 3, "5\u03c0/3"),
        (7, 4, "7\u03c0/4"),
        (2, 1, "2\u03c0"),
    ]:
        val = numer * math.pi / denom if denom != 0 else 0.0
        if abs(rad - val) < 1e-6:
            return label
        if abs(rad + val) < 1e-6 and val != 0:
            return f"-{label}"

    # Fallback: show degrees
    deg = math.degrees(rad)
    if abs(deg - round(deg)) < 0.1:
        return f"{int(round(deg))}\u00b0"
    return f"{deg:.1f}\u00b0"


def _generate_ticks(vmin: float, vmax: float, max_ticks: int = 6) -> list[float]:
    """Generate nice tick positions in the range [vmin, vmax].

    Prefers multiples of pi/4, falling back to pi/2 or pi as the
    viewport gets wider.
    """
    span = vmax - vmin
    if span <= 0:
        return []

    # Choose step: prefer pi/4, then pi/2, then pi
    candidates = [math.pi / 4, math.pi / 3, math.pi / 2, math.pi]
    step = candidates[0]
    for s in candidates:
        count = int(span / s) + 1
        if count <= max_ticks:
            step = s
            break

    # Find the first tick at or after vmin
    first = math.ceil(vmin / step) * step
    ticks = []
    t = first
    while t <= vmax + 1e-9:
        ticks.append(t)
        t = t + step

    return ticks


class FractalCanvas(QWidget):
    """Widget that displays the fractal image and handles zoom interactions."""

    viewport_changed = pyqtSignal(FractalViewport)
    ic_selected = pyqtSignal(float, float)  # theta1, theta2 (Ctrl+click)
    hover_updated = pyqtSignal(float, float)  # theta1, theta2 (inspect mode)
    trajectory_pinned = pyqtSignal(str, float, float)  # row_id, theta1, theta2
    pan_started = pyqtSignal()  # emitted at start of pan drag

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Viewport state
        self._center_theta1 = math.pi
        self._center_theta2 = math.pi
        self._span_theta1 = 2 * math.pi
        self._span_theta2 = 2 * math.pi
        self._resolution = 256

        # Display state
        self._current_image = None

        # Basin (winding number) mode — always on
        self._basin_mode = True
        self._winding_colormap_name = "Basin Hash"
        self._winding_colormap_fn = winding_basin_hash
        self._cached_winding_legend_name: str | None = None
        self._cached_winding_legend_image = None
        self._basin_theta1_final: np.ndarray | None = None
        self._basin_theta2_final: np.ndarray | None = None
        self._basin_theta1_init: np.ndarray | None = None
        self._basin_theta2_init: np.ndarray | None = None
        self._basin_convergence_times: np.ndarray | None = None

        # Tool mode
        self._tool_mode = TOOL_ZOOM
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Rectangle selection state (zoom mode)
        self._selecting = False
        self._select_anchor_x = 0.0
        self._select_anchor_y = 0.0
        self._select_rect: QRectF | None = None  # pixel-space rect

        # Pan state (pan mode)
        self._panning = False
        self._pan_anchor_px = 0.0
        self._pan_anchor_py = 0.0
        self._pan_anchor_theta1 = 0.0
        self._pan_anchor_theta2 = 0.0

        # Ghost rectangle state (for zoom-out feedback)
        self._ghost_rect: QRectF | None = None   # pixel-space rect
        self._ghost_alpha = 0
        self._ghost_timer = QTimer()
        self._ghost_timer.setInterval(GHOST_FADE_TICK_MS)
        self._ghost_timer.timeout.connect(self._on_ghost_tick)

        # Pending ghost: physics-space viewport stored at zoom-out time,
        # converted to pixel rect and fade-started when display() delivers
        # the first new image.
        self._pending_ghost_viewport: FractalViewport | None = None

        # Hover coordinate tracking
        self._hover_theta1 = None
        self._hover_theta2 = None

        # Pinned trajectory markers: row_id -> (theta1, theta2, color_rgb)
        self._pinned_markers: dict[str, tuple[float, float, tuple[int, int, int]]] = {}
        self._highlighted_marker_id: str | None = None

        # Stale overlay: previous high-res image shown during transitions
        self._stale_image: QImage | None = None
        self._stale_viewport: FractalViewport | None = None

        # Pan background: low-res cube preview shown behind shifting foreground
        self._pan_background: QImage | None = None
        self._pan_bg_viewport: FractalViewport | None = None

    # -- Public interface --

    def set_basin_mode(self, basin: bool) -> None:
        """Set basin (winding number) display mode. Always True now."""
        self._basin_mode = basin

    @property
    def winding_colormap_name(self) -> str:
        """Currently active winding colormap name."""
        return self._winding_colormap_name

    def set_winding_colormap(self, name: str) -> None:
        """Change the active winding colormap and refresh."""
        if name not in WINDING_COLORMAPS:
            return
        self._winding_colormap_name = name
        self._winding_colormap_fn = WINDING_COLORMAPS[name]
        self._cached_winding_legend_name = None  # invalidate legend cache
        if self._basin_mode and self._basin_theta1_final is not None:
            self._rebuild_image()

    def display_basin_final(
        self,
        theta1_final: np.ndarray,
        theta2_final: np.ndarray,
        convergence_times: np.ndarray | None = None,
        theta1_init: np.ndarray | None = None,
        theta2_init: np.ndarray | None = None,
    ) -> None:
        """Update display in basin mode from final angles and convergence times.

        Args:
            theta1_final: (N,) float32 array of final unwrapped theta1 values.
            theta2_final: (N,) float32 array of final unwrapped theta2 values.
            convergence_times: (N,) float32 array of convergence times.
            theta1_init: (N,) float32 array of initial theta1 values (for relative winding).
            theta2_init: (N,) float32 array of initial theta2 values (for relative winding).
        """
        self._basin_mode = True
        self._basin_theta1_final = theta1_final
        self._basin_theta2_final = theta2_final
        self._basin_theta1_init = theta1_init
        self._basin_theta2_init = theta2_init
        self._basin_convergence_times = convergence_times
        self._rebuild_image()

    def display_basin_from_cube(self, cube_slice) -> None:
        """Display a precomputed cube slice as an instant preview.

        Takes a CubeSlice (n1, n2, brightness) and renders it using the
        current winding colormap. Skips all integration — just colormap
        lookup + brightness multiply.

        Args:
            cube_slice: CubeSlice with n1 (R,R) int8, n2 (R,R) int8,
                brightness (R,R) uint8.
        """
        self._current_image = render_cube_to_qimage(
            cube_slice, self._winding_colormap_fn,
        )
        self.update()

    def set_resolution(self, resolution: int) -> None:
        """Update the target resolution."""
        self._resolution = resolution

    def get_viewport(self) -> FractalViewport:
        """Return the current viewport as a frozen dataclass."""
        return FractalViewport(
            center_theta1=self._center_theta1,
            center_theta2=self._center_theta2,
            span_theta1=self._span_theta1,
            span_theta2=self._span_theta2,
            resolution=self._resolution,
        )

    def set_viewport(
        self,
        center_theta1: float,
        center_theta2: float,
        span_theta1: float | None = None,
        span_theta2: float | None = None,
    ) -> None:
        """Set the viewport center (and optionally span)."""
        self._center_theta1 = center_theta1
        self._center_theta2 = center_theta2
        if span_theta1 is not None:
            self._span_theta1 = span_theta1
        if span_theta2 is not None:
            self._span_theta2 = span_theta2

    def set_tool_mode(self, mode: str) -> None:
        """Switch between zoom, pan, and inspect tool modes."""
        if mode not in (TOOL_ZOOM, TOOL_PAN, TOOL_INSPECT):
            return
        self._tool_mode = mode
        # Cancel any in-progress interaction
        self._selecting = False
        self._select_rect = None
        self._panning = False
        # Set default cursor for the mode
        if mode == TOOL_ZOOM:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == TOOL_PAN:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(_build_inspect_cursor(CURSOR_DEFAULT_COLOR))
        self.update()

    def set_inspect_cursor_color(
        self, r: int, g: int, b: int,
    ) -> None:
        """Update the inspect cursor fill to match the hovered winding color.

        Only takes effect when in inspect mode. Builds a new cursor pixmap.

        Args:
            r: Red component (0-255).
            g: Green component (0-255).
            b: Blue component (0-255).
        """
        if self._tool_mode != TOOL_INSPECT:
            return
        self.setCursor(_build_inspect_cursor(QColor(r, g, b)))

    def activate_pending_ghost(self) -> None:
        """Start fading the ghost rectangle (call after final render).

        The ghost is already visible at full opacity from zoom_out().
        This just kicks off the fade-out timer.
        """
        if self._pending_ghost_viewport is not None:
            self._pending_ghost_viewport = None
            # Reset alpha to full and start fading
            self._ghost_alpha = GHOST_INITIAL_ALPHA
            self._ghost_timer.start()

    def zoom_out(self) -> None:
        """Zoom out by a fixed factor, showing a ghost of the previous region.

        Called externally (e.g. by a button in FractalControls via FractalView).
        Saves the current image as a stale overlay before changing spans so
        the high-res detail remains visible at its correct sub-region.
        """
        old_viewport = self.get_viewport()

        # Snapshot high-res content before viewport changes
        self.save_stale()

        # Compute new (larger) span
        new_span1 = min(MAX_SPAN, self._span_theta1 * ZOOM_OUT_FACTOR)
        new_span2 = min(MAX_SPAN, self._span_theta2 * ZOOM_OUT_FACTOR)

        # Update viewport (center stays the same)
        self._span_theta1 = new_span1
        self._span_theta2 = new_span2

        # Show ghost rectangle of where the old viewport sits in the new one
        self._show_ghost_rect(old_viewport)

        self.viewport_changed.emit(self.get_viewport())

    @property
    def hover_theta1(self) -> float | None:
        return self._hover_theta1

    @property
    def hover_theta2(self) -> float | None:
        return self._hover_theta2

    # -- Pinned marker management --

    def add_marker(
        self, row_id: str, theta1: float, theta2: float,
        color_rgb: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Add or update a pinned trajectory marker on the canvas.

        Args:
            row_id: Unique identifier for this marker.
            theta1: Physics-space theta1 coordinate.
            theta2: Physics-space theta2 coordinate.
            color_rgb: (R, G, B) fill color for the marker.
        """
        self._pinned_markers = {
            **self._pinned_markers,
            row_id: (theta1, theta2, color_rgb),
        }
        self.update()

    def update_marker_color(
        self, row_id: str, color_rgb: tuple[int, int, int],
    ) -> None:
        """Update the color of an existing marker.

        Args:
            row_id: Marker to update.
            color_rgb: New (R, G, B) fill color.
        """
        entry = self._pinned_markers.get(row_id)
        if entry is None:
            return
        theta1, theta2, _old_color = entry
        self._pinned_markers = {
            **self._pinned_markers,
            row_id: (theta1, theta2, color_rgb),
        }
        self.update()

    def highlight_marker(self, row_id: str) -> None:
        """Highlight a marker (e.g. on indicator hover)."""
        if self._highlighted_marker_id != row_id:
            self._highlighted_marker_id = row_id
            self.update()

    def unhighlight_marker(self) -> None:
        """Remove marker highlight."""
        if self._highlighted_marker_id is not None:
            self._highlighted_marker_id = None
            self.update()

    def remove_marker(self, row_id: str) -> None:
        """Remove a pinned trajectory marker by row_id."""
        new_markers = {k: v for k, v in self._pinned_markers.items() if k != row_id}
        self._pinned_markers = new_markers
        if self._highlighted_marker_id == row_id:
            self._highlighted_marker_id = None
        self.update()

    def clear_markers(self) -> None:
        """Remove all pinned trajectory markers."""
        self._pinned_markers = {}
        self._highlighted_marker_id = None
        self.update()

    # -- Stale overlay / pan background --

    def save_stale(self, viewport: FractalViewport | None = None) -> None:
        """Snapshot _current_image as stale overlay.

        The stale overlay is drawn on top of lower-res content so the user
        sees crisp detail from the previous viewport while the progressive
        pipeline sharpens the rest.

        Args:
            viewport: The viewport the image was rendered for.
                Defaults to current viewport (correct for zoom-out
                since it's called before spans change).
        """
        if self._current_image is None:
            return
        self._stale_image = self._current_image.copy()
        self._stale_viewport = viewport if viewport is not None else self.get_viewport()

    def clear_stale(self) -> None:
        """Remove stale overlay (called when progressive pipeline finishes)."""
        self._stale_image = None
        self._stale_viewport = None
        self.update()

    def set_pan_background(self, image: QImage, viewport: FractalViewport) -> None:
        """Set low-res background for pan drag.

        The pan background fills exposed edges during drag so the user
        sees blurry cube data instead of black.

        Args:
            image: QImage of the cube preview.
            viewport: The viewport the background was rendered for
                (typically the full [0, 2pi]^2 range).
        """
        self._pan_background = image
        self._pan_bg_viewport = viewport

    def clear_pan_background(self) -> None:
        """Remove pan background (called on pan release)."""
        self._pan_background = None
        self._pan_bg_viewport = None

    # -- Image rendering --

    def _rebuild_image(self) -> None:
        """Rebuild QImage from current basin data."""
        self._rebuild_image_winding()

    def _rebuild_image_winding(self) -> None:
        """Rebuild QImage using winding number colormap (basin mode)."""
        theta1_final = self._basin_theta1_final
        theta2_final = self._basin_theta2_final
        if theta1_final is None or theta2_final is None:
            return

        res = int(math.sqrt(theta1_final.shape[0]))

        argb = winding_to_argb(
            theta1_final.astype(np.float32),
            theta2_final.astype(np.float32),
            self._winding_colormap_fn,
            res,
            convergence_times=self._basin_convergence_times,
            theta1_init=self._basin_theta1_init,
            theta2_init=self._basin_theta2_init,
        )
        self._current_image = numpy_to_qimage(argb)
        self.update()

    # -- Coordinate mapping --

    def _image_rect(self) -> tuple[float, float, float]:
        """Return (img_x, img_y, side) for the square image area.

        Reserves margins on the left (for θ₂ labels) and bottom
        (for θ₁ labels) so the image doesn't overlap the axes.
        """
        w = self.width() - AXIS_MARGIN_LEFT
        h = self.height() - AXIS_MARGIN_BOTTOM
        side = min(w, h)
        # Image sits in the top-right of the available area
        img_x = AXIS_MARGIN_LEFT + (w - side) / 2
        img_y = (h - side) / 2
        return img_x, img_y, side

    def _pixel_to_physics(self, px: float, py: float) -> tuple[float, float]:
        """Convert pixel coordinates to (theta1, theta2) physics space."""
        img_x, img_y, side = self._image_rect()

        # Normalized [0, 1] within the image
        nx = (px - img_x) / side
        ny = (py - img_y) / side

        half_span1 = self._span_theta1 / 2
        half_span2 = self._span_theta2 / 2

        theta1 = self._center_theta1 - half_span1 + nx * self._span_theta1
        theta2 = self._center_theta2 - half_span2 + ny * self._span_theta2

        return theta1, theta2

    def _physics_to_pixel(
        self, theta1: float, theta2: float
    ) -> tuple[float, float]:
        """Convert (theta1, theta2) physics space to pixel coordinates."""
        img_x, img_y, side = self._image_rect()

        half_span1 = self._span_theta1 / 2
        half_span2 = self._span_theta2 / 2

        nx = (theta1 - (self._center_theta1 - half_span1)) / self._span_theta1
        ny = (theta2 - (self._center_theta2 - half_span2)) / self._span_theta2

        px = img_x + nx * side
        py = img_y + ny * side
        return px, py

    # -- Selection rectangle helpers --

    def _build_selection_rect(
        self, anchor_x: float, anchor_y: float,
        current_x: float, current_y: float,
    ) -> QRectF:
        """Build a fixed-aspect-ratio selection rectangle.

        The rectangle always has a 1:1 aspect ratio (matching the square
        canvas), anchored at the click point and sized by the larger of
        the mouse's x or y displacement.
        """
        img_x, img_y, side = self._image_rect()

        dx = current_x - anchor_x
        dy = current_y - anchor_y

        # Use the larger displacement to set the square size
        extent = max(abs(dx), abs(dy))
        if extent < 2:
            # Too small to draw
            return QRectF(anchor_x, anchor_y, 0, 0)

        # Preserve the drag direction for each axis
        sx = extent if dx >= 0 else -extent
        sy = extent if dy >= 0 else -extent

        x0 = anchor_x
        y0 = anchor_y
        x1 = anchor_x + sx
        y1 = anchor_y + sy

        # Normalise to (left, top, width, height)
        left = min(x0, x1)
        top = min(y0, y1)
        rect_side = abs(sx)

        # Clamp to the image area
        left = max(img_x, left)
        top = max(img_y, top)
        right = min(img_x + side, left + rect_side)
        bottom = min(img_y + side, top + rect_side)

        # Re-enforce squareness after clamping
        clamped_side = min(right - left, bottom - top)
        if clamped_side < 4:
            return QRectF(anchor_x, anchor_y, 0, 0)

        return QRectF(left, top, clamped_side, clamped_side)

    # -- Ghost rectangle --

    def _show_ghost_rect(self, old_viewport: FractalViewport) -> None:
        """Show a static ghost rectangle immediately and mark it pending fade.

        The rectangle appears at full opacity right away (during computation).
        The fade timer is NOT started here — call activate_pending_ghost()
        after the final render to begin the fade-out.
        """
        self._pending_ghost_viewport = old_viewport

        # Convert to pixel space and show at full alpha immediately
        old_left = old_viewport.center_theta1 - old_viewport.span_theta1 / 2
        old_top = old_viewport.center_theta2 - old_viewport.span_theta2 / 2
        old_right = old_viewport.center_theta1 + old_viewport.span_theta1 / 2
        old_bottom = old_viewport.center_theta2 + old_viewport.span_theta2 / 2

        px_left, py_top = self._physics_to_pixel(old_left, old_top)
        px_right, py_bottom = self._physics_to_pixel(old_right, old_bottom)

        self._ghost_rect = QRectF(
            px_left, py_top,
            px_right - px_left, py_bottom - py_top,
        )
        self._ghost_alpha = GHOST_INITIAL_ALPHA
        # No timer start — ghost stays static until activate_pending_ghost()
        self.update()

    def _on_ghost_tick(self) -> None:
        """Fade the ghost rectangle by one tick."""
        fade_per_tick = GHOST_INITIAL_ALPHA / (GHOST_FADE_MS / GHOST_FADE_TICK_MS)
        self._ghost_alpha = max(0, self._ghost_alpha - fade_per_tick)

        if self._ghost_alpha <= 0:
            self._ghost_rect = None
            self._ghost_timer.stop()

        self.update()

    # -- Axes drawing --

    def _draw_axes(self, painter: QPainter) -> None:
        """Draw axis labels on borders and reference lines at θ=π."""
        img_x, img_y, side = self._image_rect()

        half_span1 = self._span_theta1 / 2
        half_span2 = self._span_theta2 / 2
        vmin1 = self._center_theta1 - half_span1
        vmax1 = self._center_theta1 + half_span1
        vmin2 = self._center_theta2 - half_span2
        vmax2 = self._center_theta2 + half_span2

        font = QFont("Helvetica", 18)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # -- θ₁ ticks (bottom edge, horizontal axis) --
        painter.setPen(AXIS_LABEL_COLOR)
        for tick in _generate_ticks(vmin1, vmax1):
            nx = (tick - vmin1) / self._span_theta1
            px = img_x + nx * side
            # Small tick mark
            painter.drawLine(
                int(px), int(img_y + side),
                int(px), int(img_y + side + TICK_LENGTH),
            )
            # Label below tick
            label = _format_angle(tick)
            tw = fm.horizontalAdvance(label)
            painter.drawText(
                int(px - tw / 2),
                int(img_y + side + TICK_LENGTH + fm.ascent() + 2),
                label,
            )

        # -- θ₂ ticks (left edge, vertical axis) --
        for tick in _generate_ticks(vmin2, vmax2):
            ny = (tick - vmin2) / self._span_theta2
            py = img_y + ny * side
            # Small tick mark
            painter.drawLine(
                int(img_x - TICK_LENGTH), int(py),
                int(img_x), int(py),
            )
            # Label left of tick
            label = _format_angle(tick)
            tw = fm.horizontalAdvance(label)
            painter.drawText(
                int(img_x - TICK_LENGTH - tw - 3),
                int(py + fm.ascent() / 2 - 1),
                label,
            )

        # -- Axis titles --
        painter.setPen(QColor(130, 130, 140))
        title_font = QFont("Helvetica", 20)
        painter.setFont(title_font)

        # θ₁ title (centered below bottom labels)
        t1_title = "\u03b8\u2081"
        t1_tw = QFontMetrics(title_font).horizontalAdvance(t1_title)
        painter.drawText(
            int(img_x + side / 2 - t1_tw / 2),
            int(self.height() - 2),
            t1_title,
        )

        # θ₂ title (centered left of left labels, rotated)
        painter.save()
        t2_title = "\u03b8\u2082"
        painter.translate(12, img_y + side / 2)
        painter.rotate(-90)
        t2_tw = QFontMetrics(title_font).horizontalAdvance(t2_title)
        painter.drawText(int(-t2_tw / 2), 0, t2_title)
        painter.restore()

        # -- Reference lines at θ = π (thin, through the image) --
        # Clip to image area
        painter.save()
        painter.setClipRect(
            int(img_x), int(img_y), int(side), int(side),
        )

        pi_pen = QPen(PI_LINE_COLOR)
        pi_pen.setWidth(2)
        pi_pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pi_pen)

        # Vertical line at θ₁ = π
        if vmin1 < math.pi < vmax1:
            nx = (math.pi - vmin1) / self._span_theta1
            px = img_x + nx * side
            painter.drawLine(
                int(px), int(img_y),
                int(px), int(img_y + side),
            )

        # Horizontal line at θ₂ = π
        if vmin2 < math.pi < vmax2:
            ny = (math.pi - vmin2) / self._span_theta2
            py = img_y + ny * side
            painter.drawLine(
                int(img_x), int(py),
                int(img_x + side), int(py),
            )

        painter.restore()

    # -- Legend --

    def _draw_winding_legend(self, painter: QPainter) -> None:
        """Draw a 2D grid legend in the bottom-right corner (basin mode)."""
        img_x, img_y, side = self._image_rect()

        # Build or reuse cached legend image
        if self._cached_winding_legend_name != self._winding_colormap_name:
            legend_data = build_winding_legend(self._winding_colormap_fn)
            self._cached_winding_legend_image = numpy_to_qimage(legend_data)
            self._cached_winding_legend_name = self._winding_colormap_name

        legend_w = self._cached_winding_legend_image.width()
        legend_h = self._cached_winding_legend_image.height()
        padding = 8

        # Position: bottom-right of image area
        lx = int(img_x + side - legend_w - padding)
        ly = int(img_y + side - legend_h - padding)

        painter.save()

        # Draw the legend grid
        painter.drawImage(lx, ly, self._cached_winding_legend_image)

        # Border
        painter.setPen(QColor(200, 200, 210))
        painter.drawRect(lx, ly, legend_w, legend_h)

        # Axis labels
        font = QFont("Helvetica", 10)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # n1 label (below, centered)
        painter.setPen(QColor(160, 160, 170))
        n1_label = "n\u2081"
        tw = fm.horizontalAdvance(n1_label)
        painter.drawText(
            int(lx + legend_w / 2 - tw / 2),
            ly + legend_h + fm.ascent() + 2,
            n1_label,
        )

        # n2 label (left, centered, rotated)
        n2_label = "n\u2082"
        painter.save()
        painter.translate(lx - fm.ascent() - 2, ly + legend_h / 2)
        painter.rotate(-90)
        tw2 = fm.horizontalAdvance(n2_label)
        painter.drawText(int(-tw2 / 2), 0, n2_label)
        painter.restore()

        # Corner range labels
        painter.setPen(QColor(130, 130, 140))
        small_font = QFont("Helvetica", 8)
        painter.setFont(small_font)
        sfm = QFontMetrics(small_font)

        # Bottom-left: "-3", bottom-right: "+3" (n1 axis)
        painter.drawText(lx, ly + legend_h + sfm.ascent(), "-3")
        tw_3 = sfm.horizontalAdvance("+3")
        painter.drawText(
            int(lx + legend_w - tw_3),
            ly + legend_h + sfm.ascent(),
            "+3",
        )

        # Top-left: "-3", bottom-left: "+3" (n2 axis)
        tw_m3 = sfm.horizontalAdvance("-3")
        painter.drawText(lx - tw_m3 - 2, ly + sfm.ascent(), "-3")
        painter.drawText(lx - tw_3 - 2, ly + legend_h, "+3")

        painter.restore()

    # -- Paint --

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), QColor(20, 20, 30))

        if self._current_image is None:
            # Draw placeholder text
            painter.setPen(QColor(100, 100, 120))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Fractal Explorer\nAdjust parameters to compute",
            )
            painter.end()
            return

        # Scale image to fill the widget (square, centered)
        img_x, img_y, side = self._image_rect()

        # During an active pan, offset the image so it tracks the cursor.
        # The image was rendered for the pre-pan viewport, so we shift it
        # by the pixel delta between the anchor and the current center.
        draw_x = img_x
        draw_y = img_y
        if self._panning:
            draw_x = img_x + (self._pan_anchor_theta1 - self._center_theta1) / self._span_theta1 * side
            draw_y = img_y + (self._pan_anchor_theta2 - self._center_theta2) / self._span_theta2 * side

        # Nearest-neighbor scaling (preserves pixel grid)
        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform, False,
        )

        # Clip all image layers to the image area
        painter.save()
        painter.setClipRect(int(img_x), int(img_y), int(side), int(side))

        # Layer 2: Pan background — cube preview behind shifting foreground
        if self._panning and self._pan_background is not None and self._pan_bg_viewport is not None:
            bv = self._pan_bg_viewport
            bg_tl_x, bg_tl_y = self._physics_to_pixel(
                bv.center_theta1 - bv.span_theta1 / 2,
                bv.center_theta2 - bv.span_theta2 / 2,
            )
            bg_br_x, bg_br_y = self._physics_to_pixel(
                bv.center_theta1 + bv.span_theta1 / 2,
                bv.center_theta2 + bv.span_theta2 / 2,
            )
            bg_rect = QRectF(bg_tl_x, bg_tl_y, bg_br_x - bg_tl_x, bg_br_y - bg_tl_y)
            painter.drawImage(bg_rect, self._pan_background)

        # Layer 3: Main image (shifted during pan)
        painter.drawImage(
            int(draw_x), int(draw_y),
            self._current_image.scaled(
                int(side), int(side),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            ),
        )

        # Layer 4: Stale overlay — old high-res at correct physics position
        if self._stale_image is not None and self._stale_viewport is not None:
            sv = self._stale_viewport
            stale_tl_x, stale_tl_y = self._physics_to_pixel(
                sv.center_theta1 - sv.span_theta1 / 2,
                sv.center_theta2 - sv.span_theta2 / 2,
            )
            stale_br_x, stale_br_y = self._physics_to_pixel(
                sv.center_theta1 + sv.span_theta1 / 2,
                sv.center_theta2 + sv.span_theta2 / 2,
            )
            stale_rect = QRectF(
                stale_tl_x, stale_tl_y,
                stale_br_x - stale_tl_x, stale_br_y - stale_tl_y,
            )
            painter.drawImage(stale_rect, self._stale_image)

        painter.restore()

        # Axes, labels, and reference lines
        self._draw_axes(painter)

        # Legend: winding grid
        self._draw_winding_legend(painter)

        # Draw selection rectangle (while dragging)
        if self._selecting and self._select_rect is not None:
            pen = QPen(QColor(255, 255, 255, 200))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(self._select_rect)

        # Draw ghost rectangle (after zoom-out)
        if self._ghost_rect is not None and self._ghost_alpha > 0:
            alpha = int(self._ghost_alpha)
            pen = QPen(QColor(220, 220, 220, alpha))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QColor(200, 200, 200, alpha // 3))
            painter.drawRect(self._ghost_rect)

        # Draw pinned trajectory markers (colored X with dark outline).
        # Two passes: non-highlighted first, then highlighted on top.
        if self._pinned_markers:
            highlighted_id = self._highlighted_marker_id
            for pass_highlighted in (False, True):
                for row_id, (theta1, theta2, color_rgb) in self._pinned_markers.items():
                    is_highlighted = (row_id == highlighted_id)
                    if is_highlighted != pass_highlighted:
                        continue

                    px, py = self._physics_to_pixel(theta1, theta2)
                    if not (img_x <= px <= img_x + side
                            and img_y <= py <= img_y + side):
                        continue

                    marker_size = 8 if is_highlighted else 5
                    ipx, ipy = int(px), int(py)

                    # Dark outline (behind the colored X)
                    outline_pen = QPen(QColor(0, 0, 0, 200))
                    outline_pen.setWidth(6 if is_highlighted else 3)
                    painter.setPen(outline_pen)
                    painter.drawLine(
                        ipx - marker_size, ipy - marker_size,
                        ipx + marker_size, ipy + marker_size,
                    )
                    painter.drawLine(
                        ipx + marker_size, ipy - marker_size,
                        ipx - marker_size, ipy + marker_size,
                    )

                    # Colored fill
                    fill_pen = QPen(QColor(*color_rgb, 240))
                    fill_pen.setWidth(2)
                    painter.setPen(fill_pen)
                    painter.drawLine(
                        ipx - marker_size, ipy - marker_size,
                        ipx + marker_size, ipy + marker_size,
                    )
                    painter.drawLine(
                        ipx + marker_size, ipy - marker_size,
                        ipx - marker_size, ipy + marker_size,
                    )

        painter.end()

    # -- Mouse events --

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl+click: select initial conditions (works in any mode)
                pos = event.position()
                theta1, theta2 = self._pixel_to_physics(pos.x(), pos.y())
                self.ic_selected.emit(theta1, theta2)
                return

            if self._tool_mode == TOOL_INSPECT:
                # Click in inspect mode: pin trajectory at this point
                pos = event.position()
                img_x, img_y, side = self._image_rect()
                if (img_x <= pos.x() <= img_x + side
                        and img_y <= pos.y() <= img_y + side):
                    theta1, theta2 = self._pixel_to_physics(pos.x(), pos.y())
                    row_id = str(uuid.uuid4())[:8]
                    self.add_marker(row_id, theta1, theta2)
                    self.trajectory_pinned.emit(row_id, theta1, theta2)
                return

            if self._tool_mode == TOOL_ZOOM:
                # Start rectangle selection for zoom
                self._selecting = True
                self._select_anchor_x = event.position().x()
                self._select_anchor_y = event.position().y()
                self._select_rect = None
            elif self._tool_mode == TOOL_PAN:
                # Start panning — clean slate for pan interaction
                self.clear_stale()
                self._panning = True
                pos = event.position()
                self._pan_anchor_px = pos.x()
                self._pan_anchor_py = pos.y()
                self._pan_anchor_theta1 = self._center_theta1
                self._pan_anchor_theta2 = self._center_theta2
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.pan_started.emit()

    def mouseMoveEvent(self, event):
        pos = event.position()

        # Update hover coordinates
        theta1, theta2 = self._pixel_to_physics(pos.x(), pos.y())
        self._hover_theta1 = theta1
        self._hover_theta2 = theta2

        # Emit hover signal in inspect mode
        if self._tool_mode == TOOL_INSPECT:
            # Only emit when cursor is within the image area
            img_x, img_y, side = self._image_rect()
            if (img_x <= pos.x() <= img_x + side
                    and img_y <= pos.y() <= img_y + side):
                self.hover_updated.emit(theta1, theta2)

        if self._selecting:
            self._select_rect = self._build_selection_rect(
                self._select_anchor_x, self._select_anchor_y,
                pos.x(), pos.y(),
            )
            self.update()
        elif self._panning:
            # Compute how far the mouse moved in pixel space,
            # convert to physics delta, and shift the viewport center
            img_x, img_y, side = self._image_rect()
            dx_px = pos.x() - self._pan_anchor_px
            dy_px = pos.y() - self._pan_anchor_py

            # Convert pixel delta to physics delta
            d_theta1 = -dx_px / side * self._span_theta1
            d_theta2 = -dy_px / side * self._span_theta2

            self._center_theta1 = self._pan_anchor_theta1 + d_theta1
            self._center_theta2 = self._pan_anchor_theta2 + d_theta2
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._selecting:
            self._selecting = False
            self.setCursor(Qt.CursorShape.CrossCursor)

            rect = self._select_rect
            self._select_rect = None

            if rect is None or rect.width() < 8 or rect.height() < 8:
                # Selection too small — treat as a click, ignore
                self.update()
                return

            # Convert selection rectangle to physics coordinates
            top_left_theta1, top_left_theta2 = self._pixel_to_physics(
                rect.left(), rect.top(),
            )
            bot_right_theta1, bot_right_theta2 = self._pixel_to_physics(
                rect.right(), rect.bottom(),
            )

            new_center_theta1 = (top_left_theta1 + bot_right_theta1) / 2
            new_center_theta2 = (top_left_theta2 + bot_right_theta2) / 2
            new_span_theta1 = abs(bot_right_theta1 - top_left_theta1)
            new_span_theta2 = abs(bot_right_theta2 - top_left_theta2)

            # Enforce limits
            new_span_theta1 = max(MIN_SPAN, min(MAX_SPAN, new_span_theta1))
            new_span_theta2 = max(MIN_SPAN, min(MAX_SPAN, new_span_theta2))

            # Update viewport
            self._center_theta1 = new_center_theta1
            self._center_theta2 = new_center_theta2
            self._span_theta1 = new_span_theta1
            self._span_theta2 = new_span_theta2

            self.viewport_changed.emit(self.get_viewport())
            self.update()

        elif event.button() == Qt.MouseButton.LeftButton and self._panning:
            # Build the viewport the current image was rendered for
            # (anchor center + current spans) so stale overlay maps correctly
            anchor_viewport = FractalViewport(
                center_theta1=self._pan_anchor_theta1,
                center_theta2=self._pan_anchor_theta2,
                span_theta1=self._span_theta1,
                span_theta2=self._span_theta2,
                resolution=self._resolution,
            )
            self.save_stale(anchor_viewport)
            self.clear_pan_background()
            self._panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.viewport_changed.emit(self.get_viewport())
            self.update()

    def wheelEvent(self, event):
        # Scroll wheel is intentionally disabled for zooming.
        # Zoom-in uses rectangle selection, zoom-out uses the button.
        event.ignore()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts: Home (reset view), Escape (cancel)."""
        key = event.key()

        if key == Qt.Key.Key_Home:
            self._center_theta1 = math.pi
            self._center_theta2 = math.pi
            self._span_theta1 = 2 * math.pi
            self._span_theta2 = 2 * math.pi
            self.viewport_changed.emit(self.get_viewport())
        elif key == Qt.Key.Key_Escape:
            if self._selecting:
                # Cancel rectangle selection
                self._selecting = False
                self._select_rect = None
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif self._panning:
                # Cancel pan: restore original center
                self._center_theta1 = self._pan_anchor_theta1
                self._center_theta2 = self._pan_anchor_theta2
                self._panning = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.update()
        else:
            super().keyPressEvent(event)

        self.update()
