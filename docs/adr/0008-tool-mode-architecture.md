# ADR-0008: Three-Mode Tool Architecture (Zoom/Pan/Inspect)

**Status**: Accepted
**Date**: 2025-06

## Context

The fractal canvas originally had only zoom (rectangle selection). Pan was
added for interactive exploration. The inspect tool required a third mode where
hover events trigger data lookups rather than canvas manipulation.

## Decision

Three exclusive tool modes managed by `QButtonGroup` in the controls panel:
- **Zoom**: drag rectangle to zoom in
- **Pan**: drag to pan with live image offset
- **Inspect**: hover emits coordinates for pendulum diagram updates

All modes support Ctrl+click for IC selection (cross-mode invariant).

## Alternatives Considered

- **Modifier keys instead of modes**: e.g. Shift+drag to pan, Alt+hover to
  inspect. Eliminates mode buttons but makes discovery harder and conflicts
  with platform shortcuts.
- **Inspect as overlay (always active)**: show inspect info in a status bar
  on hover regardless of mode. Simpler but means the inspect panel is always
  visible and processing hover events during zoom/pan drags.
- **Right-click context menu**: inspect via right-click on a pixel. Precise
  but not continuous — can't sweep across the fractal to see the pendulum
  change in real time.

## Consequences

- `QButtonGroup` with exclusive toggle ensures exactly one mode is active.
- Each mode sets its own cursor shape for visual feedback.
- The canvas `mouseMoveEvent` and `mousePressEvent` dispatch based on
  `_tool_mode` string.
- The inspect panel in controls is shown/hidden based on mode, reducing
  visual clutter when not inspecting.
- Adding a fourth tool mode requires: one button, one cursor, one dispatch
  branch, and one signal if needed.
