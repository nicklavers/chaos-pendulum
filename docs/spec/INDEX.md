# Chaos Pendulum — Specification Index

Living specification for the double-pendulum chaos explorer. Each file describes
the **current state** of a subsystem — no history, no rationale (see
[ADRs](../adr/INDEX.md) for the "why").

## Spec Files

| File | Describes | Key types / concepts |
|------|-----------|---------------------|
| [file-map.md](file-map.md) | Source file inventory, line counts, dependency graph | Module layout |
| [data-shapes.md](data-shapes.md) | Frozen dataclasses, array shapes, signal payloads | `FractalViewport`, `CacheKey`, `FractalTask`, snapshot shape |
| [simulation.md](simulation.md) | Physics engine (`simulation.py`) | `DoublePendulumParams`, `derivatives()`, `positions()` |
| [fractal-compute.md](fractal-compute.md) | Compute backends and progressive rendering | `ComputeBackend` Protocol, NumPy / Numba / JAX backends |
| [fractal-caching.md](fractal-caching.md) | Cache architecture and eviction | `FractalCache`, `CacheKey`, LRU budget |
| [coloring-pipeline.md](coloring-pipeline.md) | Color mapping from angles to pixels | Univariate (1D LUT), bivariate (torus colormaps), winding (basin mode) |
| [canvas-rendering.md](canvas-rendering.md) | Canvas drawing: axes, legend, ghost rect, tool modes | `FractalCanvas`, coordinate mapping, overlays |
| [controls-ui.md](controls-ui.md) | Controls panel layout and signals | `FractalControls`, time slider, physics params |
| [inspect-tool.md](inspect-tool.md) | Inspect tool data flow and pendulum diagrams | `PendulumDiagram`, hover → lookup → display |
| [workers.md](workers.md) | Thread architecture, signals, cancellation | `FractalWorker`, retiring workers pattern |

## Reading Guide for Agents

Match your task to the minimum set of docs:

| Task | Read first | Then |
|------|-----------|------|
| **Fix a bug in one file** | [file-map.md](file-map.md) → find the file | The spec for that subsystem |
| **Add a new compute backend** | [data-shapes.md](data-shapes.md), [fractal-compute.md](fractal-compute.md) | [workers.md](workers.md) |
| **Change the color mapping** | [data-shapes.md](data-shapes.md), [coloring-pipeline.md](coloring-pipeline.md) | [canvas-rendering.md](canvas-rendering.md) |
| **Add a torus colormap** | [coloring-pipeline.md](coloring-pipeline.md) (bivariate section) | [controls-ui.md](controls-ui.md), [canvas-rendering.md](canvas-rendering.md) |
| **Add a new UI control** | [controls-ui.md](controls-ui.md) | [canvas-rendering.md](canvas-rendering.md) if it affects the canvas |
| **Add a new tool mode** | [canvas-rendering.md](canvas-rendering.md), [inspect-tool.md](inspect-tool.md) (for pattern) | [controls-ui.md](controls-ui.md) |
| **Change cache behavior** | [fractal-caching.md](fractal-caching.md), [data-shapes.md](data-shapes.md) | — |
| **Modify the physics** | [simulation.md](simulation.md), [fractal-compute.md](fractal-compute.md) | — |
| **Refactor module boundaries** | [file-map.md](file-map.md), all relevant specs | [ADRs](../adr/INDEX.md) for context |
| **Understand the full architecture** | All specs (start here, then [data-shapes.md](data-shapes.md)) | [ADRs](../adr/INDEX.md) |

## Canonical Source for Shared Concepts

- **Data shapes and frozen dataclasses** → [data-shapes.md](data-shapes.md) (single source of truth)
- **Physics equations** → [simulation.md](simulation.md)
- **File ownership and sizes** → [file-map.md](file-map.md)

## Cross-References

- Decision rationale: [../adr/INDEX.md](../adr/INDEX.md)
- Doc maintenance workflow: [../CONTRIBUTING.md](../CONTRIBUTING.md)
