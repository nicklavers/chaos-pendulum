# File Map

Source file inventory with line counts, ownership, and dependency graph.

> **Canonical**: This is the single source of truth for module layout.
> Update this file when adding, removing, or significantly resizing files.

## Source Files

```
chaos-pendulum/
    main.py                          52 lines   Entry point, creates AppWindow
    simulation.py                   114 lines   Physics engine (unchanged from pendulum mode)
    app_window.py                   179 lines   QStackedWidget, mode switching, param sync
    ui_common.py                    249 lines   LoadingOverlay, PhysicsParamsWidget, sliders

    pendulum/
        __init__.py
        canvas.py                              PendulumCanvas (extracted from visualization.py)
        controls.py                            Pendulum ControlPanel (extracted)
        view.py                                PendulumView + PendulumSimWorker

    fractal/
        __init__.py                   1 line
        canvas.py                   887 lines   FractalCanvas: QImage, pan/zoom, axes, legend, tools
        controls.py                 368 lines   Time slider, colormap, resolution, inspect panel
        view.py                     348 lines   FractalView: orchestration, signal wiring
        pendulum_diagram.py         133 lines   PendulumDiagram: stick-figure widget
        worker.py                    96 lines   FractalWorker QThread
        compute.py                  137 lines   ComputeBackend Protocol + grid builder
        _numpy_backend.py           169 lines   Vectorized NumPy RK4
        _numba_backend.py           170 lines   @njit parallel RK4 (optional dep)
        cache.py                    176 lines   FractalCache, CacheKey, LRU
        coloring.py                 170 lines   HSV LUT, theta2_to_argb, QImage builder

    tests/
        test_simulation.py
        test_numpy_backend.py
        test_coloring.py
        test_cache.py
        test_fractal_view.py                   pytest-qt integration tests
```

**Total**: ~3,250 lines across ~20 modules.

## Notes

- `fractal/canvas.py` (887 lines) exceeds the 400-line guideline. Accumulated
  features: axes, legend, ghost rect, 3 tool modes, coordinate mapping. Consider
  extracting `_draw_axes` and `_draw_legend` into `fractal/overlays.py` if it
  grows further.
- Physics equations are duplicated between `simulation.py` (scalar) and
  `fractal/_numpy_backend.py` (vectorized batch). Cross-validation test ensures
  they stay in sync. See [ADR-0001](../adr/0001-vectorized-rk4.md).

## Dependency Graph

No circular dependencies.

```
main.py --> app_window.py
              --> pendulum/view.py
              |     --> pendulum/canvas.py
              |     --> pendulum/controls.py --> ui_common.py
              |     --> simulation.py
              |
              --> fractal/view.py
                    --> fractal/canvas.py     --> fractal/coloring.py
                    --> fractal/controls.py   --> ui_common.py
                    |     --> fractal/pendulum_diagram.py --> simulation.py
                    --> fractal/worker.py     --> fractal/compute.py
                    |                              --> fractal/_numpy_backend.py
                    |                              --> fractal/_numba_backend.py
                    --> fractal/cache.py
                    --> simulation.py  (for DoublePendulumParams only)
```
