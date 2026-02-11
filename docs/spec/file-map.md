# File Map

Source file inventory with line counts, ownership, and dependency graph.

> **Canonical**: This is the single source of truth for module layout.
> Update this file when adding, removing, or significantly resizing files.

## Source Files

```
chaos-pendulum/
    main.py                          52 lines   Entry point, creates AppWindow
    simulation.py                   119 lines   Physics engine (with friction damping)
    app_window.py                   179 lines   QStackedWidget, mode switching, param sync
    ui_common.py                    255 lines   LoadingOverlay, PhysicsParamsWidget, sliders

    pendulum/
        __init__.py
        canvas.py                              PendulumCanvas (extracted from visualization.py)
        controls.py                            Pendulum ControlPanel (extracted)
        view.py                                PendulumView + PendulumSimWorker

    fractal/
        __init__.py                   1 line
        canvas.py                  1138 lines   FractalCanvas: QImage, pan/zoom, axes, legend, tools
        controls.py                 512 lines   Time slider, colormap, resolution, basin mode, inspect
        view.py                     528 lines   FractalView: orchestration, signal wiring
        inspect_column.py           609 lines   InspectColumn: hover + stacked animation + scrub
        animated_diagram.py         373 lines   MultiTrajectoryDiagram: ghost IC + basin-colored bobs
        trajectory_indicator.py     220 lines   TrajectoryIndicator: clickable circle with winding numbers
        pendulum_diagram.py         133 lines   PendulumDiagram: stick-figure widget
        winding_circle.py           139 lines   WindingCircle: colored circle for basin hover
        worker.py                   118 lines   FractalWorker QThread (dispatches to RK4 or DOP853)
        compute.py                  183 lines   ComputeBackend Protocol, BatchResult, BasinResult, saddle_energy
        basin_solver.py             126 lines   DOP853 adaptive solver for basin mode (final state only)
        _numpy_backend.py           253 lines   Vectorized NumPy RK4 for angle mode (+ energy freeze logic)
        _numba_backend.py           225 lines   @njit parallel RK4 for angle mode (+ energy freeze logic)
        cache.py                    176 lines   FractalCache, CacheKey, LRU
        coloring.py                 170 lines   HSV LUT, angle_to_argb, QImage builder (univariate)
        bivariate.py                577 lines   Torus colormaps, bivariate_to_argb, legend builder
        winding.py                  273 lines   Winding number extraction + basin colormaps

    tests/
        test_simulation.py
        test_numpy_backend.py                  Batch RK4 + energy + freeze tests
        test_coloring.py                       Univariate coloring + angle slicing tests
        test_bivariate.py                      Torus colormaps: landmarks, periodicity, performance
        test_cache.py
        test_compute.py                        BatchResult, BasinResult, saddle_energy, FractalTask.basin
        test_basin_solver.py                   DOP853 basin solver: shape, winding, energy, cross-validation
        test_damping.py                        Friction: derivatives, energy decay, convergence
        test_winding.py                        Winding number extraction + colormaps
        test_energy_termination.py             Freeze behavior, speedup, winding stability
        test_inspect_utils.py                  rk4_single_trajectory + get_single_winding_color
        test_multi_trajectory.py               TrajectoryInfo, PinnedTrajectory, color lookup, constants
```

**Total**: ~5,800 lines across ~30 modules.

## Notes

- `fractal/canvas.py` (1006 lines) exceeds the 400-line guideline. Accumulated
  features: axes, donut legend, torus legend, ghost rect, 3 tool modes,
  coordinate mapping, bivariate display path. Consider extracting overlay
  drawing into `fractal/overlays.py` if it grows further.
- `fractal/bivariate.py` (577 lines) exceeds the 400-line guideline due to 9
  colormap functions plus helper utilities. Each function is self-contained;
  splitting would fragment related math.
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
                    |                         --> fractal/bivariate.py
                    |                         --> fractal/winding.py
                    --> fractal/controls.py   --> ui_common.py
                    |     --> fractal/coloring.py  (COLORMAPS registry)
                    |     --> fractal/bivariate.py (TORUS_COLORMAPS registry)
                    |     --> fractal/winding.py   (WINDING_COLORMAPS registry)
                    |     --> fractal/pendulum_diagram.py --> simulation.py
                    --> fractal/inspect_column.py
                    |     --> fractal/animated_diagram.py --> simulation.py
                    |     --> fractal/trajectory_indicator.py
                    |     --> fractal/pendulum_diagram.py
                    |     --> fractal/winding_circle.py
                    |     --> fractal/winding.py
                    --> fractal/worker.py     --> fractal/compute.py
                    |                         --> fractal/basin_solver.py --> simulation.py
                    |                              --> fractal/_numpy_backend.py
                    |                              --> fractal/_numba_backend.py
                    --> fractal/cache.py
                    --> simulation.py  (for DoublePendulumParams only)
```
