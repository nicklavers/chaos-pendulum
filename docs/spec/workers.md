# Workers

Thread architecture, signals, and cancellation patterns. Files:
`fractal/worker.py` (~118 lines), `fractal/basin_solver.py` (~126 lines),
`fractal/view.py` (~405 lines).

> Cross-ref: [data-shapes.md](data-shapes.md) for signal payloads.
> Cross-ref: [fractal-compute.md](fractal-compute.md) for progressive levels.

## FractalWorker(QThread)

Runs the progressive computation pipeline in a background thread.

### Lifecycle

1. Created with a `FractalTask`, a `ComputeBackend`, and a list of progressive
   resolution levels
2. If `task.basin` and `task.params.friction > 0`, computes `saddle_energy(params)`
   for early termination
3. Iterates through levels. Dispatches based on mode:
   - **Basin mode**: calls `simulate_basin_batch()` (DOP853, per-trajectory adaptive),
     emits `level_complete(resolution, final_state, None)` where `final_state` is `(N, 4)`
   - **Angle mode**: calls `backend.simulate_batch()` (vectorized RK4),
     emits `level_complete(resolution, snapshots, final_velocities)`
4. Emits `progress(steps_done, total_steps)` during computation
5. Emits `all_complete()` when all levels finish
6. Thread exits → `finished` signal

### Cancellation

`cancel()` sets an atomic flag. In angle mode, checked every ~100 RK4 steps
(~1s response time). In basin mode, checked every ~100 trajectories.
The worker exits its level loop early when cancelled.

## FractalView Orchestration

`FractalView` (in `fractal/view.py`) is the main coordinator:

### Signal Wiring

```
canvas.viewport_changed  --> _on_viewport_changed --> _start_computation
controls.time_index_changed --> _on_time_index_changed --> canvas.set_time_index
controls.colormap_changed --> _on_colormap_changed --> canvas.set_colormap
controls.resolution_changed --> _on_resolution_changed --> _start_computation
controls.physics_changed --> _on_physics_changed --> invalidate cache + recompute
controls.t_end_changed --> _on_t_end_changed --> clear cache + recompute
controls.zoom_out_clicked --> _on_zoom_out --> canvas.zoom_out
controls.tool_mode_changed --> canvas.set_tool_mode
canvas.hover_updated --> _on_hover_updated (inspect tool lookup)
canvas.pan_started --> _on_pan_started --> canvas.set_pan_background
```

### Computation Pipeline

`_start_computation(viewport)`:
1. Check cache for full-resolution result → cache hit: display immediately, clear stale overlay, return
2. Check cache for best lower-resolution match → display as interim preview
3. Filter progressive levels: skip resolutions already cached at this viewport+params
4. If all levels cached → clear stale overlay, return (no worker needed)
5. Cancel any running worker
6. Create new `FractalWorker` with task, backend, and filtered levels
7. Connect worker signals
8. Show loading overlay
9. Start worker thread

### Worker Lifecycle Management

**Retiring workers pattern**: When a new computation starts, the old worker is
moved to `_retiring_workers` list rather than being destroyed immediately.
This prevents the QThread from being garbage-collected while still running.

```python
_cancel_worker():
    old_worker = self._worker
    self._worker = None
    disconnect all signals from old_worker
    if old_worker.isRunning():
        old_worker.cancel()
        self._retiring_workers.append(old_worker)
        old_worker.finished.connect(lambda: _cleanup_retired(old_worker))
```

Retired workers clean themselves up via their `finished` signal.

### Worker Signal Handlers

- `_on_level_complete(resolution, data, final_velocities)`: cache the result, display it (data shape differs by mode)
- `_on_progress(steps_done, total_steps)`: update loading overlay percentage
- `_on_all_complete()`: stop loading overlay, activate pending ghost rectangle, clear stale overlay
- `_on_worker_finished()`: clear worker reference if it's still the current one

## Separate Workers Per Mode

- **PendulumSimWorker** (in `pendulum/view.py`): sequential DOP853 for 1–500
  trajectories. Uses `simulation.simulate()`.
- **FractalWorker** (in `fractal/worker.py`): dispatches to vectorized RK4
  (angle mode) or DOP853 basin solver (basin mode).

## No Shared Mutable State

All data passed between threads uses immutable frozen dataclasses and NumPy
arrays treated as immutable. No locks needed.
