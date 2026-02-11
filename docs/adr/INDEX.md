# Architecture Decision Records

Append-only log of design decisions. Each ADR captures the **why** behind a
choice — context, alternatives considered, and consequences. ADRs are never
edited after creation; if a decision is revisited, a new ADR supersedes the old.

## ADRs

| # | Title | Status | Date |
|---|-------|--------|------|
| [0001](0001-vectorized-rk4.md) | Vectorized fixed-step RK4 for fractal compute | Accepted | 2025-01 |
| [0002](0002-compute-backend-protocol.md) | Pluggable compute backend Protocol | Accepted | 2025-01 |
| [0003](0003-unwrapped-angle-storage.md) | Store unwrapped angles, modulo at color time | Accepted | 2025-01 |
| [0004](0004-progressive-rendering.md) | Coarse-to-fine progressive rendering | Accepted | 2025-01 |
| [0005](0005-lut-based-coloring.md) | 4096-entry LUT for color mapping | Accepted | 2025-01 |
| [0006](0006-whole-viewport-cache.md) | Whole-viewport LRU cache (not tile-based) | Accepted | 2025-01 |
| [0007](0007-dual-angle-snapshots.md) | Dual-angle snapshot storage (theta1 + theta2) | Accepted | 2025-06 |
| [0008](0008-tool-mode-architecture.md) | Three-mode tool architecture (zoom/pan/inspect) | Accepted | 2025-06 |
| [0009](0009-adr-living-spec-docs.md) | ADR + Living Spec documentation structure | Accepted | 2025-06 |
| [0010](0010-bivariate-torus-colormaps.md) | Bivariate torus colormaps for dual-angle display | Accepted | 2025-06 |
| [0011](0011-energy-based-early-termination.md) | Energy-based early termination for basin mode | Superseded by 0012 | 2026-02 |
| [0012](0012-basin-dop853-final-state.md) | DOP853 adaptive solver for basin mode (final state only) | Accepted | 2026-02 |
| [0013](0013-convergence-time-brightness.md) | Convergence-time brightness modulation for basin mode | Accepted | 2026-02 |
| [0014](0014-relative-winding-numbers.md) | Relative winding numbers over absolute | Accepted | 2026-02 |

## How to Write an ADR

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for the template and workflow.
