# ADR-0002: Pluggable Compute Backend Protocol

**Status**: Accepted
**Date**: 2025-01

## Context

Multiple compute strategies exist with very different performance profiles
(NumPy, Numba, JAX/Metal). The system needs to use the fastest available
backend without hard-coding dependencies.

## Decision

Define a `ComputeBackend` Protocol (Python structural typing) with a single
`simulate_batch()` method. Auto-select the best available backend at startup
via try/except ImportError, highest performance first (JAX → Numba → NumPy).

## Alternatives Considered

- **Single backend with optional optimizations**: simpler but couples the
  optimization to the baseline code.
- **Abstract base class**: requires explicit registration and inheritance.
  Protocol is more Pythonic and allows duck typing.
- **Plugin/entry-point system**: over-engineered for 3 backends in a desktop app.

## Consequences

- Adding a new backend requires only implementing `simulate_batch()` with the
  correct signature and adding a try/except block in `get_default_backend()`.
- All backends must return `(N, 2, n_samples)` float32 — the Protocol enforces
  this contract implicitly.
- JIT warmup (Numba) and device initialization (JAX) are backend-specific
  concerns handled within each backend file.
