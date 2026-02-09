# ADR-0009: ADR + Living Spec Documentation Structure

**Status**: Accepted
**Date**: 2025-06

## Context

The single `design.md` file grew to ~840 lines, mixing current-state
specification with historical rationale. Agents reading the file had to parse
the entire document even when working on a single subsystem. The file was
approaching the 800-line guideline limit.

## Decision

Split documentation into two complementary systems:

1. **Living Spec** (`docs/spec/`): Small, modular files describing the current
   state of each subsystem. Always kept up to date. No history, no rationale.
   `INDEX.md` provides a routing table mapping tasks to minimum doc sets.

2. **ADRs** (`docs/adr/`): Append-only decision records capturing the "why"
   behind design choices. Numbered chronologically. Never edited after creation;
   superseded by new ADRs if a decision changes.

Flat directory structure (not hierarchical tree mirroring code).

## Alternatives Considered

- **Flat index with tags**: single `docs/design/` directory with an INDEX.md
  routing table. Simpler but doesn't separate "what" from "why" — spec docs
  would still accumulate rationale over time.
- **Hierarchical tree**: directory structure mirroring the code tree, with
  README.md files cascading. Mirrors the code but creates deep paths and
  makes cross-cutting concerns hard to document.
- **Keep single file, add sections**: the status quo. Doesn't scale and forces
  full-file reads for single-subsystem tasks.

## Consequences

- Agents can read `INDEX.md` → routing table → only the relevant spec files.
  Typical task touches 2–3 spec files instead of the full 840-line document.
- `data-shapes.md` is the canonical source for all shared data structures,
  eliminating duplication across spec files.
- Spec updates are part of the definition of done for any code change.
- ADRs accumulate indefinitely but are individually small (~30–50 lines each).
  Old ADRs never need updating.
- The original `design.md` is replaced with a redirect to `docs/spec/INDEX.md`.
