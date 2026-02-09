# Contributing to Documentation

How to maintain the spec and ADR documentation for this project.

## Two Systems

| System | Location | Describes | Mutability |
|--------|----------|-----------|------------|
| **Living Spec** | `docs/spec/` | Current state of each subsystem | Updated in place |
| **ADRs** | `docs/adr/` | Why a design decision was made | Append-only, never edited |

## When to Update Specs

Spec updates are **part of the definition of done** for any code change that
affects module boundaries, data shapes, signal payloads, or public interfaces.

Ask yourself: "If another agent reads only the spec files, will they have an
accurate picture of the code?" If not, update the relevant spec.

### What to update

- **Adding a file**: update [file-map.md](spec/file-map.md)
- **Changing a data shape**: update [data-shapes.md](spec/data-shapes.md)
- **Adding a signal**: update [data-shapes.md](spec/data-shapes.md) signal table
- **Changing a subsystem**: update the relevant spec file
- **Adding a new subsystem**: create a new spec file, add to [INDEX.md](spec/INDEX.md)

### What NOT to put in specs

- History ("we used to do X, now we do Y") — that goes in an ADR
- Rationale ("we chose X because...") — that goes in an ADR
- Future plans ("phase 3 will...") — those go in ADRs or issue tracker

## When to Write an ADR

Write an ADR when you make a **non-obvious design choice** — one where a future
reader might ask "why did they do it this way?"

Examples that warrant an ADR:
- Choosing between competing approaches
- Making a performance/complexity tradeoff
- Adopting a new dependency or pattern
- Changing an existing architectural decision

Examples that do NOT warrant an ADR:
- Fixing a typo
- Adding a straightforward feature with no design choices
- Updating a dependency version

## ADR Template

```markdown
# ADR-NNNN: Title

**Status**: Accepted | Superseded by ADR-XXXX
**Date**: YYYY-MM

## Context

What is the issue? What constraints exist?

## Decision

What did we decide to do?

## Alternatives Considered

What else was considered and why was it rejected?

## Consequences

What are the results of this decision? Both positive and negative.
```

### ADR Rules

1. **Number sequentially**: next number is one more than the highest existing
2. **Never edit an existing ADR**: if a decision changes, write a new ADR that
   says "Supersedes ADR-NNNN" and update the old ADR's status to "Superseded"
3. **Add to INDEX.md**: every ADR must be listed in `docs/adr/INDEX.md`
4. **Cross-reference from specs**: spec files should link to relevant ADRs
   when the "why" would be helpful

## For Claude Agents

### Reading workflow

1. Start with `docs/spec/INDEX.md`
2. Use the "Reading Guide for Agents" table to find relevant specs
3. Read [data-shapes.md](spec/data-shapes.md) if you need shared type info
4. Read ADRs only when you need to understand *why* something is the way it is

### Writing workflow

1. Make your code changes
2. Update relevant spec files in `docs/spec/`
3. If you made a non-obvious design choice, write an ADR in `docs/adr/`
4. Update `docs/adr/INDEX.md` if you added an ADR
