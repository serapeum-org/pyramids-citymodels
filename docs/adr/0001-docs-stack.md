# ADR-0001: Documentation stack

- Status: Accepted
- Date: 2026-10-25

## Context
We need a robust, contributor-friendly documentation stack that supports API reference, diagrams,
tutorials, and versioned publishing — and that matches the rest of the stack, so a contributor
moving between repositories does not relearn it.

## Decision
- Use MkDocs with the Material theme as the primary docs framework.
- Use mkdocstrings[python] for auto-generating API reference from docstrings.
- Prefer Mermaid fenced blocks for diagrams; allow PlantUML if complexity requires it.
- Use mike to publish versioned documentation on GitHub Pages.
- Execute example notebooks ahead of time and render their stored outputs (`mkdocs-jupyter` with
  `execute: false`); `pixi run notebooks` is what proves they still run.

## Consequences
- Contributors document code with Google-style docstrings.
- Diagrams render client-side in the site; heavy diagrams can be split or pre-rendered if needed.
- CI builds the docs on pull requests, pushes to `main`, and releases, publishing with mike.
