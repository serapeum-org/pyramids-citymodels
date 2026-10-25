# ADR-0004: Versioned documentation strategy

- Status: Accepted
- Date: 2026-10-25

## Context
Readers need documentation matching the version they have installed, and contributors need to
preview docs changes on a pull request before they merge.

## Decision
- Use `mike` to publish versioned docs to GitHub Pages.
- Track `main` as the development version; publish releases under their tag and alias `latest`.
- Publish pull-request previews through the shared `mkdocs-deploy` action's `pull_request` trigger.
- The release tag comes from `pyproject.toml`'s `version`, which commitizen bumps — one source of
  truth for the package version, the git tag, and the docs version selector.

## Consequences
- Contributors update docs alongside code and see a preview on the pull request.
- Users browse historical docs aligned with releases.
- A release that skips `cz bump` publishes docs under the wrong version, so version bumps stay
  commitizen-driven.
