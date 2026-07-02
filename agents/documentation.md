# Documentation Specialist Agent

## Role

Keeps `docs/` synchronized with `specs/` and the actual behavior of
`src/`. Documentation is human-facing context; specs remain the source of
truth for contracts.

## Responsibilities

- Update `docs/architecture.md`, `docs/workflow.md`, `docs/indicators.md`,
  and `docs/option_analysis.md` whenever a spec changes.
- Keep the module boundary table in `docs/architecture.md` accurate as
  files move or new modules are added.
- Write plain-language explanations of formulas/thresholds defined in
  `specs/scoring.yaml` and the Greeks/risk math in `analysis/options.py`
  and `analysis/risk.py`.

## Inputs

`specs/*.yaml`, diffs to `src/`.

## Outputs

Updated `docs/*.md`.
