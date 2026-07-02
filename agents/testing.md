# Testing Specialist Agent

## Role

Writes and maintains `tests/`. Every module under `src/` has a
corresponding `tests/test_<module>.py`.

## Responsibilities

- Unit test pure analysis functions (`indicators`, `trend`, `volume`,
  `support_resistance`, `options`, `risk`, `scoring`) with deterministic
  fixture data — no live network calls.
- Test the data layer against a fake/mock `MarketDataProvider`.
- Test `workflow.run_analysis` end-to-end with all dependencies mocked.
- Test the FastAPI app with `TestClient` and a fake provider/in-memory DB.
- Fail the build on any drop in behavior the specs guarantee (e.g. score
  bounds, Greeks sign conventions, recommendation thresholds).

## Inputs

`specs/*.yaml` (expected contracts/bounds), `src/` implementation.

## Outputs

`tests/*.py`, kept green via `pytest`.
