# Changelog

## 0.2.0 (2026-11-16)

### Span recall accuracy

Span 4x4 recall accuracy (operation span, simple span) is scored per trial. Each recall
trial's response is compared with that trial's target sequence. A trial with no
selection scores 0 and counts toward `4x4_grid_omission_rate`. Practice files are scored
on their recall trials.

Span accuracy and the 4x4 order difference differ from 0.1.0 in runs with at least one
missed recall, and so do the span metrics of practice files.

Four recall trials, with no selection on trial 2:

| Trial | Target | Response | Irrespective | Respective |
|---|---|---|---|---|
| 1 | 3, 7, 12, 5 | 3, 7, 12, 5 | 1 | 1 |
| 2 | 1, 9, 14, 6 | none | 0 | 0 |
| 3 | 2, 8, 11, 15 | 2, 8, 15, 11 | 1 | 0.5 |
| 4 | 4, 10, 13, 0 | 4, 10, 13 | 0.75 | 0.75 |

Irrespective of order: 0.6875. Respective of order: 0.5625. `tests/test_span_metrics.py`
covers these cases.

### Workflows

- Each data job uses a single run condition that combines the repository guard with the
  job's own condition.
- Saving replaces each session's results folder with the newly processed copy.
- All-sessions runs continue across subjects (`fail-fast: false`) and skip subjects with
  no sessions to process.

### Documentation

- Install from `v0.2.0`.
- The README states the span accuracy rule and describes how `response = -1` is treated
  as an omission. The test count is 285.

## 0.1.0 (2026-10-15)

First public release.
