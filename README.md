# rdocfmriqc

Behavioral quality control pipeline for ongoing dense-fMRI aquisition in which we collect >12 hours of task-fMRI as well as resting-state, diffusion, and T1w and T2w anatomical scans from each participant. This package calculates per-session metrics for the 13 cognitive tasks we run and flags performance that may require intervention before the next session (i.e. not understanding instructions or not attempting to follow them, or otherwise subverting the tasks in problematic ways).

The quality of the imaging data is processed and evaluated in a seperate pipeline. This package handles behavioral data only. 

Given we are relying on this to ensure the quality of our data and inform decisions in real-time, it's designed to ensure no silent failures:

1. **One threshold module.** Every threshold is in [`rdocfmriqc/thresholds.py`](rdocfmriqc/thresholds.py).
2. **Every emitted metric must be covered.** The test suite enumerates what the pipeline
   actually emits (`tests/emitted_metrics.py`) and fails if a metric has no rule and is
   not on the explicit never-flagged list.
3. **Tests gate processing.** The production workflow runs `pytest` before it will
   process anything.

## Install

```bash
pip install git+https://github.com/margot-mitchell/rdoc-fmri-behavioral-qc@v0.1.0
```

Pin the tag. `@main` makes "which version produced this flag?" unanswerable, which is
the question this package exists to answer.

## Try it

No data required — the demo generates a synthetic flanker run, pushes it through all
three stages, and prints the QC email:

```bash
make demo
```

The synthetic run is built so its expected values are known in advance (congruent
accuracy 0.889 and omission 0.10; incongruent 0.625 and 0.20; feedback 0.5), and those
straddle the real thresholds, so it raises exactly four flags:

```
task,metric,value,threshold
flanker,incongruent_choice_accuracy,0.625,0.75
flanker,congruent_omission_rate,0.1,0.05
flanker,incongruent_omission_rate,0.2,0.05
flanker,proportion_feedback_flanker,0.5,0.2
```

## How it runs

Three stages, each reading what the previous one wrote, all resolving paths relative to
the working directory:

```
output/raw/<subject>/<session>/*.json
  → preprocess        → preprocessed_data/<subject>/<task>/*.parquet
  → calculate_metrics → results/<subject>/<session>/metrics/<task>_metrics.csv
  → generate_flags    → results/<subject>/<session>/flags/<task>_flags.csv
```

[`docs/architecture.md`](docs/architecture.md) describes the weekly automation, the
email report, and why the data workflows in `.github/workflows/` are inert in this
repository.

## Supported tasks

The task is read from the `task-` component of each raw filename and determines both
the output filename and which analysis function runs.

| Filename task label | Output name |
|---|---|
| `ax_cpt_rdoc` | `axCPT` |
| `cued_task_switching_rdoc` | `cuedTS` |
| `flanker_rdoc` | `flanker` |
| `go_nogo_rdoc` | `goNogo` |
| `n_back_rdoc` | `nBack` |
| `operation_span_rdoc` | `opSpan` |
| `operation_only_rdoc` | `opOnly` |
| `simple_span_rdoc` | `simpleSpan` |
| `spatial_cueing_rdoc` | `spatialCueing` |
| `spatial_task_switching_rdoc` | `spatialTS` |
| `stop_signal_rdoc` | `stopSignal` |
| `stroop_rdoc` | `stroop` |
| `visual_search_rdoc` | `visualSearch` |

A label that resolves to no analysis function is skipped with a warning rather than
failing the run, so check the log if a task you expected is missing from the output.

## Output format

**Metrics** — `metric`, `value`, `task`, `file`.
**Flags** — `task`, `metric`, `value`, `threshold` (for a two-sided check, the bound
that was crossed).

A flags file is written only when a check fails, and re-running a session that now
passes removes its old flags file.

## Flag rules

Which metric is checked against which threshold is set per task in `TASK_RULES` in
[`rdocfmriqc/analyze/generate_flags.py`](rdocfmriqc/analyze/generate_flags.py). Checks
that combine metrics:

- **Stop signal `stop_choice_accuracy`**: two-sided, flagged outside
  `STOP_SIGNAL_STOP_CHOICE_ACCURACY_MIN`–`_MAX`
- **Go/no-go mean accuracy**: mean of go and nogo choice accuracy, flagged below
  `GONOGO_MEAN_CHOICE_ACCURACY`
- **N-back weighted accuracy** per load:
  `match × NBACK_MATCH_WEIGHT + mismatch × NBACK_MISMATCH_WEIGHT`, flagged below
  `NBACK_WEIGHTED_{1,2}BACK_CHOICE_ACCURACY`; match and mismatch accuracy are also
  flagged together when both are below their minimums
- **Span order difference**: irrespective minus respective 4x4 accuracy, flagged above
  `OP_SPAN_ORDER_DIFF` / `SIMPLE_SPAN_ORDER_DIFF`

Never flagged: stop-signal SSDs, n-back `starter_trial_*`, the spatial TS `na`
condition, and the span diagnostics. Any other metric without a rule is logged as a
warning.

## Metric naming

Accuracy is always calculated only from trials with responses. Tasks with their own
metric code use the `choice_` prefix — `congruent_choice_accuracy`,
`AX_choice_accuracy`, `8x8_grid_symmetric_choice_accuracy`.

Tasks using the shared helper in
[`rdocfmriqc/analyze/tasks/utils.py`](rdocfmriqc/analyze/tasks/utils.py) (spatial
cueing, spatial TS conditions, visual search, n-back) emit `<condition>_accuracy`, e.g.
`doublecue_accuracy`. The flag rules use the exact emitted names.

## Accuracy rules

- **Standard accuracy**: from trials with responses only; omissions excluded.
- **Go/no-go `nogo_choice_accuracy`**: proportion of nogo trials with *no* response —
  successful inhibition, so the sense is inverted.
- **Stop signal `stop_choice_accuracy`**: likewise, proportion of stop trials with no
  response.

## Omission rate

A trial is an **omission** when the participant did not respond:

- `rt` is null or missing, **or**
- `response` is null or empty, **or**
- `response == '-1'`

The flanker, Stroop and AX-CPT experiments record `response = -1` when no key is
pressed, so a null-`rt` test alone misses every omission in those three tasks. One rule
serves every task (`omission_expr` in `rdocfmriqc/analyze/tasks/utils.py`), including
the longest-omission-sequence metric.

Incorrect responses are **not** omissions.

### 8x8 grid omission rate (opSpan, opOnly)

Computed from processing trials (`test_inter-stimulus`), excluding the last processing
trial in each sequence, as the proportion of no-response trials among the remaining
ones.

## Tests

```bash
pip install -e ".[dev]"
python3 -m pytest -q        # or: make test
```

277 tests, no participant data anywhere — every frame is built in memory from literals,
so the suite runs from a fresh clone. Expected values are hand-computed rather than
captured from the implementation, since a fixture recorded from the code under test only
pins in whatever that code already does.

## Contributing

Adding a task or metric: add the metric to `tests/emitted_metrics.py`, add its rule to
`TASK_RULES` (or to the never-flagged list, with a reason), write the failing test, then
implement. The suite fails if an emitted metric has neither a rule nor an exemption.

Run `make hooks` once per checkout to install the identifier guard — see
[`tools/README.md`](tools/README.md).

## Citing

See [`CITATION.cff`](CITATION.cff).
