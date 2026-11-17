# How the pipeline runs

## The three stages

Each stage is a module with a `__main__`, invoked per subject and session. They
communicate through the filesystem, each reading what the previous one wrote:

```
output/raw/<subject>/<session>/*.json     raw task output, one file per run
        │
        │  python -m rdocfmriqc.analyze.preprocess <subject> --session <session>
        ▼
preprocessed_data/<subject>/<task>/       one parquet per run, columns validated
        │
        │  python -m rdocfmriqc.analyze.calculate_metrics <subject> --session <session>
        ▼
results/<subject>/<session>/metrics/      <task>_metrics.csv   metric,value,task,file
        │
        │  python -m rdocfmriqc.analyze.generate_flags <subject> --session <session>
        ▼
results/<subject>/<session>/flags/        <task>_flags.csv     task,metric,value,threshold
```

All three resolve those directories **relative to the process working directory**, which
is why the test suite and `make demo` can run the whole pipeline inside a throwaway
sandbox with no configuration.

A flags file is written only when a check fails. Re-running a session that now passes
removes its stale flags file rather than leaving it behind.

## Running it on your data

From a directory that holds `output/raw/<subject>/<session>/*.json`:

```bash
python3 -m pytest -q        # run the tests first
python3 -m rdocfmriqc.analyze.preprocess        sub-01 --session ses-1
python3 -m rdocfmriqc.analyze.calculate_metrics sub-01 --session ses-1
python3 -m rdocfmriqc.analyze.generate_flags    sub-01 --session ses-1
```

Leave out `--session` to process every session of a subject. A session's flags are in
`results/<subject>/<session>/flags/`, one file per task with at least one flag.

## Combining sessions

`tools/export_metrics_matrix.py` flattens `results/` into one wide subject × session
matrix per task, under `results/matrix/`. Practice, prescan and pretouch sessions are
left out.

## Repository layout

```
rdocfmriqc/        the package: three CLI stages, the task metric functions,
                     and thresholds.py
tests/             tests on synthetic data; no participant data anywhere
demo/              end-to-end run on synthetic data
tools/             identifier scanner, pre-commit hook, results matrix export
docs/              this page
```
