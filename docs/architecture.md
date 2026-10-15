# How the pipeline runs

This repository is the **analysis package**. The study data, the results and the
credentials live in a separate private repository, which runs the automation against
them. The workflows under `.github/workflows/` are the real ones, reproduced here so
the design is legible — but they are inert outside the private repository (see
[Why the data workflows cannot run here](#why-the-data-workflows-cannot-run-here)).

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

## Weekly operation

In the private repository, on a weekly schedule:

1. **Detect.** Sync the remote study directory and find subjects and sessions with files
   modified inside the window, excluding `prescan`, `anat`, `practice` and `pretouch`
   sessions.
2. **Gate on the tests.** `pytest` runs before any processing. A broken flag rule or a
   renamed metric stops the run rather than publishing wrong flags — the failure mode
   this guards against is not a crash, it is a QC check silently never firing.
3. **Process.** For each new session, sync just that session, run the three stages, and
   upload the results as a build artifact.
4. **Commit.** Collect the artifacts and commit the results tree.
5. **Report.** Collate the flag files and email a summary.

## The QC email

`generate_email_body.sh` builds the report. It reads `compiled-results/<name>_results/flags/`
— the directory the workflow assembles from the per-session artifacts — and writes
`email_body.txt`. It counts flagged rows per file, skips `pretouch` and `prescan`
sessions, and emits a per-session breakdown:

```
sub-demo_ses-1:
  • incongruent_choice_accuracy: 0.625 (threshold: 0.75)
  • congruent_omission_rate: 0.1 (threshold: 0.05)
```

Identity comes entirely from the environment: recipients, credentials and the repository
name are all injected by the workflow, so the script contains no addresses.

`make demo` runs it against synthetic flags and prints the result, which is the quickest
way to see the whole chain end to end.

## Why the data workflows cannot run here

`process_data.yml`, `weekly_automated_processing.yml` and `export_metrics_matrix.yml`
are published for reference. Three things keep them inert:

- **No schedule.** The `schedule:` block is removed here. Only the private repository
  runs this weekly.
- **A repository guard.** Every data job carries
  `if: github.repository == '<private repository>'`, so a fork's run is an immediate
  no-op rather than a three-minute failure.
- **No credentials.** The remote is read from a repository variable and the credentials
  from repository secrets, neither of which exists outside the private repository.

`export_metrics_matrix.yml` additionally has nothing to operate on: it reads `results/`,
which this repository does not contain.

## Repository layout

```
rdocfmriqc/        the package: three CLI stages, the task metric functions,
                     and thresholds.py
tests/             277 tests on synthetic data; no participant data anywhere
demo/              end-to-end run on synthetic data, including the QC email
tools/             identifier scanner, pre-commit hook, results matrix export
docs/              this page
```
