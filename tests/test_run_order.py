"""Results do not depend on the order in which files are listed.

- Two raw files for one run. When two raw files would be saved under one run's
  parquet name, each recording is kept, named by its dateTime, and the clash is
  logged. If the dateTime cannot tell them apart, preprocess stops.
- Several runs in one metrics file. calculate_metrics writes every run of a
  task into one <task>_metrics.csv; generate_flags checks every run on its own,
  and each flag row names its run in a `file` column.

Every listing is also sorted. Expected values are worked by hand in each
test's docstring, not captured from the code under test.
"""

import json
import os
import subprocess
import sys

import polars as pl
import pytest

from conftest import REPO_ROOT, _raw_trial, build_flanker_raw_json, run_stage
from rdocfmriqc.analyze import calculate_metrics, preprocess
from rdocfmriqc.analyze.generate_flags import check_runs, check_thresholds_from_csv

PREPROCESS = 'rdocfmriqc.analyze.preprocess'
CALCULATE = 'rdocfmriqc.analyze.calculate_metrics'
FLAGS = 'rdocfmriqc.analyze.generate_flags'

# Raw names follow the fMRI export: run before task, then '_fmri_dateTime-<n>'.
# The double underscore in the task part carries through to the parquet name.
RAW_RUN_1 = 'sub-test_ses-1_run-1_task-flanker_rdoc__fmri_dateTime-100.json'
RAW_RUN_1_AGAIN = 'sub-test_ses-1_run-1_task-flanker_rdoc__fmri_dateTime-200.json'
RAW_RUN_2 = 'sub-test_ses-1_run-2_task-flanker_rdoc__fmri_dateTime-200.json'
PARQUET_RUN_1 = 'sub-test_ses-1_task-flanker_rdoc__run-1.parquet'
PARQUET_RUN_2 = 'sub-test_ses-1_task-flanker_rdoc__run-2.parquet'
PARQUET_RUN_1_AT_100 = 'sub-test_ses-1_task-flanker_rdoc__run-1_dateTime-100.parquet'
PARQUET_RUN_1_AT_200 = 'sub-test_ses-1_task-flanker_rdoc__run-1_dateTime-200.parquet'


def low_incongruent_accuracy_raw_json():
    """A flanker recording that fails one check.

    congruent:   6 correct                  -> acc 1.0, omission 0
    incongruent: 6 correct + 4 incorrect    -> acc 0.6, omission 0
    feedback:    3 entries, first discarded, none shown -> 0.0
    19 rows. Only incongruent accuracy fails (0.6 < 0.75); both RTs are under
    1000 ms and there are no omissions.
    """
    trials = [_raw_trial(i, 'congruent', 400.0, 'f', 1) for i in range(6)]
    trials += [_raw_trial(6 + i, 'incongruent', 450.0, 'f', 1) for i in range(6)]
    trials += [_raw_trial(12 + i, 'incongruent', 500.0, 'j', 0) for i in range(4)]
    trials += [_raw_trial(16 + i, None, None, None, None, trial_id='test_feedback')
               for i in range(3)]
    return {'trial_data': trials}


# The engineered fixture (conftest.build_flanker_raw_json, 23 rows) fails four
# checks: incongruent accuracy 0.625 < 0.75, congruent omission 0.10 > 0.05,
# incongruent omission 0.20 > 0.05, feedback 0.5 > 0.2.
ENGINEERED_FLAGS = [
    ('congruent_omission_rate', 0.10, 0.05),
    ('incongruent_choice_accuracy', 0.625, 0.75),
    ('incongruent_omission_rate', 0.20, 0.05),
    ('proportion_feedback_flanker', 0.5, 0.2),
]
LOW_INCONGRUENT_FLAGS = [('incongruent_choice_accuracy', 0.6, 0.75)]


def write_raw(sandbox, files):
    """Replace the sandbox's raw files with {name: trial data}."""
    raw = sandbox / 'output' / 'raw' / 'sub-test' / 'ses-1'
    for existing in raw.glob('*.json'):
        existing.unlink()
    for name, data in files.items():
        (raw / name).write_text(json.dumps(data))


def parquet_names(sandbox):
    return sorted(p.name for p in (sandbox / 'preprocessed_data').rglob('*.parquet'))


def session_csv(sandbox, kind, name):
    return pl.read_csv(sandbox / 'results' / 'sub-test' / 'ses-1' / kind / name)


def flag_rows(df):
    """(file, metric, value, threshold) per flag row, sorted."""
    return sorted(zip(df['file'], df['metric'], df['value'], df['threshold']))


def expected_rows(file, flags):
    return [(file, metric, pytest.approx(value), pytest.approx(threshold))
            for metric, value, threshold in flags]


def run_in_process(main, sandbox, monkeypatch):
    """Call a stage's main() in this process, so a test can control listing order."""
    monkeypatch.chdir(sandbox)
    monkeypatch.setattr(sys, 'argv', ['stage', 'sub-test', '--session', 'ses-1'])
    main()


def reverse_listings(monkeypatch):
    """Make os.listdir and os.walk list names in reverse sorted order."""
    real_listdir, real_walk = os.listdir, os.walk

    def listdir(path='.'):
        return sorted(real_listdir(path), reverse=True)

    def walk(top, *args, **kwargs):
        for root, dirs, files in real_walk(top, *args, **kwargs):
            dirs.sort(reverse=True)
            files.sort(reverse=True)
            yield root, dirs, files

    monkeypatch.setattr(os, 'listdir', listdir)
    monkeypatch.setattr(os, 'walk', walk)


# --------------------------------------------------------------------------
# Two raw files for one run
# --------------------------------------------------------------------------

def test_a_run_with_one_raw_file_keeps_its_usual_name(pipeline_sandbox):
    """Only clashing files get a dateTime suffix: run-1 and run-2 keep today's names."""
    write_raw(pipeline_sandbox, {RAW_RUN_1: build_flanker_raw_json(),
                                 RAW_RUN_2: low_incongruent_accuracy_raw_json()})
    run_stage(PREPROCESS, pipeline_sandbox)
    assert parquet_names(pipeline_sandbox) == [PARQUET_RUN_1, PARQUET_RUN_2]


@pytest.mark.parametrize('reverse', [False, True], ids=['sorted', 'reversed'])
def test_two_raw_files_for_one_run_are_both_kept(pipeline_sandbox, monkeypatch, reverse):
    """Both map to run-1, so each is saved under run-1 plus its own dateTime.

    dateTime-100 holds the engineered fixture (23 rows); dateTime-200 holds the
    low-incongruent recording (19 rows). The old code saved whichever file it
    listed first as ..._run-1.parquet and skipped the other, so the reversed
    listing scored the other recording.
    """
    write_raw(pipeline_sandbox, {RAW_RUN_1: build_flanker_raw_json(),
                                 RAW_RUN_1_AGAIN: low_incongruent_accuracy_raw_json()})
    if reverse:
        reverse_listings(monkeypatch)
    run_in_process(preprocess.main, pipeline_sandbox, monkeypatch)

    task_dir = pipeline_sandbox / 'preprocessed_data' / 'sub-test' / 'flanker'
    assert sorted(p.name for p in task_dir.glob('*.parquet')) == [
        PARQUET_RUN_1_AT_100, PARQUET_RUN_1_AT_200]
    assert pl.read_parquet(task_dir / PARQUET_RUN_1_AT_100).height == 23
    assert pl.read_parquet(task_dir / PARQUET_RUN_1_AT_200).height == 19


def test_two_raw_files_for_one_run_are_reported_as_an_error(pipeline_sandbox):
    """The clash is logged at ERROR level and names both raw files."""
    write_raw(pipeline_sandbox, {RAW_RUN_1: build_flanker_raw_json(),
                                 RAW_RUN_1_AGAIN: low_incongruent_accuracy_raw_json()})
    result = run_stage(PREPROCESS, pipeline_sandbox)
    errors = [line for line in result.stderr.splitlines() if line.startswith('ERROR')]
    assert any(RAW_RUN_1 in line and RAW_RUN_1_AGAIN in line for line in errors), result.stderr


def test_rerun_with_two_raw_files_for_one_run_is_idempotent(pipeline_sandbox):
    """A second preprocess run finds both dateTime files and adds nothing."""
    write_raw(pipeline_sandbox, {RAW_RUN_1: build_flanker_raw_json(),
                                 RAW_RUN_1_AGAIN: low_incongruent_accuracy_raw_json()})
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(PREPROCESS, pipeline_sandbox)
    assert parquet_names(pipeline_sandbox) == [PARQUET_RUN_1_AT_100, PARQUET_RUN_1_AT_200]


def test_clash_that_datetime_cannot_split_fails(pipeline_sandbox):
    """Two raw files map to ..._run-1.parquet and neither has a dateTime.

    preprocess exits non-zero, names both files, and writes no parquet.
    """
    a = 'sub-test_ses-1_task-flanker_rdoc_run-1_a.json'
    b = 'sub-test_ses-1_task-flanker_rdoc_run-1_b.json'
    write_raw(pipeline_sandbox, {a: build_flanker_raw_json(),
                                 b: low_incongruent_accuracy_raw_json()})
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, '-m', PREPROCESS, 'sub-test', '--session', 'ses-1'],
        cwd=pipeline_sandbox, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert a in output and b in output, output
    assert parquet_names(pipeline_sandbox) == []


def test_both_recordings_of_a_clashing_run_are_measured_and_flagged(pipeline_sandbox):
    """End to end: each recording gets its own metrics and its own flags.

    dateTime-100: incongruent accuracy 0.625 and the four engineered flags.
    dateTime-200: incongruent accuracy 0.6 and one flag.
    """
    write_raw(pipeline_sandbox, {RAW_RUN_1: build_flanker_raw_json(),
                                 RAW_RUN_1_AGAIN: low_incongruent_accuracy_raw_json()})
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    run_stage(FLAGS, pipeline_sandbox)

    metrics = session_csv(pipeline_sandbox, 'metrics', 'flanker_metrics.csv')
    incongruent = metrics.filter(pl.col('metric') == 'incongruent_choice_accuracy')
    assert sorted(zip(incongruent['file'], incongruent['value'])) == [
        (PARQUET_RUN_1_AT_100, pytest.approx(0.625)),
        (PARQUET_RUN_1_AT_200, pytest.approx(0.6)),
    ]

    flags = session_csv(pipeline_sandbox, 'flags', 'flanker_flags.csv')
    assert flag_rows(flags) == (expected_rows(PARQUET_RUN_1_AT_100, ENGINEERED_FLAGS)
                                + expected_rows(PARQUET_RUN_1_AT_200, LOW_INCONGRUENT_FLAGS))


# --------------------------------------------------------------------------
# Several runs in one metrics file
# --------------------------------------------------------------------------

GNG_RUN_1 = 'sub-x_ses-1_task-go_nogo_rdoc__run-1.parquet'
GNG_RUN_2 = 'sub-x_ses-1_task-go_nogo_rdoc__run-2.parquet'


def gonogo_metrics(runs):
    """A goNogo metrics frame from {file: {metric: value}}, rows in the order given."""
    rows = [(metric, value, 'goNogo', file)
            for file, metrics in runs.items() for metric, value in metrics.items()]
    return pl.DataFrame(rows, schema=['metric', 'value', 'task', 'file'], orient='row')


# A two-run session: run-1 misses many go trials, run-2 fails to stop.
TWO_RUN_SESSION = {
    GNG_RUN_1: {'go_choice_accuracy': 0.92, 'nogo_choice_accuracy': 0.9,
                'go_omission_rate': 0.5},
    GNG_RUN_2: {'go_choice_accuracy': 1.0, 'nogo_choice_accuracy': 0.25,
                'go_omission_rate': 0.03},
}


def test_every_run_in_a_metrics_file_is_checked():
    """Each run is checked against the go/no-go rules on its own.

    run-1: go 0.92 >= 0.857, nogo 0.9 >= 0.143, mean (0.92 + 0.9) / 2 = 0.91
           >= 0.75, go omission 0.5 > 0.05 -> one flag.
    run-2: go 1.0, nogo 0.25 >= 0.143, mean (1.0 + 0.25) / 2 = 0.625 < 0.75,
           go omission 0.03 <= 0.05 -> one flag.
    The old code kept the last row per metric, so it checked run-2 only.
    """
    assert check_runs(gonogo_metrics(TWO_RUN_SESSION), 'gonogo') == [
        (GNG_RUN_1, 'go_omission_rate', 0.5, 0.05),
        (GNG_RUN_2, 'mean_choice_accuracy', 0.625, 0.75),
    ]


def test_row_order_in_the_metrics_file_does_not_change_the_flags():
    """The same two runs with run-2's rows first give the same flags, in run order.

    The old code would then have checked run-1 only (go omission 0.5).
    """
    reordered = {GNG_RUN_2: TWO_RUN_SESSION[GNG_RUN_2], GNG_RUN_1: TWO_RUN_SESSION[GNG_RUN_1]}
    assert check_runs(gonogo_metrics(reordered), 'gonogo') == [
        (GNG_RUN_1, 'go_omission_rate', 0.5, 0.05),
        (GNG_RUN_2, 'mean_choice_accuracy', 0.625, 0.75),
    ]


def test_a_derived_metric_uses_its_own_run_only():
    """run-2 has no nogo accuracy, so it has no mean accuracy to check.

    run-1: mean (0.9 + 0.2) / 2 = 0.55 < 0.75 -> flag; go 0.9 and nogo 0.2
    pass their own minimums. run-2: go 0.9 passes; its mean must not borrow
    run-1's nogo accuracy, which would give the same 0.55 a second time.
    """
    runs = {GNG_RUN_1: {'go_choice_accuracy': 0.9, 'nogo_choice_accuracy': 0.2},
            GNG_RUN_2: {'go_choice_accuracy': 0.9}}
    assert check_runs(gonogo_metrics(runs), 'gonogo') == [
        (GNG_RUN_1, 'mean_choice_accuracy', pytest.approx(0.55), 0.75),
    ]


def test_a_metrics_file_without_a_file_column_is_one_run():
    """Older metrics files had no file column: the whole file is one run, file ''."""
    df = pl.DataFrame({'metric': ['go_omission_rate'], 'value': [0.5]})
    assert check_runs(df, 'gonogo') == [('', 'go_omission_rate', 0.5, 0.05)]


def test_single_run_check_refuses_a_file_with_several_runs():
    """check_thresholds_from_csv checks one run; given two it fails, naming both."""
    with pytest.raises(ValueError, match='2 runs') as raised:
        check_thresholds_from_csv(gonogo_metrics(TWO_RUN_SESSION), 'gonogo')
    assert GNG_RUN_1 in str(raised.value) and GNG_RUN_2 in str(raised.value)


def test_every_run_is_flagged_end_to_end(pipeline_sandbox):
    """Two flanker runs in one session: the flags file names each flag's run.

    run-1 (engineered fixture): four flags. run-2 (low incongruent accuracy):
    incongruent accuracy 0.6 < 0.75. The old code wrote run-2's single flag,
    or run-1's four, depending on listing order.
    """
    write_raw(pipeline_sandbox, {RAW_RUN_1: build_flanker_raw_json(),
                                 RAW_RUN_2: low_incongruent_accuracy_raw_json()})
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    run_stage(FLAGS, pipeline_sandbox)

    flags = session_csv(pipeline_sandbox, 'flags', 'flanker_flags.csv')
    assert flags.columns == ['task', 'metric', 'value', 'threshold', 'file']
    assert flags['task'].unique().to_list() == ['flanker']
    assert flag_rows(flags) == (expected_rows(PARQUET_RUN_1, ENGINEERED_FLAGS)
                                + expected_rows(PARQUET_RUN_2, LOW_INCONGRUENT_FLAGS))


def test_metrics_file_lists_runs_in_name_order(pipeline_sandbox, monkeypatch):
    """With the directory listed in reverse, run-1's rows still come first.

    Each flanker run emits 8 metrics, so the file column is 8 x run-1 then
    8 x run-2.
    """
    write_raw(pipeline_sandbox, {RAW_RUN_1: build_flanker_raw_json(),
                                 RAW_RUN_2: low_incongruent_accuracy_raw_json()})
    run_stage(PREPROCESS, pipeline_sandbox)
    reverse_listings(monkeypatch)
    run_in_process(calculate_metrics.main, pipeline_sandbox, monkeypatch)

    metrics = session_csv(pipeline_sandbox, 'metrics', 'flanker_metrics.csv')
    assert metrics['file'].to_list() == [PARQUET_RUN_1] * 8 + [PARQUET_RUN_2] * 8
