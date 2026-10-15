"""End-to-end tests for the three pipeline stages.

The stages communicate through the filesystem using paths relative to the
process cwd, so each test runs them as subprocesses inside a throwaway project
root built by the `pipeline_sandbox` fixture. Input is one synthetic flanker run
engineered to trip four specific flags.
"""

import polars as pl
import pytest

from conftest import run_stage

PREPROCESS = 'rdocfmriqc.analyze.preprocess'
CALCULATE = 'rdocfmriqc.analyze.calculate_metrics'
FLAGS = 'rdocfmriqc.analyze.generate_flags'


@pytest.fixture
def processed(pipeline_sandbox):
    """Run all three stages once and hand back the sandbox root."""
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    run_stage(FLAGS, pipeline_sandbox)
    return pipeline_sandbox


# --------------------------------------------------------------------------
# Stage 1: raw JSON -> parquet
# --------------------------------------------------------------------------

def test_preprocess_writes_one_parquet_into_a_task_subfolder(pipeline_sandbox):
    run_stage(PREPROCESS, pipeline_sandbox)
    written = list((pipeline_sandbox / 'preprocessed_data').rglob('*.parquet'))
    assert len(written) == 1
    # task name is normalised from 'flanker_rdoc' in the filename to 'flanker'
    assert written[0].parent.name == 'flanker'


def test_preprocess_preserves_every_trial(pipeline_sandbox):
    run_stage(PREPROCESS, pipeline_sandbox)
    written = next((pipeline_sandbox / 'preprocessed_data').rglob('*.parquet'))
    df = pl.read_parquet(written)
    assert df.height == 23  # 20 test trials + 3 feedback trials


def test_preprocess_is_idempotent(pipeline_sandbox):
    """Re-running must skip existing outputs rather than duplicate them."""
    run_stage(PREPROCESS, pipeline_sandbox)
    first = sorted(p.name for p in (pipeline_sandbox / 'preprocessed_data').rglob('*.parquet'))
    run_stage(PREPROCESS, pipeline_sandbox)
    second = sorted(p.name for p in (pipeline_sandbox / 'preprocessed_data').rglob('*.parquet'))
    assert first == second


def test_preprocess_rejects_a_file_missing_required_columns(pipeline_sandbox):
    """A mislabelled file should be skipped, not silently half-processed."""
    import json
    raw = pipeline_sandbox / 'output' / 'raw' / 'sub-test' / 'ses-1'
    (raw / 'sub-test_ses-1_task-stroop_rdoc_run-1_dateTime-x.json').write_text(
        json.dumps({'trial_data': [{'trial_id': 'test_trial', 'rt': 400}]})
    )
    run_stage(PREPROCESS, pipeline_sandbox)
    stroop = pipeline_sandbox / 'preprocessed_data' / 'sub-test' / 'stroop'
    assert not stroop.exists() or not list(stroop.glob('*.parquet'))


# --------------------------------------------------------------------------
# Stage 2: parquet -> metrics CSV
# --------------------------------------------------------------------------

def test_metrics_land_in_the_bids_results_path(pipeline_sandbox):
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    expected = (pipeline_sandbox / 'results' / 'sub-test' / 'ses-1'
                / 'metrics' / 'flanker_metrics.csv')
    assert expected.is_file()


def test_metrics_csv_has_the_documented_columns(pipeline_sandbox):
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    df = pl.read_csv(pipeline_sandbox / 'results' / 'sub-test' / 'ses-1'
                     / 'metrics' / 'flanker_metrics.csv')
    assert df.columns == ['metric', 'value', 'task', 'file']


def test_metric_values_match_the_engineered_fixture(pipeline_sandbox):
    """Golden values for the fixture in conftest.build_flanker_raw_json."""
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    df = pl.read_csv(pipeline_sandbox / 'results' / 'sub-test' / 'ses-1'
                     / 'metrics' / 'flanker_metrics.csv')
    got = dict(zip(df['metric'], df['value']))

    assert got['congruent_choice_accuracy'] == pytest.approx(8 / 9)
    assert got['congruent_omission_rate'] == pytest.approx(0.10)
    assert got['congruent_rt'] == pytest.approx(455.5555555, rel=1e-6)
    assert got['incongruent_choice_accuracy'] == pytest.approx(0.625)
    assert got['incongruent_omission_rate'] == pytest.approx(0.20)
    assert got['incongruent_rt'] == pytest.approx(618.75)
    assert got['proportion_feedback_flanker'] == pytest.approx(0.5)
    assert got['n_trials_in_longest_omission_sequence'] == pytest.approx(2.0)


def test_metrics_stage_records_provenance(pipeline_sandbox):
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    df = pl.read_csv(pipeline_sandbox / 'results' / 'sub-test' / 'ses-1'
                     / 'metrics' / 'flanker_metrics.csv')
    assert df['task'].unique().to_list() == ['flanker']
    assert df['file'].unique().len() == 1
    assert df['file'][0].endswith('.parquet')


# --------------------------------------------------------------------------
# Stage 3: metrics CSV -> flags CSV
# --------------------------------------------------------------------------

def test_flags_file_has_the_documented_columns(processed):
    df = pl.read_csv(processed / 'results' / 'sub-test' / 'ses-1'
                     / 'flags' / 'flanker_flags.csv')
    assert df.columns == ['task', 'metric', 'value', 'threshold']


def test_exactly_the_engineered_violations_are_flagged(processed):
    df = pl.read_csv(processed / 'results' / 'sub-test' / 'ses-1'
                     / 'flags' / 'flanker_flags.csv')
    assert sorted(df['metric'].to_list()) == [
        'congruent_omission_rate',       # 0.10 > 0.05
        'incongruent_choice_accuracy',   # 0.625 < 0.75
        'incongruent_omission_rate',     # 0.20 > 0.05
        'proportion_feedback_flanker',   # 0.50 > 0.20
    ]


def test_passing_metrics_are_not_flagged(processed):
    df = pl.read_csv(processed / 'results' / 'sub-test' / 'ses-1'
                     / 'flags' / 'flanker_flags.csv')
    flagged = set(df['metric'].to_list())
    # 0.889 clears the 0.75 accuracy floor; both RTs clear their ceilings
    assert 'congruent_choice_accuracy' not in flagged
    assert 'congruent_rt' not in flagged
    assert 'incongruent_rt' not in flagged
    assert 'n_trials_in_longest_omission_sequence' not in flagged


def test_flag_rows_carry_the_violated_threshold(processed):
    df = pl.read_csv(processed / 'results' / 'sub-test' / 'ses-1'
                     / 'flags' / 'flanker_flags.csv')
    row = df.filter(pl.col('metric') == 'incongruent_choice_accuracy')
    assert row['value'][0] == pytest.approx(0.625)
    assert row['threshold'][0] == pytest.approx(0.75)


def test_no_flags_file_when_everything_passes(pipeline_sandbox):
    """README: a flags CSV is written only if QC issues are detected."""
    import json
    from conftest import _raw_trial

    trials = [_raw_trial(i, 'congruent', 400.0, 'f', 1) for i in range(10)]
    trials += [_raw_trial(i + 10, 'incongruent', 450.0, 'f', 1) for i in range(10)]
    raw = pipeline_sandbox / 'output' / 'raw' / 'sub-test' / 'ses-1'
    for existing in raw.glob('*.json'):
        existing.unlink()
    (raw / 'sub-test_ses-1_task-flanker_rdoc_run-1_dateTime-x.json').write_text(
        json.dumps({'trial_data': trials})
    )

    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    run_stage(FLAGS, pipeline_sandbox)

    flags = pipeline_sandbox / 'results' / 'sub-test' / 'ses-1' / 'flags'
    assert not list(flags.glob('*.csv'))


def test_rerun_that_passes_removes_a_stale_flags_file(pipeline_sandbox):
    """A flags CSV left by an earlier run must not survive a run with no flags."""
    import json
    from conftest import _raw_trial

    trials = [_raw_trial(i, 'congruent', 400.0, 'f', 1) for i in range(10)]
    trials += [_raw_trial(i + 10, 'incongruent', 450.0, 'f', 1) for i in range(10)]
    raw = pipeline_sandbox / 'output' / 'raw' / 'sub-test' / 'ses-1'
    for existing in raw.glob('*.json'):
        existing.unlink()
    (raw / 'sub-test_ses-1_task-flanker_rdoc_run-1_dateTime-x.json').write_text(
        json.dumps({'trial_data': trials})
    )
    flags = pipeline_sandbox / 'results' / 'sub-test' / 'ses-1' / 'flags'
    flags.mkdir(parents=True)
    stale = flags / 'flanker_flags.csv'
    stale.write_text('task,metric,value,threshold\nflanker,congruent_omission_rate,0.5,0.05\n')

    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    run_stage(FLAGS, pipeline_sandbox)

    assert not stale.exists()


def test_practice_data_gets_metrics_but_no_flags(pipeline_sandbox):
    """README: practice runs are measured but never flagged."""
    import json
    from conftest import build_flanker_raw_json

    raw = pipeline_sandbox / 'output' / 'raw' / 'sub-test' / 'ses-1'
    (raw / 'sub-test_ses-1_task-flanker_rdoc_practice_dateTime-x.json').write_text(
        json.dumps(build_flanker_raw_json())
    )
    run_stage(PREPROCESS, pipeline_sandbox)
    run_stage(CALCULATE, pipeline_sandbox)
    run_stage(FLAGS, pipeline_sandbox)

    metrics_dir = pipeline_sandbox / 'results' / 'sub-test' / 'ses-1' / 'metrics'
    flags_dir = pipeline_sandbox / 'results' / 'sub-test' / 'ses-1' / 'flags'
    assert (metrics_dir / 'flanker_practice_metrics.csv').is_file()
    assert not (flags_dir / 'flanker_practice_flags.csv').exists()


# --------------------------------------------------------------------------
# Import hygiene
# --------------------------------------------------------------------------

def test_preprocess_module_is_importable_from_any_directory(tmp_path):
    """Importing preprocess must not scan the cwd (it used to need preprocessed_data/)."""
    import os
    import subprocess
    import sys

    from conftest import REPO_ROOT

    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, '-c', f'import {PREPROCESS}'],
        cwd=tmp_path,  # deliberately has no preprocessed_data/
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
