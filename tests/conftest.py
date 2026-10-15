"""Shared fixtures and synthetic-data builders for the QC test suite.

No participant data is used anywhere in these tests. Every frame is built
in-memory from literals so the suite is runnable from a fresh clone.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# In-memory trial frames for unit-testing the pure metric functions
# --------------------------------------------------------------------------

def trial_frame(rows, extra_cols=None):
    """Build a minimal trial DataFrame from (condition, rt, response, correct) tuples.

    `rows` entries are dicts; missing keys get task-neutral defaults. This keeps
    the tests readable -- each test declares only the columns it cares about.
    """
    defaults = {
        'trial_id': 'test_trial',
        'condition': None,
        'rt': None,
        'response': None,
        'correct_trial': None,
        'block_level_feedback': '{}',
        'block_num': 0,
        'stimulus': 'x',
    }
    filled = []
    for i, row in enumerate(rows):
        merged = dict(defaults)
        merged.update(row)
        merged.setdefault('trial_index', i)
        filled.append(merged)

    # polars needs a consistent schema; union all keys
    all_keys = set()
    for row in filled:
        all_keys.update(row.keys())
    for row in filled:
        for key in all_keys:
            row.setdefault(key, None)

    df = pl.DataFrame(filled)
    if extra_cols:
        df = df.with_columns(**extra_cols)
    return df


def feedback_rows(flags):
    """Build test_feedback trials. `flags` is a list of bools (feedback shown?).

    Note the production code always discards the FIRST feedback entry, so a
    caller wanting a proportion of 0.5 over two counted entries must pass three.
    """
    return [
        {
            'trial_id': 'test_feedback',
            'block_level_feedback': "{'accuracy': 'Respond faster!'}" if f else '{}',
        }
        for f in flags
    ]


def as_dict(metrics: pl.DataFrame) -> dict:
    """Collapse a (metric, value) frame into a plain dict for assertions."""
    return {row['metric']: row['value'] for row in metrics.to_dicts()}


# --------------------------------------------------------------------------
# End-to-end pipeline sandbox
# --------------------------------------------------------------------------

FLANKER_RAW_COLS = [
    'ITIParams', 'accuracy', 'block_duration', 'block_level_feedback', 'block_num',
    'center_letter', 'choices', 'condition', 'correct_response', 'correct_trial',
    'design_perm', 'exp_stage', 'flanker', 'internal_node_id', 'motor_perm',
    'response', 'rt', 'stimulus', 'stimulus_duration', 'success', 'time_elapsed',
    'trial_duration', 'trial_id', 'trial_index', 'trial_type',
]


def _raw_trial(index, condition, rt, response, correct, trial_id='test_trial',
               feedback=None):
    """A raw jsPsych-shaped trial carrying every column preprocess.py validates."""
    row = {col: None for col in FLANKER_RAW_COLS}
    row.update({
        'trial_id': trial_id,
        'trial_index': index,
        'trial_type': 'html-keyboard-response',
        'exp_stage': 'test',
        'block_num': 0,
        'condition': condition,
        'rt': rt,
        'response': response,
        'correct_trial': correct,
        'accuracy': correct,
        'correct_response': 'f',
        'center_letter': 'H',
        'flanker': 'H',
        'choices': ['f', 'j'],
        'ITIParams': {'min': 0, 'max': 1},
        'block_level_feedback': feedback if feedback is not None else {},
        'time_elapsed': 1000 * index,
    })
    return row


def build_flanker_raw_json():
    """A flanker run engineered to trip four specific flags.

    congruent:   8 correct + 1 incorrect + 1 omission -> acc 0.889, omission 0.10
    incongruent: 5 correct + 3 incorrect + 2 omissions -> acc 0.625, omission 0.20
    feedback:    3 entries, first discarded, 1 of remaining 2 non-empty -> 0.5
    """
    trials = []
    i = 0
    for _ in range(8):
        trials.append(_raw_trial(i, 'congruent', 450.0, 'f', 1)); i += 1
    trials.append(_raw_trial(i, 'congruent', 500.0, 'j', 0)); i += 1
    trials.append(_raw_trial(i, 'congruent', None, None, 0)); i += 1

    for _ in range(5):
        trials.append(_raw_trial(i, 'incongruent', 600.0, 'f', 1)); i += 1
    for _ in range(3):
        trials.append(_raw_trial(i, 'incongruent', 650.0, 'j', 0)); i += 1
    for _ in range(2):
        trials.append(_raw_trial(i, 'incongruent', None, None, 0)); i += 1

    trials.append(_raw_trial(i, None, None, None, None, trial_id='test_feedback')); i += 1
    trials.append(_raw_trial(i, None, None, None, None, trial_id='test_feedback',
                             feedback={'accuracy': 'Respond faster!'})); i += 1
    trials.append(_raw_trial(i, None, None, None, None, trial_id='test_feedback')); i += 1

    return {'trial_data': trials}


@pytest.fixture
def pipeline_sandbox(tmp_path):
    """A throwaway project root laid out the way the three CLI stages expect.

    The stages resolve `output/raw`, `preprocessed_data` and `results` relative
    to the process cwd, so each test gets its own tree. `preprocessed_data` is
    pre-created because rdocfmriqc.analyze.preprocess stats it before deciding
    what to do; the scan itself is in main(), not at import.
    """
    raw_dir = tmp_path / 'output' / 'raw' / 'sub-test' / 'ses-1'
    raw_dir.mkdir(parents=True)
    (tmp_path / 'preprocessed_data').mkdir()

    # A trailing component after run-N matters: the run parser does not strip
    # the .json extension, so `..._run-1.json` would yield run_num="1.json".
    name = 'sub-test_ses-1_task-flanker_rdoc_run-1_dateTime-20250101.json'
    (raw_dir / name).write_text(json.dumps(build_flanker_raw_json()))
    return tmp_path


def run_stage(module, sandbox, *args):
    """Invoke one pipeline stage as a subprocess rooted at `sandbox`."""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, '-m', module, 'sub-test', '--session', 'ses-1', *args],
        cwd=sandbox, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f'{module} exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    return result
