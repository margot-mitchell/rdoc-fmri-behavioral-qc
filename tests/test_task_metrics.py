"""Unit tests for the pure per-task metric functions in analyze/tasks/tasks.py.

These functions are the scientific core of the release: they take a trial
DataFrame and return a (metric, value) frame with no I/O. Each test below pins
a rule that the README states.
"""

import polars as pl
import pytest

from conftest import as_dict, feedback_rows, trial_frame
from rdocfmriqc.analyze.tasks import (
    ax_cpt_rdoc,
    flanker_rdoc,
    go_nogo_rdoc,
    op_only_span_rdoc,
    spatial_task_switching_rdoc,
    stop_signal_rdoc,
    stroop_rdoc,
)
from rdocfmriqc.analyze.tasks.utils import omission_expr


# --------------------------------------------------------------------------
# Trial selection
# --------------------------------------------------------------------------

def test_only_test_trials_are_scored():
    """Practice/instruction rows must not leak into the metrics."""
    df = trial_frame([
        {'trial_id': 'practice_trial', 'condition': 'congruent', 'rt': 900.0,
         'response': 'j', 'correct_trial': 0},
        {'trial_id': 'test_trial', 'condition': 'congruent', 'rt': 400.0,
         'response': 'f', 'correct_trial': 1},
    ])
    out = as_dict(flanker_rdoc(df, filename='sub-x_ses-1_task-flanker.parquet'))
    assert out['congruent_choice_accuracy'] == 1.0
    assert out['congruent_rt'] == 400.0


def test_practice_filename_scores_every_trial():
    """For practice/pretouch files the trial_id filter is intentionally skipped."""
    df = trial_frame([
        {'trial_id': 'practice_trial', 'condition': 'congruent', 'rt': 900.0,
         'response': 'j', 'correct_trial': 0},
        {'trial_id': 'practice_trial', 'condition': 'congruent', 'rt': 500.0,
         'response': 'f', 'correct_trial': 1},
    ])
    out = as_dict(flanker_rdoc(df, filename='sub-x_ses-1_task-flanker_practice.parquet'))
    assert out['congruent_choice_accuracy'] == 0.5
    assert out['congruent_rt'] == 700.0


# --------------------------------------------------------------------------
# Accuracy excludes omissions; incorrect responses are not omissions
# --------------------------------------------------------------------------

def test_accuracy_excludes_omissions_but_counts_errors():
    """README: accuracy is computed only over trials with a response."""
    df = trial_frame([
        {'condition': 'congruent', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
        {'condition': 'congruent', 'rt': 450.0, 'response': 'j', 'correct_trial': 0},
        {'condition': 'congruent', 'rt': None, 'response': None, 'correct_trial': 0},
    ])
    out = as_dict(flanker_rdoc(df, filename='f.parquet'))
    # 1 of the 2 responded trials was correct; the omission is excluded entirely
    assert out['congruent_choice_accuracy'] == 0.5
    # but the omission still counts toward the omission rate
    assert out['congruent_omission_rate'] == pytest.approx(1 / 3)


def test_rt_averages_only_responded_trials():
    df = trial_frame([
        {'condition': 'incongruent', 'rt': 600.0, 'response': 'f', 'correct_trial': 1},
        {'condition': 'incongruent', 'rt': 800.0, 'response': 'j', 'correct_trial': 0},
        {'condition': 'incongruent', 'rt': None, 'response': None, 'correct_trial': 0},
    ])
    out = as_dict(flanker_rdoc(df, filename='f.parquet'))
    assert out['incongruent_rt'] == 700.0


# --------------------------------------------------------------------------
# Omission definition -- one rule for every task (utils.omission_expr)
# --------------------------------------------------------------------------

def minus_one_rows(condition):
    """Two responses, one '-1' no-response sentinel and one null response.

    The flanker, Stroop and AX-CPT experiments write response = -1 when no key
    is pressed, so '-1' must count as an omission and be left out of accuracy.
    """
    return [
        {'condition': condition, 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
        {'condition': condition, 'rt': 410.0, 'response': '-1', 'correct_trial': 0},
        {'condition': condition, 'rt': 420.0, 'response': 'f', 'correct_trial': 1},
        {'condition': condition, 'rt': None, 'response': None, 'correct_trial': 0},
    ]


@pytest.mark.parametrize('task_fn', [flanker_rdoc, stroop_rdoc])
def test_readme_omission_definition_includes_minus_one(task_fn):
    out = as_dict(task_fn(trial_frame(minus_one_rows('congruent')), filename='f.parquet'))
    assert out['congruent_omission_rate'] == 0.5
    assert out['congruent_choice_accuracy'] == 1.0
    assert out['congruent_rt'] == 410.0


def test_ax_cpt_applies_the_same_omission_rule():
    """ax_cpt goes through utils.get_metrics; it must agree with flanker/stroop."""
    out = as_dict(ax_cpt_rdoc(trial_frame(minus_one_rows('AX')), filename='a.parquet'))
    assert out['AX_omission_rate'] == 0.5
    assert out['AX_choice_accuracy'] == 1.0


def test_numeric_minus_one_response_is_an_omission():
    """A run with only numeric responses stores `response` as an integer column."""
    df = trial_frame([
        {'condition': 'congruent', 'rt': 400.0, 'response': 1, 'correct_trial': 1},
        {'condition': 'congruent', 'rt': None, 'response': -1, 'correct_trial': 0},
    ])
    assert df.schema['response'] == pl.Int64
    out = as_dict(flanker_rdoc(df, filename='f.parquet'))
    assert out['congruent_omission_rate'] == 0.5
    assert out['congruent_choice_accuracy'] == 1.0


def test_omission_rule_without_a_response_column_uses_rt():
    df = pl.DataFrame({'rt': [400.0, None], 'correct_trial': [1, 0]})
    assert df.select(omission_expr(df)).to_series().to_list() == [False, True]


def test_spatial_ts_cue_accuracy_excludes_omissions():
    rows = [
        {'condition': 'task_stay_cue_switch', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
        {'condition': 'task_stay_cue_switch', 'rt': 450.0, 'response': 'j', 'correct_trial': 0},
        {'condition': 'task_stay_cue_switch', 'rt': None, 'response': None, 'correct_trial': 0},
    ]
    df = trial_frame(rows, extra_cols={'spatial_cue': pl.Series(['color'] * 3)})
    out = as_dict(spatial_task_switching_rdoc(df, filename='s.parquet'))
    assert out['color_choice_accuracy'] == 0.5


def test_op_only_processing_accuracy_excludes_omissions():
    """Choice accuracy on the 8x8 grid is computed over trials with a response."""
    processing = [
        {'trial_id': 'test_inter-stimulus', 'rt': 900.0, 'response': 'f', 'correct_trial': 1},
        {'trial_id': 'test_inter-stimulus', 'rt': 950.0, 'response': 'j', 'correct_trial': 0},
        {'trial_id': 'test_inter-stimulus', 'rt': None, 'response': '-1', 'correct_trial': 0},
        # last processing trial before the recall grid is excluded by design
        {'trial_id': 'test_inter-stimulus', 'rt': None, 'response': None, 'correct_trial': 0},
        {'trial_id': 'test_trial', 'rt': 1500.0, 'response': '[1]', 'correct_trial': 1},
    ]
    df = trial_frame(processing, extra_cols={'grid_symmetry': pl.Series(['symmetric'] * 5)})
    out = as_dict(op_only_span_rdoc(df, filename='o.parquet'))
    assert out['8x8_symmetric_choice_accuracy'] == 0.5
    assert out['8x8_grid_omission_rate'] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# Inhibition tasks invert the meaning of accuracy
# --------------------------------------------------------------------------

def test_nogo_accuracy_rewards_withholding():
    """README: nogo_choice_accuracy is the proportion of nogo trials with NO response."""
    df = trial_frame([
        {'condition': 'go', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
        {'condition': 'go', 'rt': 500.0, 'response': 'f', 'correct_trial': 1},
        {'condition': 'nogo', 'rt': None, 'response': None, 'correct_trial': 1},
        {'condition': 'nogo', 'rt': None, 'response': None, 'correct_trial': 1},
        {'condition': 'nogo', 'rt': 450.0, 'response': 'f', 'correct_trial': 0},
    ])
    out = as_dict(go_nogo_rdoc(df, filename='g.parquet'))
    assert out['nogo_choice_accuracy'] == pytest.approx(2 / 3)
    # nogo_rt describes only the failed inhibitions
    assert out['nogo_rt'] == 450.0
    assert out['go_choice_accuracy'] == 1.0


def test_stop_accuracy_rewards_withholding():
    df = trial_frame([
        {'condition': 'go', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
        {'condition': 'stop', 'rt': None, 'response': None, 'correct_trial': 1},
        {'condition': 'stop', 'rt': None, 'response': None, 'correct_trial': 1},
        {'condition': 'stop', 'rt': 300.0, 'response': 'f', 'correct_trial': 0},
        {'condition': 'stop', 'rt': 320.0, 'response': 'f', 'correct_trial': 0},
    ])
    out = as_dict(stop_signal_rdoc(df, filename='s.parquet'))
    assert out['stop_choice_accuracy'] == 0.5


def test_ssd_metrics_are_reported_when_present():
    df = trial_frame(
        [
            {'condition': 'go', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
            {'condition': 'stop', 'rt': None, 'response': None, 'correct_trial': 1},
            {'condition': 'stop', 'rt': None, 'response': None, 'correct_trial': 1},
        ],
        extra_cols={'SSD': pl.Series([200.0, 250.0, 300.0])},
    )
    out = as_dict(stop_signal_rdoc(df, filename='s.parquet'))
    assert out['min_SSD'] == 200.0
    assert out['max_SSD'] == 300.0
    assert out['mean_SSD'] == 250.0
    assert out['final_SSD'] == 300.0


# --------------------------------------------------------------------------
# proportion_feedback: the first entry is always discarded
# --------------------------------------------------------------------------

def test_first_feedback_entry_is_always_discarded():
    """Three feedback trials, first discarded, 1 of remaining 2 fired -> 0.5."""
    rows = [{'condition': 'congruent', 'rt': 400.0, 'response': 'f', 'correct_trial': 1}]
    rows += feedback_rows([True, True, False])  # leading True is thrown away
    out = as_dict(flanker_rdoc(trial_frame(rows), filename='f.parquet'))
    assert out['proportion_feedback_flanker'] == 0.5


def test_single_feedback_trial_yields_none():
    """With only one feedback entry there is nothing left after discarding it."""
    rows = [{'condition': 'congruent', 'rt': 400.0, 'response': 'f', 'correct_trial': 1}]
    rows += feedback_rows([True])
    out = as_dict(flanker_rdoc(trial_frame(rows), filename='f.parquet'))
    assert out['proportion_feedback_flanker'] is None


def test_no_feedback_trials_yields_none():
    df = trial_frame([
        {'condition': 'congruent', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
    ])
    out = as_dict(flanker_rdoc(df, filename='f.parquet'))
    assert out['proportion_feedback_flanker'] is None


def test_malformed_feedback_is_treated_as_absent():
    """parse_feedback swallows literal_eval failures and returns 0."""
    rows = [{'condition': 'congruent', 'rt': 400.0, 'response': 'f', 'correct_trial': 1}]
    rows += [
        {'trial_id': 'test_feedback', 'block_level_feedback': '{}'},
        {'trial_id': 'test_feedback', 'block_level_feedback': 'not-a-dict{{'},
        {'trial_id': 'test_feedback', 'block_level_feedback': None},
    ]
    out = as_dict(flanker_rdoc(trial_frame(rows), filename='f.parquet'))
    assert out['proportion_feedback_flanker'] == 0.0


# --------------------------------------------------------------------------
# Degenerate inputs must not raise
# --------------------------------------------------------------------------

@pytest.mark.parametrize('task_fn,name', [
    (flanker_rdoc, 'flanker'),
    (stroop_rdoc, 'stroop'),
    (go_nogo_rdoc, 'goNogo'),
    (stop_signal_rdoc, 'stopSignal'),
])
def test_missing_condition_yields_none_not_crash(task_fn, name):
    """A run where a condition never appears should produce nulls, not an exception."""
    df = trial_frame([
        {'condition': 'go', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
    ])
    out = as_dict(task_fn(df, filename=f'sub-x_ses-1_task-{name}.parquet'))
    assert out  # returned a frame
    assert all(isinstance(k, str) for k in out)


def test_metric_frame_shape_is_metric_value():
    """Downstream stages select exactly ['metric', 'value'], so pin the contract."""
    df = trial_frame([
        {'condition': 'congruent', 'rt': 400.0, 'response': 'f', 'correct_trial': 1},
    ])
    metrics = flanker_rdoc(df, filename='f.parquet')
    assert metrics.columns == ['metric', 'value']
    assert metrics['metric'].dtype == pl.Utf8
