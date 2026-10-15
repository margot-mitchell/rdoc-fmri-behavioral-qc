"""Tests for the longest incorrect-omission streak metric.

The rule (from rdocfmriqc.thresholds.LONGEST_OMISSION_SEQUENCE): count the longest
run of consecutive trials that are BOTH an omission AND incorrect, within a
single block. For goNogo/stopSignal, nogo/stop trials are bridged -- skipped
without breaking the run -- because withholding a response there is correct.
"""

import polars as pl
import pytest

from conftest import trial_frame
from rdocfmriqc.analyze.tasks.longest_omission import (
    _row_omission,
    append_longest_omission_metric,
    compute_longest_incorrect_omission_streak,
)
from rdocfmriqc.analyze.tasks.utils import omission_expr


def streak(rows, task_dir='flanker', filename=None):
    return compute_longest_incorrect_omission_streak(
        trial_frame(rows), task_dir, filename
    )


def omit(block=0, condition='congruent'):
    """An incorrect omission: no response and correct_trial == 0."""
    return {'condition': condition, 'rt': None, 'response': None,
            'correct_trial': 0, 'block_num': block}


def hit(block=0, condition='congruent'):
    return {'condition': condition, 'rt': 400.0, 'response': 'f',
            'correct_trial': 1, 'block_num': block}


# --------------------------------------------------------------------------
# Core counting
# --------------------------------------------------------------------------

def test_empty_frame_is_zero():
    assert compute_longest_incorrect_omission_streak(
        pl.DataFrame({'correct_trial': []}), 'flanker'
    ) == 0


def test_no_omissions_is_zero():
    assert streak([hit(), hit(), hit()]) == 0


def test_counts_a_single_run():
    assert streak([hit(), omit(), omit(), omit(), hit()]) == 3


def test_reports_the_longest_of_several_runs():
    assert streak([omit(), omit(), hit(), omit(), omit(), omit(), omit(), hit()]) == 4


def test_a_correct_response_breaks_the_run():
    assert streak([omit(), omit(), hit(), omit()]) == 2


def test_trailing_run_is_counted():
    assert streak([hit(), omit(), omit()]) == 2


# --------------------------------------------------------------------------
# What counts as an omission here
# --------------------------------------------------------------------------

def test_response_minus_one_counts_as_omission():
    """The '-1' no-response sentinel is an omission, as in utils.omission_expr."""
    rows = [
        {'condition': 'congruent', 'rt': 400.0, 'response': '-1', 'correct_trial': 0},
        {'condition': 'congruent', 'rt': 410.0, 'response': '-1', 'correct_trial': 0},
    ]
    assert streak(rows) == 2


def test_empty_string_response_counts_as_omission():
    rows = [
        {'condition': 'congruent', 'rt': 400.0, 'response': '', 'correct_trial': 0},
        {'condition': 'congruent', 'rt': 410.0, 'response': '', 'correct_trial': 0},
    ]
    assert streak(rows) == 2


@pytest.mark.parametrize('rt,response', [
    (400.0, 'f'),
    (None, None),
    (None, '-1'),
    (400.0, '-1'),
    (400.0, ''),
    (None, 'f'),
    (400.0, None),
])
def test_row_rule_matches_the_shared_omission_rule(rt, response):
    """This module's row-wise check and the task functions must agree."""
    df = pl.DataFrame({'rt': [rt, 1.0], 'response': [response, 'x']},
                      schema={'rt': pl.Float64, 'response': pl.String})
    expected = df.select(omission_expr(df)).to_series()[0]
    assert _row_omission({'rt': rt, 'response': response}) == expected


def test_incorrect_response_alone_is_not_an_omission():
    """An error with a real keypress does not extend the run."""
    rows = [
        omit(),
        {'condition': 'congruent', 'rt': 400.0, 'response': 'j', 'correct_trial': 0},
        omit(),
    ]
    assert streak(rows) == 1


def test_correct_omission_is_not_counted():
    """Both conditions must hold; an omission marked correct does not count."""
    rows = [
        {'condition': 'congruent', 'rt': None, 'response': None, 'correct_trial': 1},
        {'condition': 'congruent', 'rt': None, 'response': None, 'correct_trial': 1},
    ]
    assert streak(rows) == 0


def test_null_correct_trial_is_not_counted():
    rows = [
        {'condition': 'congruent', 'rt': None, 'response': None, 'correct_trial': None},
        {'condition': 'congruent', 'rt': None, 'response': None, 'correct_trial': None},
    ]
    assert streak(rows) == 0


def test_missing_correct_trial_column_returns_none():
    """None means 'not computable', and the caller then omits the metric entirely."""
    df = pl.DataFrame({'rt': [None, None], 'response': [None, None]})
    assert compute_longest_incorrect_omission_streak(df, 'flanker') is None


# --------------------------------------------------------------------------
# Block boundaries
# --------------------------------------------------------------------------

def test_run_does_not_cross_a_block_boundary():
    rows = [omit(block=0), omit(block=0), omit(block=1), omit(block=1), omit(block=1)]
    assert streak(rows) == 3


def test_block_change_resets_even_mid_run():
    rows = [omit(block=0), omit(block=0), omit(block=0), omit(block=1)]
    assert streak(rows) == 3


# --------------------------------------------------------------------------
# nogo/stop bridging
# --------------------------------------------------------------------------

def test_nogo_trials_bridge_a_run_in_gonogo():
    """A withheld nogo response sits between two go omissions without breaking the run."""
    rows = [
        omit(condition='go'),
        {'condition': 'nogo', 'rt': None, 'response': None, 'correct_trial': 1,
         'block_num': 0},
        omit(condition='go'),
    ]
    assert streak(rows, task_dir='goNogo') == 2


def test_stop_trials_bridge_a_run_in_stopsignal():
    rows = [
        omit(condition='go'),
        {'condition': 'stop', 'rt': None, 'response': None, 'correct_trial': 1,
         'block_num': 0},
        omit(condition='go'),
    ]
    assert streak(rows, task_dir='stopSignal') == 2


def test_nogo_does_not_bridge_in_an_unrelated_task():
    """Bridging is task-specific; elsewhere a 'nogo' label is just another trial."""
    rows = [
        omit(condition='go'),
        {'condition': 'nogo', 'rt': 400.0, 'response': 'f', 'correct_trial': 1,
         'block_num': 0},
        omit(condition='go'),
    ]
    assert streak(rows, task_dir='flanker') == 1


# --------------------------------------------------------------------------
# Ordering and the appending wrapper
# --------------------------------------------------------------------------

def test_trials_are_sorted_by_trial_index_before_counting():
    """Row order in the parquet must not change the answer."""
    ordered = trial_frame([omit(), omit(), hit(), omit()])
    shuffled = ordered.sort('trial_index', descending=True)
    assert compute_longest_incorrect_omission_streak(shuffled, 'flanker') == 2


def test_append_adds_one_row_with_a_float_value():
    metrics = pl.DataFrame({'metric': ['congruent_choice_accuracy'], 'value': [0.9]})
    df = trial_frame([omit(), omit(), hit()])
    out = append_longest_omission_metric(metrics, df, 'flanker', 'f.parquet')
    assert out.height == 2
    row = out.filter(pl.col('metric') == 'n_trials_in_longest_omission_sequence')
    assert row['value'][0] == pytest.approx(2.0)


def test_append_is_a_no_op_when_the_metric_cannot_be_computed():
    metrics = pl.DataFrame({'metric': ['congruent_choice_accuracy'], 'value': [0.9]})
    df = pl.DataFrame({'trial_id': ['test_trial'], 'rt': [None]})
    out = append_longest_omission_metric(metrics, df, 'flanker', 'f.parquet')
    assert out.height == 1


def test_append_reports_zero_for_an_empty_frame():
    """An empty trial stream is a real answer (no omissions), not an error."""
    metrics = pl.DataFrame({'metric': ['congruent_choice_accuracy'], 'value': [0.9]})
    out = append_longest_omission_metric(metrics, pl.DataFrame({}), 'flanker', 'f.parquet')
    row = out.filter(pl.col('metric') == 'n_trials_in_longest_omission_sequence')
    assert row['value'][0] == pytest.approx(0.0)


def test_append_swallows_errors_and_returns_metrics_unchanged():
    """A malformed frame must not take down the whole metrics run.

    The opOnly trial selector indexes trial_index directly, so a parquet missing
    that column raises ColumnNotFoundError inside the helper; the wrapper is
    expected to log and drop the metric rather than abort the task.
    """
    metrics = pl.DataFrame({'metric': ['congruent_choice_accuracy'], 'value': [0.9]})
    malformed = pl.DataFrame({
        'trial_id': ['test_inter-stimulus', 'test_inter-stimulus'],
        'correct_trial': [0, 0],
        'rt': [None, None],
    })
    out = append_longest_omission_metric(metrics, malformed, 'opOnly', 'f.parquet')
    assert out.equals(metrics)


def test_only_test_trials_are_considered():
    """Practice rows in the same frame must not contribute to the streak."""
    rows = [
        {'trial_id': 'practice_trial', 'condition': 'congruent', 'rt': None,
         'response': None, 'correct_trial': 0, 'block_num': 0},
        {'trial_id': 'practice_trial', 'condition': 'congruent', 'rt': None,
         'response': None, 'correct_trial': 0, 'block_num': 0},
        {'trial_id': 'test_trial', 'condition': 'congruent', 'rt': None,
         'response': None, 'correct_trial': 0, 'block_num': 0},
    ]
    metrics = pl.DataFrame({'metric': ['x'], 'value': [0.0]})
    out = append_longest_omission_metric(metrics, trial_frame(rows), 'flanker', 'f.parquet')
    row = out.filter(pl.col('metric') == 'n_trials_in_longest_omission_sequence')
    assert row['value'][0] == pytest.approx(1.0)
