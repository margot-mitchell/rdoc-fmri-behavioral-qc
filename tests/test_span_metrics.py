"""Span recall accuracy: each response is scored against its own target.

Before 0.2.0, get_span_metrics() built the response list and the target list
separately, paired them by position, and skipped any recall whose response was
empty. A missed recall therefore either dropped out of the accuracy average
(stored as "[]", which is what the fMRI export writes) or shifted every later
response onto the previous trial's target (stored as null). A missed recall now
scores 0 against its own target, as in rdocmetrics and rdoc_beh_preprocessing,
and still counts toward the 4x4 omission rate.

Every expected value is worked by hand in the test's docstring, not captured
from the code under test.
"""

import polars as pl
import pytest

from conftest import as_dict
from rdocfmriqc.analyze.generate_flags import derive_metrics
from rdocfmriqc.analyze.tasks import operation_span_rdoc, simple_span_rdoc

IRR = 'mean_4x4_grid_choice_accuracy_irrespective_of_order'
RESP = 'mean_4x4_grid_choice_accuracy_respective_of_order'
OMIT = '4x4_grid_omission_rate'

# Four recall trials, four cells each.
TARGETS = ['[3, 7, 12, 5]', '[1, 9, 14, 6]', '[2, 8, 11, 15]', '[4, 10, 13, 0]']

# Trial 1 perfect; trial 2 is replaced per test; trial 3 has the last two
# cells swapped; trial 4 recalls three of four cells, in order.
TRIAL_1 = '[3, 7, 12, 5]'
TRIAL_3 = '[2, 8, 15, 11]'
TRIAL_4 = '[4, 10, 13]'


def span_frame(rows):
    """Build a span trial frame; list-valued columns are strings, as after preprocessing."""
    columns = ['trial_id', 'trial_index', 'valid_responses', 'correct_cell_order',
               'correct_trial', 'trial_duration', 'block_level_feedback',
               'grid_symmetry', 'rt', 'response']
    filled = [{col: row.get(col) for col in columns} | {'trial_index': i}
              for i, row in enumerate(rows)]
    schema = {'trial_id': pl.String, 'trial_index': pl.Int64,
              'valid_responses': pl.String, 'correct_cell_order': pl.String,
              'correct_trial': pl.Int64, 'trial_duration': pl.Int64,
              'block_level_feedback': pl.String, 'grid_symmetry': pl.String,
              'rt': pl.Float64, 'response': pl.String}
    return pl.DataFrame(filled, schema=schema)


def recall(responses, targets=TARGETS, trial_id='test_trial'):
    return [{'trial_id': trial_id, 'valid_responses': r, 'correct_cell_order': t,
             'trial_duration': 5000} for r, t in zip(responses, targets)]


def processing(n=2):
    return [{'trial_id': 'test_inter-stimulus', 'grid_symmetry': 'symmetric',
             'rt': 900.0, 'response': 'f', 'correct_trial': 1} for _ in range(n)]


def test_missed_recall_stored_as_empty_list_scores_zero():
    """The fMRI export writes a missed recall as "[]".

    Per trial (irrespective, respective): 1 = (4/4, 4/4); 2 = missed (0, 0);
    3 = (4/4, 2/4); 4 = (3/4, 3/4).
    Irrespective = (1 + 0 + 1 + 0.75) / 4 = 0.6875.
    Respective   = (1 + 0 + 0.5 + 0.75) / 4 = 0.5625.
    Omission     = 1 / 4.
    The old code skipped trial 2 and reported 2.75 / 3 and 2.25 / 3.
    """
    df = span_frame(recall([TRIAL_1, '[]', TRIAL_3, TRIAL_4]))
    out = as_dict(simple_span_rdoc(df, filename='s.parquet'))
    assert out[IRR] == pytest.approx(0.6875)
    assert out[RESP] == pytest.approx(0.5625)
    assert out[OMIT] == pytest.approx(0.25)


def test_missed_recall_stored_as_null_does_not_shift_later_trials():
    """Same trials, the miss stored as null.

    Each response is still scored against its own target, so the values match
    the "[]" case: 0.6875, 0.5625, omission 1 / 4. The old code dropped the null
    from the responses only, scored trial 3's response against trial 2's
    target and trial 4's against trial 3's, and never scored trial 4's target:
    (1 + 0 + 0) / 3 for both.
    """
    df = span_frame(recall([TRIAL_1, None, TRIAL_3, TRIAL_4]))
    out = as_dict(simple_span_rdoc(df, filename='s.parquet'))
    assert out[IRR] == pytest.approx(0.6875)
    assert out[RESP] == pytest.approx(0.5625)
    assert out[OMIT] == pytest.approx(0.25)


def test_no_missed_recall_is_unchanged():
    """Trial 2 recalled perfectly.

    Irrespective = (1 + 1 + 1 + 0.75) / 4 = 0.9375.
    Respective   = (1 + 1 + 0.5 + 0.75) / 4 = 0.8125.
    """
    df = span_frame(recall([TRIAL_1, '[1, 9, 14, 6]', TRIAL_3, TRIAL_4]))
    out = as_dict(simple_span_rdoc(df, filename='s.parquet'))
    assert out[IRR] == pytest.approx(0.9375)
    assert out[RESP] == pytest.approx(0.8125)
    assert out[OMIT] == 0.0


def test_every_recall_missed_scores_zero():
    """Four misses: every trial scores 0, and the omission rate is 4 / 4."""
    df = span_frame(recall(['[]', '[]', None, '[]']))
    out = as_dict(simple_span_rdoc(df, filename='s.parquet'))
    assert out[IRR] == 0.0
    assert out[RESP] == 0.0
    assert out[OMIT] == 1.0


def test_order_difference_uses_the_corrected_accuracies():
    """The flagged order difference is irrespective minus respective.

    0.6875 - 0.5625 = 0.125. The old code gave 2.75 / 3 - 2.25 / 3 = 0.1667.
    """
    df = span_frame(recall([TRIAL_1, '[]', TRIAL_3, TRIAL_4]))
    out = as_dict(simple_span_rdoc(df, filename='s.parquet'))
    derived = derive_metrics(out, 'simple_span')
    assert derived['4x4_grid_order_difference'] == pytest.approx(0.125)


def test_operation_span_scores_recall_the_same_way():
    """Operation span shares get_span_metrics(): 0.6875, 0.5625, omission 1 / 4."""
    df = span_frame(processing() + recall([TRIAL_1, '[]', TRIAL_3, TRIAL_4]))
    out = as_dict(operation_span_rdoc(df, filename='o.parquet'))
    assert out[IRR] == pytest.approx(0.6875)
    assert out[RESP] == pytest.approx(0.5625)
    assert out[OMIT] == pytest.approx(0.25)


@pytest.mark.parametrize('task_fn', [simple_span_rdoc, operation_span_rdoc])
def test_practice_span_metrics_use_recall_trials_only(task_fn):
    """A practice file holds instruction and stimulus rows as well as recall rows.

    Two practice recall trials: one perfect, one missed. Over recall trials only,
    irrespective = respective = (1 + 0) / 2 = 0.5 and omission = 1 / 2. The old
    code scored every row of the file, so the five non-recall rows and the miss
    all read as omissions (6 / 7), and the miss left the accuracy (1.0).
    """
    rows = ([{'trial_id': 'instructions'}]
            + [{'trial_id': 'practice_stim'} for _ in range(4)]
            + recall(['[1, 2, 3, 4]', '[]'], targets=['[1, 2, 3, 4]', '[5, 6, 7, 8]'],
                     trial_id='practice_trial'))
    df = span_frame(rows)
    out = as_dict(task_fn(df, filename='sub-x_ses-00_task-span_practice.parquet'))
    assert out[IRR] == pytest.approx(0.5)
    assert out[RESP] == pytest.approx(0.5)
    assert out[OMIT] == pytest.approx(0.5)
