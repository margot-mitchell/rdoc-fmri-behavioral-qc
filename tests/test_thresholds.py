"""Tests for the flagging logic in analyze/generate_flags.py.

check_thresholds_from_csv decides what counts as a failing run, so the
direction of every comparison matters. The reachability tests check every
metric name the production pipeline actually writes (tests/emitted_metrics.py)
against the rule table, so a renamed metric cannot silently stop being checked.
"""

import logging

import polars as pl
import pytest

from rdocfmriqc import thresholds
from emitted_metrics import EMITTED_METRICS
from rdocfmriqc.analyze.generate_flags import (
    COMMON_RULES,
    DERIVATION_INPUTS,
    JOINT_MIN_RULES,
    NOT_FLAGGED_BY_DESIGN,
    TASK_RULES,
    THRESHOLDS,
    check_thresholds_from_csv,
)


def violations(metric, value, task_name):
    df = pl.DataFrame({'metric': [metric], 'value': [value]})
    return check_thresholds_from_csv(df, task_name)


def run_violations(metrics, task_name):
    """Violations for a run with several metrics, as {metric: value}."""
    df = pl.DataFrame({'metric': list(metrics), 'value': list(metrics.values())})
    return check_thresholds_from_csv(df, task_name)


# --------------------------------------------------------------------------
# Comparison direction
# --------------------------------------------------------------------------

def test_accuracy_flags_below_threshold():
    assert violations('congruent_choice_accuracy', 0.50, 'flanker') == [
        ('congruent_choice_accuracy', 0.50, 0.75)
    ]


def test_accuracy_at_threshold_is_not_flagged():
    """The comparison is strict (`<`), so a value exactly at threshold passes."""
    assert violations('congruent_choice_accuracy', 0.75, 'flanker') == []


def test_accuracy_above_threshold_is_not_flagged():
    assert violations('congruent_choice_accuracy', 0.95, 'flanker') == []


def test_omission_rate_flags_above_threshold():
    assert violations('congruent_omission_rate', 0.30, 'flanker') == [
        ('congruent_omission_rate', 0.30, 0.05)
    ]


def test_omission_rate_below_threshold_is_not_flagged():
    assert violations('congruent_omission_rate', 0.01, 'flanker') == []


def test_rt_flags_above_threshold():
    """RT is 'too slow is bad', the opposite of accuracy."""
    assert violations('stop_signal_go_rt', 900.0, 'stop_signal') == [
        ('stop_signal_go_rt', 900.0, 700)
    ]


def test_rt_below_threshold_is_not_flagged():
    assert violations('stop_signal_go_rt', 500.0, 'stop_signal') == []


def test_unmapped_rt_metric_falls_back_to_general_rt_threshold():
    """Any metric ending in '_rt' with no specific threshold uses RT_THRESHOLD."""
    assert violations('task_stay_cue_stay_rt', 1500.0, 'cued_ts') == [
        ('task_stay_cue_stay_rt', 1500.0, thresholds.RT_THRESHOLD)
    ]
    assert violations('task_stay_cue_stay_rt', 500.0, 'cued_ts') == []


def test_proportion_feedback_flags_above_threshold():
    """Any proportion_feedback_* metric shares one threshold regardless of task."""
    assert violations('proportion_feedback_flanker', 0.5, 'flanker') == [
        ('proportion_feedback_flanker', 0.5,
         thresholds.PROPORTION_FEEDBACK_THRESHOLD)
    ]


def test_longest_omission_sequence_flags_above_threshold():
    assert violations('n_trials_in_longest_omission_sequence', 9.0, 'flanker') == [
        ('n_trials_in_longest_omission_sequence', 9.0,
         thresholds.LONGEST_OMISSION_SEQUENCE)
    ]
    assert violations('n_trials_in_longest_omission_sequence', 2.0, 'flanker') == []


def test_go_rt_is_renamed_per_task():
    """A bare 'go_rt' resolves to a different threshold for each task."""
    assert violations('go_rt', 900.0, 'stop_signal') == [
        ('stop_signal_go_rt', 900.0, thresholds.STOP_SIGNAL_GO_RT)
    ]
    assert violations('go_rt', 1200.0, 'gonogo') == [
        ('gonogo_go_rt', 1200.0, thresholds.RT_THRESHOLD)
    ]


# --------------------------------------------------------------------------
# Null handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize('metric', [
    'congruent_choice_accuracy',
    'congruent_omission_rate',
    'stop_signal_go_rt',
    'proportion_feedback_flanker',
    'n_trials_in_longest_omission_sequence',
])
def test_none_values_never_flag(metric):
    """A metric that could not be computed must not be reported as a failure."""
    assert violations(metric, None, 'flanker') == []
    assert violations(metric, None, 'stop_signal') == []


def test_unknown_metric_is_ignored():
    assert violations('some_metric_nobody_defined', 0.0, 'flanker') == []


def test_unknown_metric_is_logged(caplog):
    """A metric with no rule is reported, so a rename shows up in the CI log."""
    with caplog.at_level(logging.WARNING):
        violations('some_metric_nobody_defined', 0.0, 'flanker')
    assert "No QC rule for flanker metric 'some_metric_nobody_defined'" in caplog.text


# --------------------------------------------------------------------------
# Task-specific rules that used to be unreachable
# --------------------------------------------------------------------------

@pytest.mark.parametrize('task_name,metric,bad_value', [
    ('stop_signal', 'go_omission_rate', 0.5),
    ('gonogo', 'go_choice_accuracy', 0.5),
    ('gonogo', 'nogo_choice_accuracy', 0.0),
    ('cued_ts', 'task_stay_cue_stay_choice_accuracy', 0.5),
    ('cued_ts', 'task_stay_cue_switch_omission_rate', 0.5),
    ('cued_ts', 'task_switch_cue_switch_choice_accuracy', 0.5),
    ('spatial_cueing', 'doublecue_accuracy', 0.5),
    ('spatial_cueing', 'invalid_omission_rate', 0.5),
    ('spatial_ts', 'task_switch_cue_stay_accuracy', 0.5),
    ('spatial_ts', 'task_stay_cue_switch_omission_rate', 0.5),
    ('visual_search', 'conjunction_24_accuracy', 0.5),
    ('operation_span', '8x8_grid_omission_rate', 0.5),
    ('operation_span', '4x4_grid_omission_rate', 0.5),
    ('operation_only_span', '8x8_grid_omission_rate', 0.5),
    ('simple_span', '4x4_grid_omission_rate', 0.5),
])
def test_previously_unreachable_checks_now_flag(task_name, metric, bad_value):
    assert [v[0] for v in violations(metric, bad_value, task_name)] == [metric]


@pytest.mark.parametrize('value,threshold', [
    (0.30, thresholds.STOP_SIGNAL_STOP_CHOICE_ACCURACY_MIN),
    (0.70, thresholds.STOP_SIGNAL_STOP_CHOICE_ACCURACY_MAX),
])
def test_stop_accuracy_is_flagged_outside_its_range(value, threshold):
    """Stop accuracy is two-sided; the flag reports the bound that was crossed."""
    assert violations('stop_choice_accuracy', value, 'stop_signal') == [
        ('stop_choice_accuracy', value, threshold)
    ]


def test_stop_accuracy_inside_its_range_is_not_flagged():
    assert violations('stop_choice_accuracy', 0.50, 'stop_signal') == []


def test_op_span_and_simple_span_use_their_own_4x4_thresholds():
    """Both tasks emit the same metric name but have different minimums (0.4 vs 0.6)."""
    metric = 'mean_4x4_grid_choice_accuracy_irrespective_of_order'
    assert violations(metric, 0.5, 'operation_span') == []
    assert violations(metric, 0.5, 'simple_span') == [
        (metric, 0.5, thresholds.SIMPLE_SPAN_4X4_IRRESPECTIVE_CHOICE_ACCURACY)
    ]


@pytest.mark.parametrize('task_name,respective,flagged', [
    ('operation_span', 0.45, True),   # 0.90 - 0.45 = 0.45 > 0.4
    ('operation_span', 0.60, False),  # 0.30
    ('simple_span', 0.60, True),      # 0.30 > 0.2
    ('simple_span', 0.75, False),     # 0.15
])
def test_order_difference_compares_irrespective_minus_respective(task_name, respective, flagged):
    found = run_violations({
        'mean_4x4_grid_choice_accuracy_irrespective_of_order': 0.90,
        'mean_4x4_grid_choice_accuracy_respective_of_order': respective,
    }, task_name)
    assert ('4x4_grid_order_difference' in [v[0] for v in found]) == flagged


def test_respective_accuracy_alone_is_not_compared_to_the_order_threshold():
    """The old table compared respective accuracy itself against ORDER_DIFF."""
    assert violations('mean_4x4_grid_choice_accuracy_respective_of_order', 0.1,
                      'simple_span') == []


def test_nback_weighted_accuracy_is_flagged():
    found = run_violations({'match_2_accuracy': 0.5, 'mismatch_2_accuracy': 0.7}, 'nback')
    # 0.5 * 0.2 + 0.7 * 0.8 = 0.66 < 0.75; match is not below 0.2, so no joint flag
    assert [v[0] for v in found] == ['weighted_2back_accuracy']
    assert found[0][1] == pytest.approx(0.66)
    assert found[0][2] == thresholds.NBACK_WEIGHTED_2BACK_CHOICE_ACCURACY


def test_nback_joint_rule_needs_both_accuracies_below_their_minimums():
    both_low = run_violations({'match_1_accuracy': 0.1, 'mismatch_1_accuracy': 0.7}, 'nback')
    assert {'match_1_accuracy', 'mismatch_1_accuracy'} <= {v[0] for v in both_low}

    one_low = run_violations({'match_1_accuracy': 0.1, 'mismatch_1_accuracy': 0.95}, 'nback')
    assert not {'match_1_accuracy', 'mismatch_1_accuracy'} & {v[0] for v in one_low}


def test_gonogo_mean_accuracy_is_flagged():
    """go 0.9 and nogo 0.4 each pass their own floor, but their mean (0.65) does not."""
    found = run_violations({'go_choice_accuracy': 0.9, 'nogo_choice_accuracy': 0.4}, 'gonogo')
    assert [v[0] for v in found] == ['mean_choice_accuracy']
    assert found[0][1] == pytest.approx(0.65)


@pytest.mark.parametrize('task_name,metric', [
    ('spatial_ts', 'na_rt'),
    ('nback', 'starter_trial_1_rt'),
])
def test_rt_of_unscored_conditions_is_not_flagged(task_name, metric):
    assert violations(metric, 5000.0, task_name) == []


# --------------------------------------------------------------------------
# Reachability: every metric production emits is checked or exempt
# --------------------------------------------------------------------------

EMITTED = [(task, metric) for task, metrics in EMITTED_METRICS.items() for metric in metrics]

# Metrics checked only together with another metric: the partner value is
# chosen so that the probed metric alone decides whether a flag is raised.
PARTNERS = {
    **{('nback', f'match_{n}_accuracy'): (f'mismatch_{n}_accuracy', 0.7) for n in (1, 2)},
    **{('nback', f'mismatch_{n}_accuracy'): (f'match_{n}_accuracy', 0.1) for n in (1, 2)},
    ('operation_span', 'mean_4x4_grid_choice_accuracy_respective_of_order'):
        ('mean_4x4_grid_choice_accuracy_irrespective_of_order', 1.0),
    ('simple_span', 'mean_4x4_grid_choice_accuracy_respective_of_order'):
        ('mean_4x4_grid_choice_accuracy_irrespective_of_order', 1.0),
}


def probe_values(metric):
    """An implausibly bad and a perfect value, in the metric's own units."""
    if metric.endswith('_rt'):
        return 5000.0, 0.0
    if metric.startswith('n_trials'):
        return 1000.0, 0.0
    return 0.0, 1.0


def flags_at(task, metric, value):
    run = {metric: value}
    if (task, metric) in PARTNERS:
        partner, partner_value = PARTNERS[(task, metric)]
        run[partner] = partner_value
    return run_violations(run, task)


@pytest.mark.parametrize('task,metric', EMITTED, ids=[f'{t}:{m}' for t, m in EMITTED])
def test_every_emitted_metric_is_checked_or_exempt(task, metric):
    first, second = probe_values(metric)
    if metric in NOT_FLAGGED_BY_DESIGN.get(task, set()):
        assert flags_at(task, metric, first) == flags_at(task, metric, second) == []
    else:
        assert flags_at(task, metric, first) != flags_at(task, metric, second), (
            f'{task}:{metric} gives the same flags at {first} and {second}, '
            'so nothing about it is being checked'
        )


def test_no_emitted_metric_is_reported_as_unknown(caplog):
    with caplog.at_level(logging.WARNING):
        for task, metrics in EMITTED_METRICS.items():
            run_violations({m: 0.5 for m in metrics}, task)
    assert 'No QC rule' not in caplog.text


DERIVED = {'mean_choice_accuracy', 'weighted_1back_accuracy',
           'weighted_2back_accuracy', '4x4_grid_order_difference'}


def test_every_rule_targets_an_emitted_or_derived_metric():
    """A rule for a name no task emits is a check that can never run."""
    dead = [(task, metric) for task, rules in TASK_RULES.items() for metric, _, _ in rules
            if metric not in EMITTED_METRICS.get(task, []) and metric not in DERIVED]
    dead += [(task, metric) for task, groups in JOINT_MIN_RULES.items()
             for group in groups for metric, _ in group
             if metric not in EMITTED_METRICS.get(task, [])]
    dead += [(task, metric) for task, metrics in DERIVATION_INPUTS.items()
             for metric in metrics if metric not in EMITTED_METRICS.get(task, [])]
    assert not dead, f'rules for metrics that are never emitted: {dead}'


# --------------------------------------------------------------------------
# Threshold table integrity
# --------------------------------------------------------------------------

def test_every_rule_threshold_exists():
    names = []
    for rules in TASK_RULES.values():
        for _, name, direction in rules:
            names += [f'{name}_MIN', f'{name}_MAX'] if direction == 'range' else [name]
    names += [name for _, name, _ in COMMON_RULES]
    names += [name for groups in JOINT_MIN_RULES.values() for group in groups for _, name in group]
    names += ['RT_THRESHOLD', 'NBACK_MATCH_WEIGHT', 'NBACK_MISMATCH_WEIGHT']
    missing = sorted(n for n in set(names) if n not in THRESHOLDS)
    assert not missing, f'rules point at thresholds that do not exist: {missing}'


def test_rule_directions_are_valid():
    directions = {d for rules in TASK_RULES.values() for _, _, d in rules}
    directions |= {d for _, _, d in COMMON_RULES}
    assert directions <= {'min', 'max', 'range'}


def test_each_metric_has_at_most_one_rule_per_task():
    for task, rules in TASK_RULES.items():
        metrics = [metric for metric, _, _ in rules]
        assert len(metrics) == len(set(metrics)), f'duplicate rule in {task}'


def test_all_thresholds_are_numeric():
    """Rules compare numbers; a non-numeric threshold would be a config error."""
    non_numeric = {
        k: v for k, v in THRESHOLDS.items() if not isinstance(v, (int, float))
    }
    assert not non_numeric, f'non-numeric thresholds would be ignored: {non_numeric}'
