"""Task analysis functions for various cognitive tasks."""

import ast
from typing import Callable, List, Dict

import numpy as np
import polars as pl
import logging

from .utils import get_metrics, omission_expr, organize_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_span_metrics(test_trials: pl.DataFrame):
    """Calculate span task metrics.

    Args:
        test_trials (pl.DataFrame): Input dataframe containing span task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    def rename_rt_cols(col: str) -> str:
        return col.replace(
            'rt_each_spatial_location_response_grid', '4x4_grid_response_time'
        ).replace(
            'rt_moving_each_spatial_location_response_grid', '4x4_grid_movement_time'
        )

    def mean_of_differences(values: list[float]) -> float:
        if not values:
            return 0.0
        total = values[0] + sum(
            values[i] - values[i - 1] for i in range(1, len(values))
        )
        return total / len(values)

    def safe_eval(x):
        """Safely evaluate a string as a Python literal.
        
        Args:
            x: Value to evaluate.
            
        Returns:
            Evaluated value or empty list if evaluation fails.
        """
        if x is None:
            return []
        try:
            # Remove any whitespace and ensure it's a valid list string
            x = x.strip()
            if not (x.startswith('[') and x.endswith(']')):
                logging.debug(f"Invalid list format: {x}")
                return []
            # Convert string numbers to integers
            result = ast.literal_eval(x)
            if not isinstance(result, list):
                logging.debug(f"Not a list: {x}")
                return []
            # Convert all elements to integers
            result = [int(i) for i in result]
            logging.debug(f"Successfully evaluated: {x} -> {result}")
            return result
        except (ValueError, SyntaxError) as e:
            logging.warning(f"Failed to evaluate {x}: {e}")
            return []

    def extract_row_rts(
        rt_cols: list[str], func: Callable[[list[float]], float]
    ) -> dict:
        rts = {}
        for col in rt_cols:
            if col not in test_trials.columns:
                continue
            series = test_trials.select(pl.col(col)).to_series()
            rts[col] = [
                func(safe_eval(item))
                for item in series
                if item is not None and safe_eval(item)
            ]
        return rts

    def extract_row_accuracies(test_trials: pl.DataFrame) -> dict[str, list[float]]:
        def get_irrespective_of_order_accuracy(
            response: list[str], correct: list[str]
        ) -> float:
            if not response or not correct:
                return 0.0
            response_set = set(response)
            correct_set = set(correct)
            matches = response_set.intersection(correct_set)
            accuracy = len(matches) / len(correct_set) if correct_set else 0.0
            logging.debug(f"Irrespective accuracy: {response_set} vs {correct_set} = {accuracy}")
            return accuracy

        def get_with_respect_to_order_accuracy(
            response: list[str], correct: list[str]
        ) -> float:
            if not response or not correct:
                return 0.0
            matches = sum(1 for r, c in zip(response, correct) if r == c)
            accuracy = matches / len(correct) if correct else 0.0
            logging.debug(f"Respective accuracy: {response} vs {correct} = {accuracy}")
            return accuracy

        response_series = test_trials.select(pl.col('valid_responses')).to_series()
        responses = [safe_eval(item) for item in response_series if item is not None]
        
        # Check if spatial_sequence exists, otherwise use correct_cell_order
        if 'spatial_sequence' in test_trials.columns:
            correct_series = test_trials.select(pl.col('spatial_sequence')).to_series()
        elif 'correct_cell_order' in test_trials.columns:
            correct_series = test_trials.select(pl.col('correct_cell_order')).to_series()
        else:
            return {
                'accuracy_irrespective_of_order': [0.0],
                'accuracy_respective_of_order': [0.0],
            }
            
        correct_responses = [safe_eval(item) for item in correct_series if item is not None]

        accuracy_irrespective_of_order = []
        accuracy_with_respect_to_order = []

        for response, correct in zip(responses, correct_responses):
            if not response or not correct:
                continue
            irr_acc = get_irrespective_of_order_accuracy(response, correct)
            resp_acc = get_with_respect_to_order_accuracy(response, correct)
            accuracy_irrespective_of_order.append(irr_acc)
            accuracy_with_respect_to_order.append(resp_acc)
            logging.debug(f"Response: {response}, Correct: {correct}")
            logging.debug(f"Irrespective accuracy: {irr_acc}, Respective accuracy: {resp_acc}")

        # If we have no valid accuracies, return 0.0 for both
        if not accuracy_irrespective_of_order and not accuracy_with_respect_to_order:
            return {
                'accuracy_irrespective_of_order': [0.0],
                'accuracy_respective_of_order': [0.0],
            }

        return {
            'accuracy_irrespective_of_order': accuracy_irrespective_of_order,
            'accuracy_respective_of_order': accuracy_with_respect_to_order,
        }

    def get_mean_number_of_responses(test_trials: pl.DataFrame) -> float:
        responses = test_trials.select(pl.col('valid_responses')).to_series()
        response_lengths = [len(safe_eval(x)) for x in responses]
        non_empty_lengths = [length for length in response_lengths if length > 0]
        mean = np.mean(non_empty_lengths) if non_empty_lengths else 0.0
        return float(mean)

    def get_4x4_grid_omission_rate(test_trials: pl.DataFrame) -> float:
        responses = test_trials.select(pl.col('valid_responses')).to_series()
        response_lengths = [len(safe_eval(x)) for x in responses]
        num_empty = sum(1 for length in response_lengths if length == 0)
        total = len(response_lengths)
        omission_rate = num_empty / total if total > 0 else 0.0
        return float(omission_rate)

    def get_time_remaining_after_last_response(test_trials: pl.DataFrame) -> float:
        # Check for response time column with more flexible matching
        rt_col = None
        for col in test_trials.columns:
            if 'valid_responses_timestamps' in col:
                rt_col = col
                break
                
        if rt_col is None:
            logging.warning("No valid_responses_timestamps column found")
            return 0.0
            
        # Get trial durations
        test_trial_durations = test_trials.select(pl.col('trial_duration')).to_series()
        if test_trial_durations.n_unique() > 1:
            logging.warning(f"Multiple trial durations found: {test_trial_durations.unique()}")
            test_trial_duration = test_trial_durations.mean()
        else:
            test_trial_duration = test_trial_durations.item(0)
        
        # Process response times with more lenient parsing
        def get_last_response_time(x):
            try:
                if x is None:
                    return None
                    
                # Try to parse as is first
                try:
                    result = ast.literal_eval(x)
                except:
                    # If that fails, try cleaning the string
                    x = x.strip()
                    if not (x.startswith('[') and x.endswith(']')):
                        return None
                    result = ast.literal_eval(x)
                
                if not isinstance(result, list):
                    return None
                    
                if not result:
                    return None
                    
                # Get the last value and ensure it's a float
                last_val = result[-1]
                if isinstance(last_val, str):
                    last_val = float(last_val)
                elif not isinstance(last_val, (int, float)):
                    return None
                    
                return float(last_val)
                
            except Exception as e:
                logging.warning(f"Failed to process response time {x}: {e}")
                return None
        
        # Get the response times and convert to a list of floats
        responses = test_trials.select(pl.col(rt_col)).to_series()
        last_response_times = []
        
        for response in responses:
            last_time = get_last_response_time(response)
            if last_time is not None:
                last_response_times.append(last_time)
        
        if not last_response_times:
            logging.warning("No valid response times found")
            return 0.0
            
        # Calculate time remaining
        time_remaining = [test_trial_duration - t for t in last_response_times]
        mean_remaining = sum(time_remaining) / len(time_remaining)
        
        return float(mean_remaining)

    rt_cols = [
        'rt_each_spatial_location_response_grid',
        'rt_moving_each_spatial_location_response_grid',
    ]

    first_rts = extract_row_rts(rt_cols, lambda x: x[0])
    mean_rts = extract_row_rts(rt_cols, mean_of_differences)

    mean_first_rts = {}
    mean_mean_rts = {}
    for col in rt_cols:
        if col in first_rts:
            mean_first_rts[col] = np.mean(first_rts[col])
        if col in mean_rts:
            mean_mean_rts[col] = np.mean(mean_rts[col])

    mean_rts = {
        f'mean_{rename_rt_cols(col)}': mean_mean_rts[col] for col in mean_mean_rts
    }
    first_rts = {
        f'first_{rename_rt_cols(col)}': mean_first_rts[col] for col in mean_first_rts
    }

    accuracies = extract_row_accuracies(test_trials)
    logging.debug(f"Accuracies: {accuracies}")

    # Calculate means only if we have valid accuracies
    mean_accuracy_irrespective_of_order = np.mean(accuracies['accuracy_irrespective_of_order']) if accuracies['accuracy_irrespective_of_order'] else 0.0
    mean_accuracy_respective_of_order = np.mean(accuracies['accuracy_respective_of_order']) if accuracies['accuracy_respective_of_order'] else 0.0

    mean_num_responses = get_mean_number_of_responses(test_trials)
    omission_rate = get_4x4_grid_omission_rate(test_trials)
    time_remaining_after_last_response = get_time_remaining_after_last_response(
        test_trials
    )

    # Create a dictionary of all metrics
    metrics_dict = {
        **{k: float(v) for k, v in mean_rts.items()},
        **{k: float(v) for k, v in first_rts.items()},
        'mean_4x4_grid_choice_accuracy_irrespective_of_order': float(mean_accuracy_irrespective_of_order),
        'mean_4x4_grid_choice_accuracy_respective_of_order': float(mean_accuracy_respective_of_order),
        'mean_number_of_responses': float(mean_num_responses),
        '4x4_grid_omission_rate': float(omission_rate),
        'mean_time_remaining_after_last_response': float(time_remaining_after_last_response),
    }

    # Convert to DataFrame with 'metric' and 'value' columns
    return pl.DataFrame(
        {'metric': list(metrics_dict.keys()), 'value': list(metrics_dict.values())}
    )

def ax_cpt_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process AX-CPT task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.
        filename (str, optional): Name of the file, used to determine if it's a practice/pretouch file.

    Returns:
        pl.DataFrame: DataFrame with specified metrics in the requested order.
    """
    # Use filename to determine if this is a practice/pretouch file
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True

    if is_pretouch:
        # For pretouch/practice data, use all trials
        test_trials = df
    else:
        # For test data, filter for test trials
        trial_ids = df.select('trial_id').unique().to_series().to_list()
        if 'test_trial' in trial_ids:
            test_trials = df.filter(pl.col('trial_id') == 'test_trial')
        else:
            test_trials = df.filter(pl.col('trial_id') == 'test_probe')

    if 'success' not in test_trials.columns and 'correct_trial' in test_trials.columns:
        test_trials = test_trials.with_columns(success=pl.col('correct_trial'))

    # Get metrics for each trial type
    metrics = get_metrics(test_trials, group_by=['condition'])
    
    # Helper to get metric for a given condition and metric type
    def get_metric(condition, metric_type):
        row = metrics.filter(pl.col('condition').str.to_lowercase() == condition.lower())
        return row[metric_type][0] if row.height > 0 else None

    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None

    # Calculate proportion_cue/fixation_responses
    fixation_cue_trials = df.filter(
        (pl.col('trial_id').is_in(['test_fixation', 'test_cue']))
        | (
            (pl.col('trial_id') == 'test_inter-stimulus')
            & (pl.col('stimulus').str.contains('fixation'))
        )
    )

    if fixation_cue_trials.height > 0:
        test_indices = test_trials.select('trial_index').to_series().to_list()
        responses = []
        for test_idx in test_indices:
            prev_trials = fixation_cue_trials.filter(
                (pl.col('trial_index') < test_idx)
                & (pl.col('trial_index') >= test_idx - 3)
            )
            has_response = prev_trials.filter(pl.col('rt').is_not_null()).height > 0
            responses.append(1 if has_response else 0)

        total_responses = sum(responses)
        total_test_trials = len(test_indices)
        fixation_cue_responses = min(total_responses / total_test_trials, 1.0)
    else:
        fixation_cue_responses = 0.0

    # Create final metrics DataFrame with specified order
    metrics_df = pl.DataFrame({
        'metric': [
            'AX_choice_accuracy',
            'AX_omission_rate',
            'AX_rt',
            'BX_choice_accuracy',
            'BX_omission_rate',
            'BX_rt',
            'AY_choice_accuracy',
            'AY_omission_rate',
            'AY_rt',
            'BY_choice_accuracy',
            'BY_omission_rate',
            'BY_rt',
            'proportion_cue/fixation_responses',
            'proportion_feedback_ax_cpt'
        ],
        'value': [
            get_metric('AX', 'accuracy'),
            get_metric('AX', 'omission_rate'),
            get_metric('AX', 'rt'),
            get_metric('BX', 'accuracy'),
            get_metric('BX', 'omission_rate'),
            get_metric('BX', 'rt'),
            get_metric('AY', 'accuracy'),
            get_metric('AY', 'omission_rate'),
            get_metric('AY', 'rt'),
            get_metric('BY', 'accuracy'),
            get_metric('BY', 'omission_rate'),
            get_metric('BY', 'rt'),
            fixation_cue_responses,
            proportion_feedback
        ]
    })
    
    return metrics_df

def ax_cpt_rdoc_time_resolved(df: pl.DataFrame) -> pl.DataFrame:
    """Process AX-CPT task data with time-resolved metrics.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    trial_ids = df.select('trial_id').unique().to_series().to_list()

    if 'test_trial' in trial_ids:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_probe')

    if 'success' not in test_trials.columns and 'correct_trial' in test_trials.columns:
        test_trials = test_trials.with_columns(success=pl.col('correct_trial'))

    fixation_cue_trials = df.filter(
        (pl.col('trial_id').is_in(['test_fixation', 'test_cue']))
        | (
            (pl.col('trial_id') == 'test_inter-stimulus')
            & (pl.col('stimulus').str.contains('fixation'))
        )
    )

    if fixation_cue_trials.height > 0:
        test_indices = test_trials.select('trial_index').to_series().to_list()
        responses = []
        for test_idx in test_indices:
            prev_trials = fixation_cue_trials.filter(
                (pl.col('trial_index') < test_idx)
                & (pl.col('trial_index') >= test_idx - 3)
            )
            has_response = prev_trials.filter(pl.col('rt').is_not_null()).height > 0
            responses.append(1 if has_response else 0)

        total_responses = sum(responses)
        total_test_trials = len(test_indices)
        fixation_cue_responses = min(total_responses / total_test_trials, 1.0)
    else:
        fixation_cue_responses = 0.0

    metrics = get_metrics(test_trials)
    fixation_metric = pl.DataFrame(
        {
            'metric': ['proportion_cue/fixation_responses'],
            'value': [fixation_cue_responses],
        }
    )

    organized_metrics = organize_metrics(metrics)
    return pl.concat([organized_metrics, fixation_metric])

def cued_task_switching_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process cued task switching task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    test_trials = test_trials.with_columns(
        success=pl.when(omission_expr(test_trials))
        .then(None)
        .when(pl.col('response') == pl.col('correct_response'))
        .then(True)
        .otherwise(False)
    )

    # Calculate accuracy for each cue type using the 'cue' column
    parity_trials = test_trials.filter(pl.col('cue') == 'Parity')
    parity_accuracy = (
        parity_trials.select(pl.col('success').mean()).item()
        if parity_trials.height > 0
        else None
    )

    odd_even_trials = test_trials.filter(pl.col('cue') == 'Odd-Even')
    odd_even_accuracy = (
        odd_even_trials.select(pl.col('success').mean()).item()
        if odd_even_trials.height > 0
        else None
    )

    high_low_trials = test_trials.filter(pl.col('cue') == 'High-Low')
    high_low_accuracy = (
        high_low_trials.select(pl.col('success').mean()).item()
        if high_low_trials.height > 0
        else None
    )

    magnitude_trials = test_trials.filter(pl.col('cue') == 'Magnitude')
    magnitude_accuracy = (
        magnitude_trials.select(pl.col('success').mean()).item()
        if magnitude_trials.height > 0
        else None
    )

    # Get metrics for task and cue conditions
    metrics = get_metrics(test_trials, group_by=['task_condition', 'cue_condition'])
    
    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None

    # Convert to long format with consistent column names
    melted = pl.DataFrame({
        'metric': [
            'task_stay_cue_stay_choice_accuracy',
            'task_stay_cue_stay_omission_rate',
            'task_stay_cue_stay_rt',
            'task_stay_cue_switch_choice_accuracy',
            'task_stay_cue_switch_omission_rate',
            'task_stay_cue_switch_rt',
            'task_switch_cue_switch_choice_accuracy',
            'task_switch_cue_switch_omission_rate',
            'task_switch_cue_switch_rt',
            'parity_choice_accuracy',
            'odd_even_choice_accuracy',
            'high_low_choice_accuracy',
            'magnitude_choice_accuracy',
            'proportion_feedback_cued_task_switching'
        ],
        'value': [
            metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'stay'))['accuracy'][0] if metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'stay')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'stay'))['omission_rate'][0] if metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'stay')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'stay'))['rt'][0] if metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'stay')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'switch'))['accuracy'][0] if metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'switch')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'switch'))['omission_rate'][0] if metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'switch')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'switch'))['rt'][0] if metrics.filter((pl.col('task_condition') == 'stay') & (pl.col('cue_condition') == 'switch')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'switch') & (pl.col('cue_condition') == 'switch'))['accuracy'][0] if metrics.filter((pl.col('task_condition') == 'switch') & (pl.col('cue_condition') == 'switch')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'switch') & (pl.col('cue_condition') == 'switch'))['omission_rate'][0] if metrics.filter((pl.col('task_condition') == 'switch') & (pl.col('cue_condition') == 'switch')).height > 0 else None,
            metrics.filter((pl.col('task_condition') == 'switch') & (pl.col('cue_condition') == 'switch'))['rt'][0] if metrics.filter((pl.col('task_condition') == 'switch') & (pl.col('cue_condition') == 'switch')).height > 0 else None,
            parity_accuracy,
            odd_even_accuracy,
            high_low_accuracy,
            magnitude_accuracy,
            proportion_feedback
        ]
    })
    
    return melted

def flanker_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process flanker task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        trials = df
    else:
        trials = df.filter(pl.col('trial_id') == 'test_trial')
    omitted = omission_expr(trials)

    # Calculate metrics for congruent trials
    congruent_trials = trials.filter(pl.col('condition').str.to_lowercase() == 'congruent')
    # Calculate accuracy only for trials with responses (filter out omissions)
    congruent_accuracy = congruent_trials.filter(~omitted).select(pl.col('correct_trial').mean()).item()
    congruent_omission = congruent_trials.filter(omitted).height / congruent_trials.height if congruent_trials.height > 0 else None
    congruent_rt = congruent_trials.filter(~omitted).select(pl.col('rt').mean()).item()

    # Calculate metrics for incongruent trials
    incongruent_trials = trials.filter(pl.col('condition').str.to_lowercase() == 'incongruent')
    # Calculate accuracy only for trials with responses (filter out omissions)
    incongruent_accuracy = incongruent_trials.filter(~omitted).select(pl.col('correct_trial').mean()).item()
    incongruent_omission = incongruent_trials.filter(omitted).height / incongruent_trials.height if incongruent_trials.height > 0 else None
    incongruent_rt = incongruent_trials.filter(~omitted).select(pl.col('rt').mean()).item()
    
    # Calculate proportion feedback from block_level_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None
    
    # Create DataFrame with all metrics in the specified order
    metrics = pl.DataFrame({
        'metric': [
            'congruent_choice_accuracy',
            'congruent_omission_rate',
            'congruent_rt',
            'incongruent_choice_accuracy',
            'incongruent_omission_rate',
            'incongruent_rt',
            'proportion_feedback_flanker'
        ],
        'value': [
            congruent_accuracy,
            congruent_omission,
            congruent_rt,
            incongruent_accuracy,
            incongruent_omission,
            incongruent_rt,
            proportion_feedback
        ]
    })
    
    return metrics

def go_nogo_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process go/no-go task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    
    # Calculate go trial metrics
    go_trials = test_trials.filter(pl.col('condition') == 'go')
    go_metrics = get_metrics(go_trials)
    
    # Calculate nogo trial metrics
    nogo_trials = test_trials.filter(pl.col('condition') == 'nogo')
    
    # For nogo trials, accuracy is the proportion of trials with no response (successful inhibition)
    nogo_omitted = omission_expr(nogo_trials)
    nogo_accuracy = nogo_trials.select(nogo_omitted.mean()).item() if nogo_trials.height > 0 else None

    # Calculate nogo RT (only for trials where participant incorrectly responded)
    nogo_rt = nogo_trials.select(pl.col('rt').filter(~nogo_omitted).mean()).item() if nogo_trials.height > 0 else None
    
    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None
    
    # Create final metrics DataFrame with specified order
    metrics_df = pl.DataFrame({
        'metric': [
            'go_choice_accuracy',
            'go_omission_rate',
            'gonogo_go_rt',
            'nogo_choice_accuracy',
            'nogo_rt',
            'proportion_feedback_go_nogo'
        ],
        'value': [
            go_metrics['accuracy'][0] if go_metrics.height > 0 else None,
            go_metrics['omission_rate'][0] if go_metrics.height > 0 else None,
            go_metrics['rt'][0] if go_metrics.height > 0 else None,
            nogo_accuracy,
            nogo_rt,
            proportion_feedback
        ]
    })
    
    return metrics_df

def n_back_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process n-back task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    
    # If delay column doesn't exist, use condition column instead
    group_by = ['condition']
    if 'delay' in test_trials.columns:
        group_by.append('delay')
        
    metrics = get_metrics(test_trials, group_by=group_by)
    melted = organize_metrics(metrics, group_by=group_by)
    
    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None
    
    # Filter out task_na metrics and ensure we have the right columns
    filtered_melted = melted.filter(~pl.col('metric').str.starts_with('task_na')).select(['metric', 'value'])
    
    # Add proportion_feedback as a new row
    if proportion_feedback is not None:
        feedback_df = pl.DataFrame({
            'metric': ['proportion_feedback_n_back'],
            'value': [proportion_feedback]
        }).select(['metric', 'value'])
        filtered_melted = pl.concat([filtered_melted, feedback_df])
    
    return filtered_melted

def operation_span_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process operation span task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
        processing_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
        processing_trials = df.filter(pl.col('trial_id') == 'test_inter-stimulus')

    # Debug logging for data presence
    num_valid_responses = test_trials.filter(pl.col('valid_responses').is_not_null()).height
    num_correct_cell_order = test_trials.filter(pl.col('correct_cell_order').is_not_null()).height
    logging.debug(f"[operation_span_rdoc] test_trials: {test_trials.height}, valid_responses: {num_valid_responses}, correct_cell_order: {num_correct_cell_order}")

    def calculate_processing_metrics(
        processing_trials: pl.DataFrame, full_df: pl.DataFrame
    ) -> pl.DataFrame:
        # Find processing trials to exclude:
        # Exclude if: 
        #   - Row above is test_inter-stimulus 
        #   - AND row after is NOT test_inter-stimulus (i.e., it's the last in a sequence)
        
        trials_to_exclude = []
        
        for idx in processing_trials['trial_index'].to_list():
            # Get the row before this trial
            row_before = full_df.filter(pl.col('trial_index') == idx - 1)
            # Get the row after this trial
            row_after = full_df.filter(pl.col('trial_index') == idx + 1)
            
            # Check if row before is test_inter-stimulus
            if row_before.height > 0 and row_before['trial_id'][0] == 'test_inter-stimulus':
                # Check if row after is NOT test_inter-stimulus
                if row_after.height > 0 and row_after['trial_id'][0] != 'test_inter-stimulus':
                    trials_to_exclude.append(idx)
                # Or if there's no row after (end of data)
                elif row_after.height == 0:
                    trials_to_exclude.append(idx)
        
        # Get all processing trials except those to exclude
        relevant_trials = processing_trials.filter(
            ~pl.col('trial_index').is_in(trials_to_exclude)
        )

        # Calculate omission rate (trials with no response)
        if relevant_trials.height > 0:
            # Omissions are trials where rt is null OR response is empty/null OR response is -1
            omissions = relevant_trials.filter(omission_expr(relevant_trials)).height
            total_trials = relevant_trials.height
            overall_omission_rate = omissions / total_trials if total_trials > 0 else 0.0
        else:
            overall_omission_rate = 0.0

        metrics = get_metrics(processing_trials, group_by=['grid_symmetry'])
        
        # Create processing metrics DataFrame
        processing_metrics = pl.DataFrame({
            'metric': [
                '8x8_grid_symmetric_choice_accuracy',
                '8x8_grid_asymmetric_choice_accuracy',
                '8x8_grid_symmetric_rt',
                '8x8_grid_asymmetric_rt',
                '8x8_grid_omission_rate'
            ],
            'value': [
                metrics.filter(pl.col('grid_symmetry') == 'symmetric')['accuracy'][0] if metrics.filter(pl.col('grid_symmetry') == 'symmetric').height > 0 else None,
                metrics.filter(pl.col('grid_symmetry') == 'asymmetric')['accuracy'][0] if metrics.filter(pl.col('grid_symmetry') == 'asymmetric').height > 0 else None,
                metrics.filter(pl.col('grid_symmetry') == 'symmetric')['rt'][0] if metrics.filter(pl.col('grid_symmetry') == 'symmetric').height > 0 else None,
                metrics.filter(pl.col('grid_symmetry') == 'asymmetric')['rt'][0] if metrics.filter(pl.col('grid_symmetry') == 'asymmetric').height > 0 else None,
                overall_omission_rate
            ]
        })
        return processing_metrics

    processing_metrics = calculate_processing_metrics(processing_trials, df)

    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None

    # Get span metrics as DataFrame
    span_metrics = get_span_metrics(test_trials)
    
    # Add mean_4x4_grid_accuracy_entirely_correct
    mean_4x4_grid_choice_accuracy_entirely_correct = test_trials.select(
        pl.col('correct_trial').mean()
    ).item()
    
    # Create additional metrics DataFrame
    additional_metrics = pl.DataFrame({
        'metric': [
            'mean_4x4_grid_choice_accuracy_entirely_correct',
            'proportion_feedback_operation_span'
        ],
        'value': [
            mean_4x4_grid_choice_accuracy_entirely_correct,
            proportion_feedback
        ]
    })

    # Combine all metrics
    final_metrics = pl.concat([processing_metrics, span_metrics, additional_metrics])
    
    # Sort by metric name
    return final_metrics.sort('metric')

def op_only_span_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process operation-only span task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
        processing_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
        processing_trials = df.filter(pl.col('trial_id') == 'test_inter-stimulus')

    def calculate_processing_metrics(
        processing_trials: pl.DataFrame, full_df: pl.DataFrame
    ) -> pl.DataFrame:
        # Find processing trials to exclude:
        # Exclude if: 
        #   - Row above is test_inter-stimulus 
        #   - AND row after is NOT test_inter-stimulus (i.e., it's the last in a sequence)
        
        trials_to_exclude = []
        
        for idx in processing_trials['trial_index'].to_list():
            # Get the row before this trial
            row_before = full_df.filter(pl.col('trial_index') == idx - 1)
            # Get the row after this trial
            row_after = full_df.filter(pl.col('trial_index') == idx + 1)
            
            # Check if row before is test_inter-stimulus
            if row_before.height > 0 and row_before['trial_id'][0] == 'test_inter-stimulus':
                # Check if row after is NOT test_inter-stimulus
                if row_after.height > 0 and row_after['trial_id'][0] != 'test_inter-stimulus':
                    trials_to_exclude.append(idx)
                # Or if there's no row after (end of data)
                elif row_after.height == 0:
                    trials_to_exclude.append(idx)
        
        # Get all processing trials except those to exclude
        relevant_trials = processing_trials.filter(
            ~pl.col('trial_index').is_in(trials_to_exclude)
        )

        # Calculate omission rate (trials with no response)
        if relevant_trials.height > 0:
            # Omissions are trials where rt is null OR response is empty/null OR response is -1
            omissions = relevant_trials.filter(omission_expr(relevant_trials)).height
            total_trials = relevant_trials.height
            overall_omission_rate = omissions / total_trials if total_trials > 0 else 0.0
        else:
            overall_omission_rate = 0.0

        # Get metrics for both symmetric and asymmetric trials
        symmetric_trials = relevant_trials.filter(pl.col('grid_symmetry') == 'symmetric')
        asymmetric_trials = relevant_trials.filter(pl.col('grid_symmetry') == 'asymmetric')
        
        # Calculate accuracy using correct_trial for processing trials with a response
        responded = ~omission_expr(relevant_trials)
        symmetric_accuracy = symmetric_trials.filter(responded).select(pl.col('correct_trial').mean()).item() if symmetric_trials.height > 0 else None
        symmetric_rt = symmetric_trials.filter(responded).select(pl.col('rt').mean()).item() if symmetric_trials.height > 0 else None

        asymmetric_accuracy = asymmetric_trials.filter(responded).select(pl.col('correct_trial').mean()).item() if asymmetric_trials.height > 0 else None
        asymmetric_rt = asymmetric_trials.filter(responded).select(pl.col('rt').mean()).item() if asymmetric_trials.height > 0 else None
        
        # Create processing metrics DataFrame with specific metrics
        metrics_dict = {
            '8x8_symmetric_choice_accuracy': symmetric_accuracy,
            '8x8_symmetric_rt': symmetric_rt,
            '8x8_asymmetric_choice_accuracy': asymmetric_accuracy,
            '8x8_asymmetric_rt': asymmetric_rt,
            '8x8_grid_omission_rate': overall_omission_rate
        }
        
        # Convert all values to float, handling None values
        metrics_dict = {k: float(v) if v is not None else None for k, v in metrics_dict.items()}
        
        return pl.DataFrame({
            'metric': list(metrics_dict.keys()),
            'value': list(metrics_dict.values())
        })

    processing_metrics = calculate_processing_metrics(processing_trials, df)
    
    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None
    
    # Add proportion_feedback to metrics
    feedback_df = pl.DataFrame({
        'metric': ['proportion_feedback_operation_only_span'],
        'value': [proportion_feedback]
    })
    
    # Combine metrics using concat instead of vstack
    final_metrics = pl.concat([processing_metrics, feedback_df])
    
    return final_metrics

def simple_span_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process simple span task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    mean_4x4_grid_choice_accuracy_entirely_correct = test_trials.select(
        pl.col('correct_trial').mean()
    ).item()

    # Get span metrics as DataFrame
    span_metrics = get_span_metrics(test_trials)
    
    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None
    
    # Add the entirely correct accuracy metric and proportion_feedback
    new_metrics = pl.DataFrame({
        'metric': ['mean_4x4_grid_choice_accuracy_entirely_correct', 'proportion_feedback_simple_span'],
        'value': [mean_4x4_grid_choice_accuracy_entirely_correct, proportion_feedback]
    })
    
    # Combine metrics
    final_metrics = pl.concat([span_metrics, new_metrics])
    
    # Sort by metric name
    return final_metrics.sort('metric')

def spatial_cueing_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process spatial cueing task data and report specified metrics and proportion_feedback."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    
    # Group by 'condition' to get metrics for each cue type
    metrics = get_metrics(test_trials, group_by=['condition'])

    # Helper to get metric for a given cue and metric type
    def get_metric(cue, metric_type):
        row = metrics.filter(pl.col('condition').str.to_lowercase() == cue.lower())
        return row[metric_type][0] if row.height > 0 else None

    # Calculate proportion_feedback from block_level_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None

    # List of cues and metrics in the requested order
    cues = ['doublecue', 'invalid', 'nocue', 'valid']
    metric_types = ['accuracy', 'omission_rate', 'rt']
    metric_names = [f"{cue}_{mtype}" for cue in cues for mtype in metric_types]

    # Gather all metrics in order
    values = [get_metric(cue, mtype) for cue in cues for mtype in metric_types]
    # Insert proportion_feedback at the correct position (after nocue_rt)
    metric_names.insert(9, 'proportion_feedback_spatial_cueing')
    values.insert(9, proportion_feedback)

    return pl.DataFrame({'metric': metric_names, 'value': values})

def spatial_task_switching_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process spatial task switching task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    # Use correct_trial directly instead of calculating success
    test_trials = test_trials.with_columns(
        success=pl.col('correct_trial').cast(pl.Boolean)
    )
    
    # Calculate color judgment accuracy using spatial_cue (trials with a response)
    responded = ~omission_expr(test_trials)
    color_trials = test_trials.filter(
        pl.col('spatial_cue').str.to_lowercase().str.contains('color|col')
    )
    color_accuracy = (
        color_trials.filter(responded).select(pl.col('correct_trial').mean()).item()
        if color_trials.height > 0
        else None
    )

    # Calculate form judgment accuracy using spatial_cue (trials with a response)
    form_trials = test_trials.filter(
        pl.col('spatial_cue').str.to_lowercase().str.contains('form|shape')
    )
    form_accuracy = (
        form_trials.filter(responded).select(pl.col('correct_trial').mean()).item()
        if form_trials.height > 0
        else None
    )

    # Get base metrics grouped by condition
    metrics = get_metrics(test_trials, group_by=['condition'])
    melted = organize_metrics(metrics, group_by=['condition'])

    # Filter out task_na metrics and na_na metrics
    filtered_melted = melted.filter(
        (~pl.col('metric').str.starts_with('task_na'))
        & (
            ~pl.col('metric').is_in(
                ['na_na_accuracy', 'na_na_omission_rate', 'na_na_rt']
            )
        )
    )
    # Ensure filtered_melted has the correct columns even if empty
    if filtered_melted.is_empty():
        filtered_melted = pl.DataFrame({'metric': [], 'value': []})
    filtered_melted = filtered_melted.select(['metric', 'value'])

    # Add color and form accuracy as separate rows
    new_metrics = pl.DataFrame(
        {
            'metric': ['color_choice_accuracy', 'form_choice_accuracy'],
            'value': [color_accuracy, form_accuracy],
        }
    )
    if new_metrics.is_empty():
        new_metrics = pl.DataFrame({'metric': [], 'value': []})
    new_metrics = new_metrics.select(['metric', 'value'])

    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None
    
    # Add proportion_feedback
    feedback_df = pl.DataFrame({
        'metric': ['proportion_feedback_spatial_task_switching'],
        'value': [proportion_feedback]
    })

    # Rename metrics as requested
    rename_map = {
        'task_switch_cue_stay_accuracy': 'task_switch_cue_stay_accuracy',
        'task_switch_cue_stay_omission_rate': 'task_switch_cue_stay_omission_rate',
        'task_switch_cue_stay_rt': 'task_switch_cue_stay_rt',
        'task_stay_cue_switch_choice_accuracy': 'task_stay_cue_switch_choice_accuracy',
        'task_stay_cue_switch_omission_rate': 'task_stay_cue_switch_omission_rate',
        'task_stay_cue_switch_rt': 'task_stay_cue_switch_rt',
        'task_switch_cue_switch_choice_accuracy': 'task_switch_cue_switch_choice_accuracy',
        'task_switch_cue_switch_omission_rate': 'task_switch_cue_switch_omission_rate',
        'task_switch_cue_switch_rt': 'task_switch_cue_switch_rt',
    }
    result = pl.concat([filtered_melted, new_metrics, feedback_df]).with_columns(
        pl.when(pl.col('metric').is_in(list(rename_map.keys())))
        .then(pl.col('metric').replace(rename_map))
        .otherwise(pl.col('metric'))
        .alias('metric')
    )
    # Remove any metrics with names switch_stay_accuracy, switch_stay_omission_rate, or switch_stay_rt if present
    result = result.filter(~pl.col('metric').is_in([
        'switch_stay_accuracy', 'switch_stay_omission_rate', 'switch_stay_rt', 'stay_stay_accuracy', 'stay_stay_omission_rate', 'stay_stay_rt'
    ]))
    return result

def stop_signal_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process stop signal task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    
    # Calculate go trial metrics
    go_trials = test_trials.filter(pl.col('condition') == 'go')
    go_metrics = get_metrics(go_trials)
    
    # Calculate stop trial metrics
    stop_trials = test_trials.filter(pl.col('condition') == 'stop')
    
    # For stop trials, accuracy is the proportion of trials with no response (successful inhibition)
    stop_accuracy = stop_trials.select(omission_expr(stop_trials).mean()).item() if stop_trials.height > 0 else None
    
    # Calculate SSD metrics
    ssd_metrics = {}
    if 'SSD' in test_trials.columns:
        try:
            ssd_metrics = {
                'min_SSD': float(test_trials.select(pl.col('SSD').min()).item() or 0),
                'max_SSD': float(test_trials.select(pl.col('SSD').max()).item() or 0),
                'mean_SSD': float(test_trials.select(pl.col('SSD').mean()).item() or 0),
                'final_SSD': float(test_trials.select(pl.col('SSD').last()).item() or 0),
            }
        except (TypeError, ValueError):
            # If any conversion fails, use 0 as default
            ssd_metrics = {
                'min_SSD': 0.0,
                'max_SSD': 0.0,
                'mean_SSD': 0.0,
                'final_SSD': 0.0,
            }
    
    # Calculate proportion_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None
    
    # Create final metrics DataFrame with specified order
    metrics_df = pl.DataFrame({
        'metric': [
            'go_choice_accuracy',
            'go_omission_rate',
            'stop_signal_go_rt',
            'stop_choice_accuracy',
            'min_SSD',
            'max_SSD',
            'mean_SSD',
            'final_SSD',
            'proportion_feedback_stop_signal'
        ],
        'value': [
            go_metrics['accuracy'][0] if go_metrics.height > 0 else None,
            go_metrics['omission_rate'][0] if go_metrics.height > 0 else None,
            go_metrics['rt'][0] if go_metrics.height > 0 else None,
            stop_accuracy,
            ssd_metrics.get('min_SSD'),
            ssd_metrics.get('max_SSD'),
            ssd_metrics.get('mean_SSD'),
            ssd_metrics.get('final_SSD'),
            proportion_feedback
        ]
    })
    
    return metrics_df

def stroop_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process Stroop task data."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')
    omitted = omission_expr(test_trials)

    # Calculate metrics for congruent trials
    congruent_trials = test_trials.filter(pl.col('condition').str.to_lowercase() == 'congruent')
    # Calculate accuracy only for trials with responses (filter out omissions)
    congruent_accuracy = congruent_trials.filter(~omitted).select(pl.col('correct_trial').mean()).item()
    congruent_omission = congruent_trials.filter(omitted).height / congruent_trials.height if congruent_trials.height > 0 else None
    congruent_rt = congruent_trials.filter(~omitted).select(pl.col('rt').mean()).item()

    # Calculate metrics for incongruent trials
    incongruent_trials = test_trials.filter(pl.col('condition').str.to_lowercase() == 'incongruent')
    # Calculate accuracy only for trials with responses (filter out omissions)
    incongruent_accuracy = incongruent_trials.filter(~omitted).select(pl.col('correct_trial').mean()).item()
    incongruent_omission = incongruent_trials.filter(omitted).height / incongruent_trials.height if incongruent_trials.height > 0 else None
    incongruent_rt = incongruent_trials.filter(~omitted).select(pl.col('rt').mean()).item()
    
    # Calculate proportion feedback from block_level_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0
    
    feedback_values = df.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
    proportion_feedback = feedback_values.mean() if feedback_values.len() > 0 else None
    
    # Create final metrics DataFrame with specified order
    metrics_df = pl.DataFrame({
        'metric': [
            'congruent_choice_accuracy',
            'congruent_omission_rate',
            'congruent_rt',
            'incongruent_choice_accuracy',
            'incongruent_omission_rate',
            'incongruent_rt',
            'proportion_feedback_stroop'
        ],
        'value': [
            congruent_accuracy,
            congruent_omission,
            congruent_rt,
            incongruent_accuracy,
            incongruent_omission,
            incongruent_rt,
            proportion_feedback
        ]
    })
    
    return metrics_df

def visual_search_rdoc(df: pl.DataFrame, filename: str = None) -> pl.DataFrame:
    """Process visual search task data and report specified metrics and proportion_feedback."""
    is_pretouch = False
    if filename is not None:
        fname = filename.lower()
        if 'practice' in fname or 'pretouch' in fname:
            is_pretouch = True
    if is_pretouch:
        test_trials = df
    else:
        test_trials = df.filter(pl.col('trial_id') == 'test_trial')

    # Map condition and set size
    test_trials = test_trials.with_columns([
        pl.col('condition').map_elements(
            lambda x: 'feature' if x == 'feature' else 'conjunction',
            return_dtype=pl.Utf8
        ).alias('condition'),
        pl.col('target_present').map_elements(
            lambda x: 24 if x else 8,
            return_dtype=pl.Int64
        ).alias('set_size')
    ])

    # Group by 'condition' and 'set_size'
    metrics = get_metrics(test_trials, group_by=['condition', 'set_size'])

    # Helper to get metric for a given condition, set size, and metric type
    def get_metric(condition, set_size, metric_type):
        row = metrics.filter(
            (pl.col('condition').str.to_lowercase() == condition)
            & (pl.col('set_size') == set_size)
        )
        return row[metric_type][0] if row.height > 0 else None

    # Calculate proportion_feedback from block_level_feedback
    def parse_feedback(x):
        if x is None or x == '{}':
            return 0
        try:
            feedback = ast.literal_eval(x)
            return 1 if feedback else 0
        except:
            return 0

    # Filter for test_feedback trials and calculate proportion_feedback
    test_feedback_trials = df.filter(pl.col('trial_id') == 'test_feedback')
    if test_feedback_trials.height > 0:
        feedback_values = test_feedback_trials.select(pl.col('block_level_feedback').map_elements(parse_feedback, return_dtype=pl.Int64)).to_series()
        # Always discount the first feedback entry
        if feedback_values.len() > 1:
            proportion_feedback = feedback_values.tail(-1).mean()
        else:
            proportion_feedback = None
    else:
        proportion_feedback = None

    # List of (condition, set_size) and metrics in the requested order
    combos = [
        ('conjunction', 24),
        ('conjunction', 8),
        ('feature', 24),
        ('feature', 8),
    ]
    metric_types = ['accuracy', 'omission_rate', 'rt']

    # Create metric names using 24/8 convention
    metric_names = [f"{cond}_{size}_{mtype}" for (cond, size) in combos for mtype in metric_types]
    values = [get_metric(cond, size, mtype) for (cond, size) in combos for mtype in metric_types]

    # Append proportion_feedback at the end
    metric_names.append('proportion_feedback_visual_search')
    values.append(proportion_feedback)

    return pl.DataFrame({'metric': metric_names, 'value': values}) 