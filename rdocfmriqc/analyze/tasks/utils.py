"""Utility functions for task analysis."""

from typing import List, Optional
import polars as pl

def omission_expr(df: pl.DataFrame) -> pl.Expr:
    """Expression that is True for trials where the participant did not respond.

    A trial is an omission when `rt` is null, `response` is null, or `response`
    is -1 (or empty). The flanker, Stroop and AX-CPT experiments write
    `response = -1` when no key is pressed, so a null-response test alone misses
    every omission in those tasks. This is the single omission rule for all
    tasks; longest_omission._row_omission applies the same rule row by row.

    Args:
        df (pl.DataFrame): Frame the expression will be evaluated on. Only its
            schema is used, to compare `response` against -1 with the right type.

    Returns:
        pl.Expr: Boolean expression.
    """
    omitted = pl.col('rt').is_null() if 'rt' in df.columns else pl.lit(False)
    if 'response' in df.columns:
        omitted = omitted | pl.col('response').is_null()
        dtype = df.schema['response']
        if dtype == pl.String:
            omitted = omitted | pl.col('response').is_in(['-1', ''])
        elif dtype.is_numeric():
            omitted = omitted | (pl.col('response') == -1)
    return omitted

def get_metrics(df: pl.DataFrame, group_by: Optional[List[str]] = None) -> pl.DataFrame:
    """Calculate basic metrics from task data.

    Args:
        df (pl.DataFrame): Input dataframe containing task data.
        group_by (Optional[List[str]]): List of column names to group by.

    Returns:
        pl.DataFrame: DataFrame with calculated metrics.
    """
    omitted = omission_expr(df)

    # Calculate accuracy using correct_trial (only for trials with responses)
    accuracy = df.group_by(group_by).agg(
        pl.col('correct_trial').filter(~omitted).mean().alias('accuracy')
    )

    # Calculate omission rate (trials where no response was given)
    omission_rate = df.group_by(group_by).agg(
        omitted.mean().alias('omission_rate')
    )

    # Calculate reaction time (only for trials with valid responses)
    rt = df.group_by(group_by).agg(
        pl.col('rt').filter(~omitted).mean().alias('rt')
    )

    # Join all metrics
    if group_by:
        return accuracy.join(omission_rate, on=group_by).join(rt, on=group_by)
    else:
        # When not grouping, we can just concatenate the columns
        return pl.concat([
            accuracy.select('accuracy'),
            omission_rate.select('omission_rate'),
            rt.select('rt')
        ], how='horizontal')

def organize_metrics(
    metrics: pl.DataFrame, group_by: Optional[List[str]] = None
) -> pl.DataFrame:
    """Organize metrics into a long format with 'metric' and 'value' columns.

    Args:
        metrics (pl.DataFrame): DataFrame containing metrics.
        group_by (Optional[List[str]]): List of column names that were used for grouping.

    Returns:
        pl.DataFrame: DataFrame with 'metric' and 'value' columns.
    """
    def create_metric_column(group_by_cols):
        if not group_by_cols:
            return pl.col('metric')
        
        # Create metric name by concatenating group values
        metric_parts = []
        for col in group_by_cols:
            metric_parts.append(pl.col(col).cast(pl.Utf8))
        
        return pl.concat_str(metric_parts, separator='_')

    # Reshape to metric/value columns. `unpivot` replaces `melt`, which polars
    # removed outright in 2.0; `unpivot` exists from 1.x onward, so this works
    # on both sides of that break.
    melted = metrics.unpivot(
        index=group_by if group_by else [],
        on=['accuracy', 'omission_rate', 'rt'],
        variable_name='metric_type',
        value_name='value'
    )

    # Create the metric column
    if group_by:
        melted = melted.with_columns(
            metric=create_metric_column(group_by) + pl.lit('_') + pl.col('metric_type')
        )
    else:
        melted = melted.with_columns(
            metric=pl.col('metric_type')
        )

    # Drop the group_by columns and metric_type column
    if group_by:
        melted = melted.drop(group_by)
    melted = melted.drop('metric_type')

    return melted 