"""Longest run of sequential incorrect omission trials (per block), for QC metrics."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import polars as pl


def _is_pretouch_or_practice(filename: Optional[str]) -> bool:
    if not filename:
        return False
    fn = filename.lower()
    return "practice" in fn or "pretouch" in fn


def _normalize_task_dir(task_dir: str) -> str:
    s = task_dir.lower().replace("-", "_")
    for suf in ("_rdoc", "_fmri", "__fmri", "_practice"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def _task_skip_nogo_stop(task_key: str) -> tuple[bool, bool]:
    """Return (skip_nogo, skip_stop) for bridging trials."""
    skip_nogo = "gonogo" in task_key or "go_nogo" in task_key
    skip_stop = "stopsignal" in task_key or "stop_signal" in task_key
    return skip_nogo, skip_stop


def _select_trials(df: pl.DataFrame, task_dir: str, filename: Optional[str]) -> pl.DataFrame:
    """Mirror each task's primary trial stream used for choice/RT QC."""
    is_pre = _is_pretouch_or_practice(filename)
    task_key = _normalize_task_dir(task_dir)

    if "trial_id" not in df.columns:
        return df

    trial_ids = df.select(pl.col("trial_id").unique()).to_series().to_list()

    if is_pre:
        return df

    # AX-CPT: test_trial or test_probe
    if "ax_cpt" in task_key or "axcpt" in task_key:
        if "test_trial" in trial_ids:
            return df.filter(pl.col("trial_id") == "test_trial")
        return df.filter(pl.col("trial_id") == "test_probe")

    # Operation-only span: processing stream (same omission definition as op_only_span_rdoc)
    if "op_only" in task_key or "oponly" in task_key:
        processing = df.filter(pl.col("trial_id") == "test_inter-stimulus")
        if processing.height == 0:
            return processing
        trials_to_exclude: list[Any] = []
        for idx in processing["trial_index"].to_list():
            row_before = df.filter(pl.col("trial_index") == idx - 1)
            row_after = df.filter(pl.col("trial_index") == idx + 1)
            if row_before.height > 0 and row_before["trial_id"][0] == "test_inter-stimulus":
                if row_after.height > 0 and row_after["trial_id"][0] != "test_inter-stimulus":
                    trials_to_exclude.append(idx)
                elif row_after.height == 0:
                    trials_to_exclude.append(idx)
        return processing.filter(~pl.col("trial_index").is_in(trials_to_exclude))

    # Default RDOC choice tasks
    if "test_trial" in trial_ids:
        return df.filter(pl.col("trial_id") == "test_trial")

    return df


def _row_omission(row: dict[str, Any]) -> bool:
    """Omission: no usable RT, or a null, empty or -1 response.

    Row-wise form of utils.omission_expr, so this metric and the omission rates
    agree on which trials are omissions.
    """
    rt = row.get("rt")
    if isinstance(rt, float) and math.isnan(rt):
        rt = None
    if "response" not in row:
        return rt is None
    if "rt" in row and rt is None:
        return True
    resp = row.get("response")
    return resp is None or resp in ("", "-1", -1)


def _row_incorrect(row: dict[str, Any]) -> bool:
    ct = row.get("correct_trial")
    if ct is None:
        return False
    if isinstance(ct, (bool,)):
        return not ct
    try:
        return float(ct) == 0.0
    except (TypeError, ValueError):
        return not bool(ct)


def _condition_skip(row: dict[str, Any], skip_nogo: bool, skip_stop: bool) -> bool:
    cond = row.get("condition")
    if cond is None:
        return False
    cl = str(cond).lower().strip()
    if skip_nogo and cl == "nogo":
        return True
    if skip_stop and cl == "stop":
        return True
    return False


def compute_longest_incorrect_omission_streak(
    trials: pl.DataFrame, task_dir: str, filename: Optional[str] = None
) -> Optional[int]:
    """Max run length of incorrect omissions within a block; nogo/stop trials are skipped (bridge)."""
    task_key = _normalize_task_dir(task_dir)
    skip_nogo, skip_stop = _task_skip_nogo_stop(task_key)

    if trials.height == 0:
        return 0

    if "correct_trial" not in trials.columns:
        logging.warning(
            "[longest_omission] correct_trial missing for %s; skipping metric", task_dir
        )
        return None

    sort_cols: list[str] = []
    if "trial_index" in trials.columns:
        sort_cols.append("trial_index")
    trials_sorted = trials.sort(sort_cols) if sort_cols else trials

    use_cols = ["correct_trial", "condition", "rt"]
    if "response" in trials_sorted.columns:
        use_cols.append("response")
    if "block_num" in trials_sorted.columns:
        use_cols.append("block_num")

    rows = trials_sorted.select([c for c in use_cols if c in trials_sorted.columns]).to_dicts()

    best = 0
    streak = 0
    prev_block: Any = None

    for row in rows:
        bn = row.get("block_num")
        if prev_block is not None and bn is not None and bn != prev_block:
            streak = 0
        if bn is not None:
            prev_block = bn

        if _condition_skip(row, skip_nogo, skip_stop):
            continue

        if _row_omission(row) and _row_incorrect(row):
            streak += 1
            if streak > best:
                best = streak
        else:
            streak = 0

    return int(best)


def append_longest_omission_metric(
    metrics: pl.DataFrame,
    df: pl.DataFrame,
    task_dir: str,
    filename: Optional[str] = None,
) -> pl.DataFrame:
    """Append n_trials_in_longest_omission_sequence row to a task metrics frame."""
    try:
        trials = _select_trials(df, task_dir, filename)
        n = compute_longest_incorrect_omission_streak(trials, task_dir, filename)
    except Exception as e:
        logging.warning("[longest_omission] failed for %s: %s", task_dir, e)
        return metrics

    if n is None:
        return metrics

    extra = pl.DataFrame(
        {
            "metric": ["n_trials_in_longest_omission_sequence"],
            "value": [float(n)],
        }
    )
    return pl.concat([metrics, extra], how="vertical")
