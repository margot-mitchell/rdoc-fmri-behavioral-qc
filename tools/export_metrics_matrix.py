#!/usr/bin/env python3
"""Export subject/session metrics matrices for specified subjects.

For each task, produce a CSV under results/matrix/ containing all metrics as columns
with rows for subject/session combinations.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Set

RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "matrix")
EXCLUDED_KEYWORDS = {"practice", "prescan", "pretouch"}


def parse_subjects(subjects_raw: str) -> List[str]:
    subjects = [s.strip() for s in re.split(r"[,\s]+", subjects_raw) if s.strip()]
    return subjects


def clean_subject_name(subject: str) -> str:
    return subject[4:] if subject.startswith("sub-") else subject


def load_metrics_for_subject(subject: str) -> Dict[str, List[Dict[str, str]]]:
    subject_dir = os.path.join(RESULTS_DIR, subject)
    if not os.path.isdir(subject_dir):
        print(f"Warning: Subject directory not found: {subject_dir}")
        return {}

    task_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    session_dirs = sorted(
        [d for d in os.listdir(subject_dir) if re.match(r"ses-\d+", d)]
    )

    for session_dir in session_dirs:
        match = re.match(r"ses-(\d+)", session_dir)
        if not match:
            continue
        session_num = match.group(1)
        metrics_path_pattern = os.path.join(
            subject_dir, session_dir, "metrics", "*.csv"
        )
        for metrics_file in glob.glob(metrics_path_pattern):
            basename_lower = os.path.basename(metrics_file).lower()
            if any(keyword in basename_lower for keyword in EXCLUDED_KEYWORDS):
                continue

            task_name = os.path.splitext(os.path.basename(metrics_file))[0]
            if task_name.endswith("_metrics"):
                task_name = task_name[: -len("_metrics")]

            row_data: Dict[str, str] = {"subject": clean_subject_name(subject), "session": session_num}
            try:
                with open(metrics_file, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metric_name = row.get("metric")
                        value = row.get("value")
                        if metric_name:
                            row_data[metric_name] = format_value(value)
            except FileNotFoundError:
                print(f"Warning: Metrics file not found: {metrics_file}")
                continue

            flag_summary = gather_flags_for_task(subject_dir, session_dir, task_name)
            if flag_summary:
                row_data["flags"] = flag_summary

            task_rows[task_name].append(row_data)
    return task_rows


def gather_flags_for_task(subject_dir: str, session_dir: str, task_name: str) -> str:
    flags_dir = os.path.join(subject_dir, session_dir, "flags")
    if not os.path.isdir(flags_dir):
        return ""

    flag_file = os.path.join(flags_dir, f"{task_name}_flags.csv")
    flag_file_lower = flag_file.lower()
    if any(keyword in flag_file_lower for keyword in EXCLUDED_KEYWORDS):
        return ""

    if not os.path.isfile(flag_file):
        return ""

    flags: List[str] = []
    try:
        with open(flag_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get("metric")
                value = format_value(row.get("value"))
                threshold = format_value(row.get("threshold"))
                if metric:
                    if threshold:
                        flags.append(f"{metric}={value} (thr {threshold})")
                    else:
                        flags.append(f"{metric}={value}")
    except FileNotFoundError:
        return ""

    return "; ".join(flags)


def format_value(value: str | None) -> str:
    if value is None or value == "":
        return ""
    try:
        value_float = float(value)
    except ValueError:
        return value

    if value_float.is_integer():
        return str(int(value_float))
    return f"{value_float:.3f}"


def write_task_matrix(task: str, rows: List[Dict[str, str]]):
    if not rows:
        return

    metric_names: Set[str] = set()
    for row in rows:
        metric_names.update(k for k in row.keys() if k not in {"subject", "session"})

    metric_names.discard("flags")

    def extract_subject_num(subject: str) -> int:
        match = re.search(r"(\d+)", subject)
        if match:
            return int(match.group(1))
        return 0

    def sort_key(row: Dict[str, str]):
        subject = row.get("subject", "")
        subject_num = extract_subject_num(subject)
        session_raw = row.get("session", "0")
        try:
            session_num = int(session_raw)
        except ValueError:
            session_num = 0
        return subject_num, subject, session_num

    rows_sorted = sorted(rows, key=sort_key)

    header = ["subject", "session"] + sorted(metric_names)
    if any("flags" in row for row in rows_sorted):
        header.append("flags")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{task}_metrics_matrix.csv")

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows_sorted:
            if "flags" not in row and "flags" in header:
                row = {**row, "flags": ""}
            writer.writerow({key: row.get(key, "") for key in header})

    print(f"Wrote matrix for {task}: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build metrics matrices for specified subjects."
    )
    parser.add_argument(
        "subjects",
        help="Comma-separated list of subjects (e.g., sub-s11,sub-s12)",
    )
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    if not subjects:
        parser.error("At least one subject must be specified")

    combined_task_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for subject in subjects:
        subject_tasks = load_metrics_for_subject(subject)
        for task, rows in subject_tasks.items():
            combined_task_rows[task].extend(rows)

    if not combined_task_rows:
        print("No metrics found for the specified subjects.")
        return

    # Clear existing matrix directory to avoid stale files
    if os.path.isdir(OUTPUT_DIR):
        for existing in glob.glob(os.path.join(OUTPUT_DIR, "*.csv")):
            os.remove(existing)

    for task, rows in combined_task_rows.items():
        write_task_matrix(task, rows)

    print("Metric matrices generated in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
