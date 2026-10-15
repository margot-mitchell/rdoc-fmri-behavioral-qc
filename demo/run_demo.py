#!/usr/bin/env python3
"""Run the full QC pipeline on synthetic data and print the resulting report.

No participant data is involved. The synthetic flanker run comes from the same
generator the test suite uses (tests/conftest.py::build_flanker_raw_json), which
is engineered so the expected numbers are known in advance:

    congruent:   8 correct + 1 incorrect + 1 omission -> accuracy 0.889, omission 0.10
    incongruent: 5 correct + 3 incorrect + 2 omissions -> accuracy 0.625, omission 0.20
    feedback:    3 entries, first discarded, 1 of the remaining 2 non-empty -> 0.5

Those values straddle the thresholds in rdocfmriqc/thresholds.py, so the run is
expected to raise flags rather than pass cleanly. That is the point: a demo where
nothing fails would not show what the QC actually does.

    python3 demo/run_demo.py        (or: make demo)
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SANDBOX = REPO_ROOT / 'demo' / '_sandbox'
SUBJECT = 'sub-demo'
SESSION = 'ses-1'

sys.path.insert(0, str(REPO_ROOT / 'tests'))
from conftest import build_flanker_raw_json  # noqa: E402


def rule(title):
    print(f"\n\033[1m{'=' * 72}\n{title}\n{'=' * 72}\033[0m")


def build_sandbox():
    """Lay out the tree the three CLI stages expect, exactly as the tests do."""
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    raw = SANDBOX / 'output' / 'raw' / SUBJECT / SESSION
    raw.mkdir(parents=True)
    (SANDBOX / 'preprocessed_data').mkdir()

    # A trailing component after run-N matters: the run parser does not strip the
    # .json extension, so `..._run-1.json` alone would yield run_num="1.json".
    name = f'{SUBJECT}_{SESSION}_task-flanker_rdoc_run-1_dateTime-20250101.json'
    (raw / name).write_text(json.dumps(build_flanker_raw_json()))
    return raw / name


def run_stage(module):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, '-m', module, SUBJECT, '--session', SESSION],
        cwd=SANDBOX, env=env, capture_output=True, text=True,
    )
    status = 'ok' if proc.returncode == 0 else f'FAILED ({proc.returncode})'
    print(f"  {module:45s} {status}")
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, sep='\n')
        sys.exit(1)


def show_csvs(kind):
    found = sorted((SANDBOX / 'results' / SUBJECT / SESSION / kind).glob('*.csv'))
    if not found:
        print(f"  (no {kind} written)")
        return found
    for f in found:
        print(f"\n  \033[1m{f.relative_to(SANDBOX)}\033[0m")
        for line in f.read_text().splitlines():
            print(f"    {line}")
    return found


def build_compiled_results(flag_files):
    """Recreate the artifact layout generate_email_body.sh consumes in CI.

    The script walks compiled-results/<name>_results/flags/, which the real
    workflow assembles by downloading one artifact per processed session.
    """
    target = SANDBOX / 'compiled-results' / f'{SUBJECT}_{SESSION}_results' / 'flags'
    target.mkdir(parents=True, exist_ok=True)
    for f in flag_files:
        shutil.copy(f, target / f.name)


def main():
    rule('1. Synthetic input')
    raw_file = build_sandbox()
    trials = json.loads(raw_file.read_text())['trial_data']
    print(f"  {raw_file.relative_to(SANDBOX)}")
    print(f"  {len(trials)} trials, generated in-memory. No participant data.")

    rule('2. Pipeline')
    for stage in ('preprocess', 'calculate_metrics', 'generate_flags'):
        run_stage(f'rdocfmriqc.analyze.{stage}')

    rule('3. Metrics')
    show_csvs('metrics')

    rule('4. Flags')
    flag_files = show_csvs('flags')

    rule('5. QC report email')
    if not flag_files:
        print("  No flags raised, so there is nothing to report.")
        return
    build_compiled_results(flag_files)
    env = dict(
        os.environ,
        GITHUB_EVENT_NAME='workflow_dispatch',
        GITHUB_RUN_ID='0',
        GITHUB_REPOSITORY='margot-mitchell/rdoc-fmri-behavioral-qc',
        GITHUB_SERVER_URL='https://github.com',
        SUBJECTS_PROCESSED=SUBJECT,
        SESSIONS_PROCESSED=f'{SUBJECT}/{SESSION}',
    )
    proc = subprocess.run(
        ['bash', str(REPO_ROOT / 'generate_email_body.sh')],
        cwd=SANDBOX, env=env, capture_output=True, text=True,
    )
    body = SANDBOX / 'email_body.txt'
    if proc.returncode != 0 or not body.exists():
        print(proc.stdout, proc.stderr, sep='\n')
        sys.exit(1)
    for line in body.read_text().splitlines():
        print(f"  {line}")

    print(f"\n  Sandbox left at {SANDBOX.relative_to(REPO_ROOT)} for inspection."
          f"  Remove with: make clean")


if __name__ == '__main__':
    main()
