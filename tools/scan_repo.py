#!/usr/bin/env python3
# Vendored from rdocmetrics/tools/scan_repo.py on 2026-10-15.
#
# Vendored rather than referenced: this repository is public and its CI cannot
# reach a local path. Taken from rdocmetrics rather than from
# public-release-tooling, because that upstream copy still lets --plain win
# silently over --history, so `--history --plain` scans the working tree and
# reports a clean history it never read. Re-vendor from rdocmetrics if the
# scanner changes.
"""
Identifier / secret scanner for pre-public-release review.

Usage:
    ./scan_repo.py PATH                 # scan tracked files in working tree
    ./scan_repo.py PATH --history       # scan EVERY blob in git history
    ./scan_repo.py PATH --plain         # scan a plain directory (no git)

Exit codes:
    0  no HIGH findings
    1  at least one HIGH finding  (use this to gate a release)

Values are masked in output so running this never itself leaks identifiers
into a terminal log or CI transcript.

False positives are expected on 24-hex (git SHAs, dependency hashes) and on
epoch-ms (any large integer). Both are reported for human review, not
auto-trusted. Add regexes to .publicscanignore to silence known-good hits.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- patterns

HIGH = "HIGH"
MED = "MEDIUM"
LOW = "LOW"

PATTERNS = [
    # --- direct identifiers: these block a release outright
    (HIGH, "email", re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (HIGH, "phone", re.compile(
        r"(?<!\d)\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")),
    (HIGH, "nda_guid", re.compile(r"NDAR[A-Z0-9]{8,}")),
    (HIGH, "uuid", re.compile(
        r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
        r"-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])")),
    # Prolific participant IDs. Stable across studies -> personal data.
    (HIGH, "prolific_id_24hex", re.compile(
        r"(?<![0-9a-fA-F])[0-9a-f]{24}(?![0-9a-fA-F])")),
    (HIGH, "private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (HIGH, "secret_assignment", re.compile(
        r"(?i)\b(api[_-]?key|secret|passwd|password|access[_-]?token|"
        r"auth[_-]?token|bearer)\b\s*[=:]\s*['\"][^'\"]{8,}['\"]")),

    # --- quasi-identifiers: judgement call, but must be seen before release
    # Millisecond epoch timestamps uniquely fingerprint a session and can be
    # re-linked against platform-side submission records.
    (MED, "epoch_ms_timestamp", re.compile(r"(?<!\d)1[6-8]\d{11}(?!\d)")),
    (MED, "absolute_home_path", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")),
    # Initials-as-subject-ID, e.g. sub-SK
    (MED, "initials_subject_id", re.compile(r"\bsub-[A-Z]{2,3}(?![a-z0-9])")),
    (MED, "dob_like_date", re.compile(
        r"(?<!\d)(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-]"
        r"(?:19[3-9]\d|200\d)(?!\d)")),
]

# Filenames that should never appear in a public release.
JUNK_NAMES = re.compile(
    r"(?:^|/)(?:\.DS_Store|\.Rhistory|\.RData|Thumbs\.db|\.env)$"
    r"|\.pyc$|\.knit\.md$")

# Filename shapes that suggest the file's whole purpose is linking identities.
DANGER_NAMES = re.compile(
    r"(?i)(?:^|/)[^/]*(?:pii|guid|id_mapping|subject_mapping|worker_id|"
    r"prolific_mapping|config_mapping|participant_tracking|roster|consent|"
    r"contact)[^/]*$")

# Built-in benign values: documentation placeholders and well-known bot
# identities that carry no personal information.
BENIGN = re.compile(
    r"(?i)(example\.com|example\.org|placeholder|your[_-]?email|"
    r"user@host|foo@bar|xxxxx|<redacted>|\bTODO\b|"
    r"action@github\.com|noreply@github\.com|"
    r"\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com)")

BINARY_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".rds", ".RData", ".psydat",
    ".zip", ".gz", ".so", ".pyc", ".nii", ".mat", ".xlsx", ".ico", ".woff",
    ".woff2", ".ttf", ".parquet", ".feather", ".h5", ".pt", ".pth",
}


def mask(value: str) -> str:
    """Show enough to locate the hit, never enough to leak it."""
    if len(value) <= 6:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def load_ignores(root: Path):
    f = root / ".publicscanignore"
    pats = []
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                pats.append(re.compile(line))
    return pats


def scan_text(path_label: str, text: str, ignores):
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if len(line) > 20000:            # pathological single-line data blobs
            line = line[:20000]
        for sev, name, rx in PATTERNS:
            for m in rx.finditer(line):
                val = m.group(0)
                if BENIGN.search(val) or any(p.search(val) for p in ignores):
                    continue
                # keep a hash of the raw value so distinct-counting is exact
                # without ever retaining the value itself
                findings.append((sev, name, path_label, lineno, mask(val),
                                 hash(val)))
    return findings


def git(root: Path, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True).stdout


def iter_tracked(root: Path):
    out = git(root, "ls-files", "-z")
    for rel in filter(None, out.split("\0")):
        p = root / rel
        if p.is_file():
            yield rel, p.read_bytes()


def iter_history(root: Path):
    """Every blob ever committed, on any ref."""
    out = git(root, "rev-list", "--all", "--objects")
    seen = set()
    for line in out.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        sha, rel = parts
        if sha in seen:
            continue
        seen.add(sha)
        t = subprocess.run(["git", "-C", str(root), "cat-file", "-t", sha],
                           capture_output=True, text=True).stdout.strip()
        if t != "blob":
            continue
        blob = subprocess.run(["git", "-C", str(root), "cat-file", "-p", sha],
                              capture_output=True).stdout
        yield f"{rel} @{sha[:8]}", blob


def iter_plain(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git/" not in str(p):
            yield str(p.relative_to(root)), p.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    # Mutually exclusive, and enforced rather than documented: --plain used to
    # win silently, so `--history --plain` scanned the working tree and reported
    # a clean history it had never looked at. A scan that quietly checks less
    # than it claims is worse than no scan.
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--history", action="store_true",
                      help="scan every blob in git history, not just HEAD")
    mode.add_argument("--plain", action="store_true",
                      help="scan a plain directory, no git required")
    ap.add_argument("--quiet", action="store_true",
                    help="summary only")
    args = ap.parse_args()

    root = Path(args.path).resolve()
    ignores = load_ignores(root)

    if args.plain:
        source, label = iter_plain(root), "plain directory"
    elif args.history:
        source, label = iter_history(root), "FULL GIT HISTORY"
    else:
        source, label = iter_tracked(root), "tracked files at HEAD"

    print(f"Scanning {root}  [{label}]")

    findings, junk, danger, nfiles = [], [], [], 0
    for rel, raw in source:
        nfiles += 1
        clean = rel.split(" @")[0]
        if JUNK_NAMES.search(clean):
            junk.append(rel)
        if DANGER_NAMES.search(clean):
            danger.append(rel)
        if Path(clean).suffix in BINARY_EXT or b"\0" in raw[:4096]:
            continue
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        findings.extend(scan_text(rel, text, ignores))

    by_sev = {HIGH: [], MED: [], LOW: []}
    for f in findings:
        by_sev[f[0]].append(f)

    if danger:
        print(f"\n!! {len(danger)} file(s) whose NAME implies identity linkage:")
        for d in danger[:20]:
            print(f"     {d}")

    for sev in (HIGH, MED):
        group = by_sev[sev]
        if not group:
            continue
        print(f"\n{sev}: {len(group)} hit(s)")
        counts = {}
        for _, name, path, line, val, vhash in group:
            counts.setdefault(name, []).append((path, line, val, vhash))
        for name, hits in sorted(counts.items(),
                                 key=lambda kv: -len(kv[1])):
            uniq = len({h[3] for h in hits})
            print(f"  [{name}] {len(hits)} hit(s), {uniq} distinct value(s)")
            if not args.quiet:
                for path, line, val, _ in hits[:5]:
                    print(f"      {path}:{line}  {val}")
                if len(hits) > 5:
                    print(f"      ... {len(hits) - 5} more")

    if junk:
        print(f"\nLOW: {len(junk)} junk/hygiene file(s) "
              f"(should not ship): {junk[:8]}")

    n_high = len(by_sev[HIGH])
    print(f"\n{'=' * 60}")
    print(f"scanned {nfiles} objects | HIGH={n_high} "
          f"MEDIUM={len(by_sev[MED])} junk={len(junk)} "
          f"suspicious-names={len(danger)}")
    if n_high:
        print("RESULT: BLOCKED — resolve HIGH findings before publishing.")
    else:
        print("RESULT: no HIGH findings. Review MEDIUM by hand before release.")
    print("=" * 60)
    return 1 if n_high else 0


if __name__ == "__main__":
    sys.exit(main())
