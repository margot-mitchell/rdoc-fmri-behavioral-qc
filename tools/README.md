# tools/

Development utilities. Nothing here is installed with the package.

| File | Purpose |
|---|---|
| `scan_repo.py` | Identifier scanner. Fails the build if anything that looks like participant data reaches the tracked tree or the history. |
| `pre-commit` | Git hook wrapping the scanner, so a bad commit is stopped before it exists. |
| `export_metrics_matrix.py` | Flattens `results/` into a wide subject × session matrix per task. Requires a populated `results/`, which this repository does not contain. |

## The identifier scanner

`scan_repo.py` is vendored rather than referenced, because CI cannot reach a local
path. Its header records where it came from and when.

It classifies findings by severity. **HIGH** blocks a release: email addresses, phone
numbers, NDA GUIDs, UUIDs, Prolific-format IDs, private keys, hardcoded secret
assignments. **MEDIUM** needs a human decision: millisecond epoch timestamps, absolute
home paths, initials-style subject IDs, dates shaped like a DOB. It also flags files
whose *name* implies identity linkage, regardless of content.

Millisecond timestamps deserve the attention they get. They look innocuous, but they
uniquely fingerprint a session, so anyone holding platform-side submission records can
re-link them to a participant without ever seeing an ID mapping.

Every value it reports is masked, so running the scanner never itself writes an
identifier into a CI log.

```bash
make scan                                  # the tracked tree, i.e. what would be published
python3 tools/scan_repo.py . --history     # every blob on every ref
make hooks                                 # install the pre-commit guard
```

`make hooks` has to be run once per checkout: hooks live in `.git/hooks/`, which is not
version controlled, so a fresh clone starts without one. The hook fails closed — if it
cannot find the scanner it blocks the commit rather than passing silently, because a
guard that quietly stops running is worse than no guard.

False positives go in `.publicscanignore`, one regex per line, **each with a comment
saying why it is safe**. That file is a record of reviewed exceptions, not a mute
button.

Its limits are worth stating: the scanner matches patterns. It cannot recognise a
participant's name in a free-text note, or notice that a small cell count is
re-identifying. It is a floor, not a substitute for reading the diff.
