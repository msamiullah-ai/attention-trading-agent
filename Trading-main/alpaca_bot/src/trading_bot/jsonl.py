"""Append-only JSONL storage, in one place.

Three modules were each carrying their own copy of this -- `options_trader`
(cycle reports), `learning` (strategy outcomes) and `advisor_memory` (LLM
verdicts). The bodies had drifted apart in ways that mattered: one caught
`json.JSONDecodeError` only, one also caught `ValueError` and `TypeError`, one
checked `path.exists()` and the others relied on the open failing. Same
intent, three behaviours, three places to fix anything.

Everything here is deliberately failure-tolerant in the same direction:

  reading   a torn final line is skipped, not raised. A process killed
            mid-write leaves half a line, and one bad row must not cost the
            whole history.
  writing   an unwritable path logs and returns. These files are bookkeeping;
            losing a row degrades a report, while raising would end a trading
            cycle over it.

That is the right trade for a decision log and the wrong one for an order, so
nothing in the order path uses this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .logger import get_logger

log = get_logger(__name__)


def read(path: Path | str, limit: int | None = None) -> list[dict]:
    """Every parseable row, oldest first. Missing file reads as empty.

    `limit` keeps the most RECENT rows rather than the first, because every
    caller wants the tail -- a decision log is read to see what just happened.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue          # torn final line from a killed process
                if isinstance(row, dict):
                    rows.append(row)
    except OSError as exc:
        log.warning("could not read %s: %s", p.name, exc)
        return []
    return rows[-limit:] if limit else rows


def append(path: Path | str, row: dict) -> bool:
    """Add one row. Returns False if it could not be written.

    The return value is not decoration: a caller that keeps an in-memory copy
    needs to know the two have diverged, and silently succeeding would make a
    lost row indistinguishable from a written one.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return True
    except (OSError, TypeError) as exc:
        log.warning("could not append to %s: %s", p.name, exc)
        return False


def rewrite(path: Path | str, rows: Iterable[dict], keep: int | None = None) -> bool:
    """Replace the file's contents atomically.

    Needed by any caller that MUTATES rows rather than only adding them --
    advisor verdicts get scored after the fact, so append-only cannot express
    them. Written to a sibling temp file and renamed, so a crash mid-write
    leaves the previous file intact rather than a truncated one.
    """
    p = Path(path)
    rows = list(rows)
    if keep:
        rows = rows[-keep:]
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        tmp.replace(p)
        return True
    except (OSError, TypeError) as exc:
        log.warning("could not rewrite %s: %s", p.name, exc)
        return False


def state_path(name: str) -> Path:
    """A file under `state/`, resolved the same way everywhere.

    Every caller was building this from PROJECT_ROOT with its own local import,
    which is how two of them ended up importing `Path` a second time inside a
    function body.
    """
    from .config import PROJECT_ROOT
    return PROJECT_ROOT / "state" / name


def coerce(value: Any, kind: type, default: Any):
    """Read a field from an untrusted row without raising.

    Rows come off disk and may predate a schema change, so a missing or
    wrong-typed field has to degrade to a default rather than kill the load.
    """
    try:
        return kind(value) if value is not None else default
    except (TypeError, ValueError):
        return default
