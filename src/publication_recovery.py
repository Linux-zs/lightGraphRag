"""Idempotent local publication rollback; callers supply validated owned paths."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    active: Path
    candidate: Path
    backup: Path
    had_active: bool
    had_candidate: bool


def rollback_artifacts(artifacts: list[Artifact]) -> None:
    """Preflight the entire generation before moving anything; never delete data.

    Caller must hold the workspace writer lock and validate path ownership.
    Existence patterns also accept already-restored artifacts, so restarting
    after any individual rename is safe. Ambiguous states fail closed.
    """
    paths = [path.resolve() for item in artifacts
             for path in (item.active, item.candidate, item.backup)]
    if len(set(paths)) != len(paths) or any(
        left in right.parents for left in paths for right in paths if left != right
    ):
        raise RuntimeError("Overlapping publication recovery paths")
    actions: list[tuple[Path, Path]] = []
    for item in artifacts:
        a, c, b = item.active.exists(), item.candidate.exists(), item.backup.exists()
        if not item.had_candidate:
            if (a, c, b) != (item.had_active, False, False):
                raise RuntimeError("Unexpected untouched publication artifact")
            continue
        if item.had_active:
            if (a, c, b) == (True, True, False):
                continue  # Not moved, or already restored.
            if (a, c, b) == (True, False, True):
                actions.append((item.active, item.candidate))
            elif (a, c, b) != (False, True, True):
                raise RuntimeError("Ambiguous publication artifact; manual recovery required")
            actions.append((item.backup, item.active))
        else:
            if (a, c, b) == (False, True, False):
                continue
            if (a, c, b) != (True, False, False):
                raise RuntimeError("Ambiguous new publication artifact")
            actions.append((item.active, item.candidate))
    for source, destination in actions:
        if destination.exists() or not source.exists():
            raise RuntimeError("Publication files changed during recovery")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, destination)
