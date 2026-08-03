"""Cross-platform single-process lock for the local LightRAG data directory."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import BinaryIO

from src.exceptions import RuntimeLockError


_registry_guard = Lock()
_registry: dict[Path, tuple[BinaryIO, int]] = {}


def _lock_file(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeLockError(
                "Another LightGraphRAG server or CLI process is already using this data directory"
            ) from exc
        return

    import fcntl

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeLockError(
            "Another LightGraphRAG server or CLI process is already using this data directory"
        ) from exc


def _unlock_file(stream: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RuntimeLock:
    """A process-wide, reentrant wrapper around an OS-level file lock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._acquired = False

    def acquire(self) -> "RuntimeLock":
        with _registry_guard:
            registered = _registry.get(self.path)
            if registered is not None:
                stream, count = registered
                _registry[self.path] = (stream, count + 1)
                self._acquired = True
                return self

            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = self.path.open("a+b")
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            try:
                _lock_file(stream)
            except Exception:
                stream.close()
                raise
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()).encode("ascii"))
            stream.flush()
            _registry[self.path] = (stream, 1)
            self._acquired = True
            return self

    def release(self) -> None:
        if not self._acquired:
            return
        with _registry_guard:
            registered = _registry.get(self.path)
            if registered is None:
                self._acquired = False
                return
            stream, count = registered
            if count > 1:
                _registry[self.path] = (stream, count - 1)
            else:
                try:
                    _unlock_file(stream)
                finally:
                    stream.close()
                    _registry.pop(self.path, None)
            self._acquired = False

    def __enter__(self) -> "RuntimeLock":
        return self.acquire()

    def __exit__(self, *_args: object) -> None:
        self.release()
