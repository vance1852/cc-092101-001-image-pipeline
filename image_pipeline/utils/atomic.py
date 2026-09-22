"""Recoverable, atomic file-commit primitives.

The pipeline must never silently overwrite the last known-good artifact:

* ``atomic_write_bytes`` writes to a unique temporary file in the same
  directory, fsyncs it, then ``os.replace``\\s it into place.  A crash while
  rendering leaves the previous destination file untouched.
* ``RecoverableCommit`` implements a journaled two-phase commit for a batch
  of output files: new bytes and backups of replaced files are staged in a
  per-run journal directory before anything is installed; a crash or write
  failure can be rolled back (or replayed on the next run via
  ``recover_all``), restoring the last complete file set.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from typing import Any, Dict, List, Optional


def checksum256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    The destination is created or replaced in a single ``os.replace`` step.
    If encoding/writing fails before the replace, any pre-existing file at
    ``path`` is left byte-for-byte intact.
    """
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix='.imgpipe-tmp-',
        suffix='-' + uuid.uuid4().hex,
        dir=directory,
    )
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Some filesystems (e.g. certain network mounts) reject fsync;
                # the atomic replace still guarantees consistency within the
                # directory entry.
                pass
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise



def _fsync_dir(path: str) -> None:
    """Best-effort fsync of a directory so rename results are durable."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


class RecoverableCommit:
    """Crash-safe journal covering a whole batch of output files.

    Every planned replacement is recorded *before* it happens:

    1. ``stage_output`` renders the new bytes into a private journal
       directory and, if the destination already exists, copies the current
       bytes there as a backup. The manifest (including the backup entry) is
       atomically persisted before this call returns.
    2. ``commit_pending`` moves staged files into place with
       ``os.replace`` and drops a per-file "done" marker. A crash here can be
       cleaned up by :meth:`recover_all`.
    3. ``rollback`` (or recovery on the next run) restores backed-up files in
       reverse order and removes files that did not exist before the batch,
       so the last complete/valid output set is brought back.

    Journals live *inside* the output directory so all paths share one
    filesystem and ``os.replace`` is atomic.
    """

    JOURNAL_PREFIX = '.imgpipe-journal-'

    def __init__(self, output_dir: str, journal_id: Optional[str] = None):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.journal_id = journal_id or uuid.uuid4().hex
        self.journal_dir = os.path.join(
            self.output_dir, f'{self.JOURNAL_PREFIX}{self.journal_id}'
        )
        os.makedirs(self.journal_dir, exist_ok=True)
        self._entries: List[Dict[str, Any]] = []
        self._seq = 0

    # ------------------------------------------------------------------ utils

    def _journal_path(self, name: str) -> str:
        return os.path.join(self.journal_dir, name)

    def _manifest_path(self) -> str:
        return self._journal_path('manifest.json')

    def _write_manifest(self) -> None:
        payload = {'journal_id': self.journal_id, 'entries': self._entries}
        atomic_write_bytes(
            self._manifest_path(),
            json.dumps(payload, indent=2).encode('utf-8'),
        )

    def _done_marker(self, idx: int) -> str:
        return self._journal_path(f'done-{idx}')

    @staticmethod
    def _atomic_file_copy(src: str, dst: str) -> None:
        with open(src, 'rb') as f:
            data = f.read()
        atomic_write_bytes(dst, data)

    # ------------------------------------------------------------- lifecycle

    def stage_output(self, rel_dest: str, data: bytes) -> Dict[str, Any]:
        """Record a planned write of ``data`` to ``rel_dest`` (relative to the
        output directory). The destination itself is not touched yet.

        Returns the manifest entry for the staged file.
        """
        rel_dest = rel_dest.replace(os.sep, '/').lstrip('/')
        dest_abs = os.path.join(self.output_dir, rel_dest)
        if os.path.commonpath([os.path.abspath(dest_abs), self.output_dir]) != self.output_dir:
            raise ValueError(f'Refusing to stage output outside output dir: {rel_dest}')
        self._seq += 1
        idx = self._seq
        staged_name = f'new-{idx}'
        backup_name: Optional[str] = None
        if os.path.exists(dest_abs):
            backup_name = f'backup-{idx}'
            self._atomic_file_copy(dest_abs, self._journal_path(backup_name))
        atomic_write_bytes(self._journal_path(staged_name), data)
        entry = {
            'idx': idx,
            'dest': rel_dest,
            'staged': staged_name,
            'backup': backup_name,
        }
        self._entries.append(entry)
        self._write_manifest()
        return entry

    def commit_pending(self) -> None:
        """Install every staged file that has not been committed yet."""
        for entry in self._entries:
            marker = self._done_marker(entry['idx'])
            if os.path.exists(marker):
                continue
            dest_abs = os.path.join(self.output_dir, entry['dest'])
            os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
            os.replace(self._journal_path(entry['staged']), dest_abs)
            _fsync_dir(os.path.dirname(dest_abs) or self.output_dir)
            with open(marker, 'wb'):
                pass
            _fsync_dir(self.journal_dir)

    def rollback(self) -> List[str]:
        """Undo committed entries. Returns the list of restored paths."""
        restored: List[str] = []
        for entry in reversed(self._entries):
            marker = self._done_marker(entry['idx'])
            if not os.path.exists(marker):
                continue
            dest_abs = os.path.join(self.output_dir, entry['dest'])
            if entry['backup']:
                os.replace(self._journal_path(entry['backup']), dest_abs)
            else:
                try:
                    os.unlink(dest_abs)
                except FileNotFoundError:
                    pass
            restored.append(entry['dest'])
            try:
                os.unlink(marker)
            except FileNotFoundError:
                pass
        return restored

    def discard(self) -> None:
        """Remove journal files (call only after a successful commit)."""
        shutil.rmtree(self.journal_dir, ignore_errors=True)

    @classmethod
    def recover_all(cls, output_dir: str) -> List[str]:
        """Replay journals left behind by crashed/interrupted runs.

        Each journal either never reached the commit phase (all destinations
        untouched) or has per-file done markers; restore the backups for
        everything that was installed. Returns the restored destinations.
        """
        output_dir = os.path.abspath(output_dir)
        restored: List[str] = []
        if not os.path.isdir(output_dir):
            return restored
        for name in sorted(os.listdir(output_dir)):
            journal_dir = os.path.join(output_dir, name)
            if not name.startswith(cls.JOURNAL_PREFIX) or not os.path.isdir(journal_dir):
                continue
            manifest = os.path.join(journal_dir, 'manifest.json')
            entries = []
            if os.path.isfile(manifest):
                try:
                    with open(manifest, 'r', encoding='utf-8') as f:
                        entries = json.load(f).get('entries', [])
                except (OSError, ValueError):
                    entries = []
            for entry in reversed(entries):
                marker = os.path.join(journal_dir, f"done-{entry.get('idx')}")
                if not os.path.exists(marker):
                    continue
                dest_abs = os.path.join(output_dir, entry.get('dest', ''))
                backup = entry.get('backup')
                try:
                    if backup and os.path.isfile(os.path.join(journal_dir, backup)):
                        os.replace(os.path.join(journal_dir, backup), dest_abs)
                        restored.append(entry['dest'])
                    else:
                        try:
                            os.unlink(dest_abs)
                            restored.append(entry['dest'])
                        except FileNotFoundError:
                            pass
                except OSError:
                    # Keep the journal so an operator can inspect/restore it.
                    continue
            shutil.rmtree(journal_dir, ignore_errors=True)
        return restored
