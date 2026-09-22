"""Recoverable atomic commits for multi-output batches.

Each input image is processed entirely in memory/on staging; nothing appears
at its final output path until :meth:`OutputTransaction.commit`. The commit is
group-atomic over every output node of one image:

* pre-existing target files are moved to backup paths first and recorded in a
  journal;
* all staged files are installed with ``os.replace`` (each replacement itself
  atomic and crash-safe);
* on any failure (or a crash discovered on the next run), backups are moved
  back and newly-created targets are removed, so a failed batch can never
  leave half a set of products nor overwrite the previous valid files.
"""
from typing import Dict, List, Tuple
import glob
import json
import os
import tempfile
import uuid

from .image_io import write_bytes_atomic

JOURNAL_DIRNAME = '.imgpipe-journal'
STAGING_DIRNAME = '.imgpipe-staging'
BACKUP_SUFFIX = '.imgpipe-bak'


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


class OutputTransaction:
    """Stages output blobs and commits them as a group."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.txn_id = uuid.uuid4().hex[:12]
        self._journal_dir = os.path.join(output_dir, JOURNAL_DIRNAME)
        self._staging_dir = os.path.join(output_dir, STAGING_DIRNAME, self.txn_id)
        self._journal_path = os.path.join(self._journal_dir, f'txn-{self.txn_id}.json')
        # final_path -> (staged_path, size_in_bytes)
        self._staged: Dict[str, Tuple[str, int]] = {}
        self._finished = False

    @property
    def journal_path(self) -> str:
        return self._journal_path

    @property
    def staged_paths(self) -> List[str]:
        return [p for p, _ in self._staged.values()]

    @property
    def has_staged(self) -> bool:
        return bool(self._staged)

    def stage(self, final_path: str, data: bytes) -> int:
        """Write a blob to staging; final_path is not touched yet."""
        norm = os.path.abspath(final_path)
        if norm in self._staged:
            raise RuntimeError(f"Output path staged twice in one transaction: '{norm}'")
        if not os.path.isdir(self._staging_dir):
            os.makedirs(self._staging_dir, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix='stage-', dir=self._staging_dir)
        staged_path = os.path.join(self._staging_dir, f'out-{len(self._staged):03d}-' + os.path.basename(norm))
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, staged_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._staged[norm] = (staged_path, len(data))
        return len(data)

    def _write_journal(self, payload: dict) -> None:
        os.makedirs(self._journal_dir, exist_ok=True)
        write_bytes_atomic(json.dumps(payload, indent=2).encode('utf-8'), self._journal_path)

    def commit(self) -> List[str]:
        """Install all staged files as a group; roll back on any failure.

        Returns the list of final paths installed. Raises on failure after
        restoring the filesystem to its pre-commit state.

        Journal states advance prepared -> backed_up -> committed, and the
        journal always lands on disk *before* the move it describes, so a
        crash at any point is recoverable on the next run.
        """
        if self._finished:
            raise RuntimeError('Transaction already finished')
        targets = list(self._staged.keys())
        operations = [{'target': t, 'backup': (t + f'{BACKUP_SUFFIX}.{self.txn_id}') if os.path.exists(t) else None} for t in targets]
        try:
            # Phase 0: journal intent before touching any target.
            self._write_journal({'txn_id': self.txn_id, 'state': 'prepared', 'operations': operations})

            # Phase 1: move pre-existing targets to backups.
            for op in operations:
                if op['backup']:
                    os.replace(op['target'], op['backup'])
            self._write_journal({'txn_id': self.txn_id, 'state': 'backed_up', 'operations': operations})
            _fsync_dir(self.output_dir)

            # Phase 2: install staged files (each os.replace is atomic).
            for target in targets:
                staged_path, _ = self._staged[target]
                os.replace(staged_path, target)
            _fsync_dir(self.output_dir)
            self._write_journal({'txn_id': self.txn_id, 'state': 'committed', 'operations': operations})

            # Phase 3: discard backups.
            for op in operations:
                if op['backup']:
                    try:
                        os.unlink(op['backup'])
                    except FileNotFoundError:
                        pass
            self._finish()
            return targets
        except Exception:
            self._restore(operations)
            raise

    def _restore(self, operations: List[dict]) -> None:
        for op in operations:
            target = op.get('target')
            backup = op.get('backup')
            if not target:
                continue
            try:
                if backup and os.path.exists(backup):
                    # Undoes both "target moved away" and any partial install.
                    os.replace(backup, target)
                elif backup is None and os.path.exists(target):
                    # Target did not exist before this transaction.
                    os.unlink(target)
            except OSError:
                pass
        self._finish()

    def rollback(self) -> None:
        """Discard staged products without touching the output directory."""
        self._finish()

    def _finish(self) -> None:
        self._finished = True
        try:
            if os.path.isdir(self._staging_dir):
                for name in os.listdir(self._staging_dir):
                    try:
                        os.unlink(os.path.join(self._staging_dir, name))
                    except OSError:
                        pass
                os.rmdir(self._staging_dir)
            # Remove the staging root only if no other transaction remains.
            staging_root = os.path.dirname(self._staging_dir)
            if os.path.isdir(staging_root) and not os.listdir(staging_root):
                os.rmdir(staging_root)
        except OSError:
            pass
        try:
            if os.path.isfile(self._journal_path):
                os.unlink(self._journal_path)
            if os.path.isdir(self._journal_dir) and not os.listdir(self._journal_dir):
                os.rmdir(self._journal_dir)
        except OSError:
            pass


def recover_pending_transactions(output_dir: str) -> List[str]:
    """Restore files from interrupted commits found on disk (crash recovery).

    Looks for journals left by a previous process: targets with a backup are
    moved back to their pre-commit state, targets that did not exist before
    are removed. Safe to call on every batch start. Returns recovered
    journal paths.
    """
    journal_dir = os.path.join(output_dir, JOURNAL_DIRNAME)
    recovered = []
    for journal_path in sorted(glob.glob(os.path.join(journal_dir, 'txn-*.json'))):
        try:
            with open(journal_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (OSError, ValueError):
            payload = None
        state = payload.get('state') if payload else None
        ops = payload.get('operations', []) if payload else []
        if state in ('prepared', 'backed_up'):
            # Commit did not finish: restore pre-commit state.
            for op in ops:
                target = op.get('target')
                backup = op.get('backup')
                if not target:
                    continue
                try:
                    if backup and os.path.exists(backup):
                        # Works whether target is missing or holds a partial install.
                        os.replace(backup, target)
                    elif state == 'backed_up' and backup is None and os.path.exists(target):
                        os.unlink(target)
                    # state == 'prepared' with no backup: the original was never moved.
                except OSError:
                    pass
        elif state == 'committed':
            # Installs completed; only backup cleanup remained.
            for op in ops:
                backup = op.get('backup')
                if backup:
                    try:
                        os.unlink(backup)
                    except OSError:
                        pass
        try:
            os.unlink(journal_path)
            recovered.append(journal_path)
        except OSError:
            pass
    # Remove stale staging trees (their commit never happened).
    staging_root = os.path.join(output_dir, STAGING_DIRNAME)
    if os.path.isdir(staging_root):
        for name in os.listdir(staging_root):
            stale = os.path.join(staging_root, name)
            try:
                for sub in sorted(os.listdir(stale), reverse=True):
                    os.unlink(os.path.join(stale, sub))
                os.rmdir(stale)
            except OSError:
                pass
        try:
            os.rmdir(staging_root)
        except OSError:
            pass
    try:
        if os.path.isdir(journal_dir) and not os.listdir(journal_dir):
            os.rmdir(journal_dir)
    except OSError:
        pass
    return recovered
