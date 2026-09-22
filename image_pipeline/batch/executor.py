from typing import Any, Dict, List, Optional, Callable
import os
import json
import time
import traceback
from ..pipeline.engine import PipelineExecutor
from ..utils.types import (
    BatchReport,
    ImageProcessingResult,
    STATUS_COMPLETE,
    STATUS_PREFLIGHT_REJECTED,
    STATUS_ROLLED_BACK,
    STATUS_FAILED,
)
from ..utils.image_io import find_images, is_valid_image
from ..utils.targets import resolve_output_name
from ..utils.atomic import RecoverableCommit, atomic_write_bytes

# Exit codes shared with the CLI. Kept here so other frontends can reuse them.
EXIT_COMPLETE = 0
EXIT_PARTIAL_FAIL = 1
EXIT_TOTAL_FAIL = 2
EXIT_SETUP_ERROR = 3
EXIT_PREFLIGHT_REJECTED = 4
EXIT_ROLLED_BACK = 5


class BatchExecutor:

    def __init__(self, pipeline_executor: PipelineExecutor, input_dir: str, output_dir: str, config_file: str='', progress_callback: Optional[Callable]=None):
        self.executor = pipeline_executor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config_file = config_file
        self.progress_callback = progress_callback

    def _ensure_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def _collect_input_images(self) -> List[str]:
        if not os.path.isdir(self.input_dir):
            from ..utils.types import ValidationError
            raise ValidationError(f"Input directory does not exist: '{self.input_dir}'")
        return find_images(self.input_dir)

    # ---------------------------------------------------------------- preflight

    def _output_nodes(self):
        return self.executor.graph.get_output_nodes()

    def _plan_targets(self, image_paths: List[str]) -> List[str]:
        """Compute every planned output file before touching the disk.

        Returns a list of human-readable collision descriptions covering both:

        * within one image: two output nodes resolving to the same file
          (e.g. identical suffix with the same effective format);
        * across inputs: two different source images resolving to the same
          file (e.g. ``a.jpg``/``a.png`` both forced to PNG, or an input
          named after another input's suffixed output).
        """
        nodes = self._output_nodes()
        # rel output name (normalized) -> list of (image_path, node_id)
        owners: Dict[str, List[tuple]] = {}
        per_image_owners: Dict[str, Dict[str, List[str]]] = {}
        conflicts: List[str] = []
        for img_path in image_paths:
            fname = os.path.basename(img_path)
            per_image: Dict[str, List[str]] = {}
            for node in nodes:
                out_name, _ext, _fmt = resolve_output_name(fname, node.effective_params())
                key = os.path.normcase(out_name)
                per_image.setdefault(key, []).append(node.node_id)
                owners.setdefault(key, []).append((img_path, node.node_id))
            per_image_owners[fname] = per_image
        for fname, per_image in per_image_owners.items():
            for key, node_ids in per_image.items():
                if len(node_ids) > 1:
                    conflicts.append(
                        f"Within-image conflict for '{fname}': output nodes "
                        f"{sorted(node_ids)} all resolve to the same file "
                        f"'{os.path.join(self.output_dir, key)}'"
                    )
        for key, entries in owners.items():
            source_images = sorted({img for img, _n in entries})
            if len(source_images) > 1:
                detail = ', '.join(
                    f"'{os.path.basename(img)}' via node '{nid}'" for img, nid in entries
                )
                conflicts.append(
                    f"Cross-input conflict on '{os.path.join(self.output_dir, key)}': {detail}"
                )
        return conflicts

    # ------------------------------------------------------------------- run

    def run(self) -> BatchReport:
        report = BatchReport(pipeline_config_file=self.config_file, input_dir=self.input_dir, output_dir=self.output_dir)
        overall_start = time.perf_counter()
        try:
            self._ensure_output_dir()
        except Exception as e:
            report.total = 0
            report.failed = 0
            report.succeeded = 0
            report.status = STATUS_FAILED
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=f'Failed to create output directory: {e}')
            report.results.append(dummy)
            return report
        # Replay journals left behind by a crashed earlier run so the output
        # directory holds the last complete/valid file set before we start.
        try:
            RecoverableCommit.recover_all(self.output_dir)
        except Exception:
            pass
        try:
            image_paths = self._collect_input_images()
        except Exception as e:
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=str(e))
            report.results.append(dummy)
            report.status = STATUS_FAILED
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report
        report.total = len(image_paths)

        conflicts = self._plan_targets(image_paths)
        if conflicts:
            # Structural conflicts are decided purely from filenames/formats,
            # so refuse everything up front: no image is processed and no
            # output file is created or replaced.
            report.status = STATUS_PREFLIGHT_REJECTED
            report.conflicts = conflicts
            report.skipped = report.total
            for img_path in image_paths:
                fname = os.path.basename(img_path)
                relevant = [c for c in conflicts if f"'{fname}'" in c]
                detail = '; '.join(relevant) if relevant else 'Batch rejected due to conflicts on other inputs'
                img_result = ImageProcessingResult(
                    input_path=img_path,
                    success=False,
                    status=STATUS_PREFLIGHT_REJECTED,
                    error=detail,
                )
                report.results.append(img_result)
                self._emit_progress(len(report.results), report.total, img_result)
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report

        # One journal for the whole batch so a write failure never leaves a
        # half-set behind: all outputs are staged first and committed together
        # only after every image has been processed.
        commit = RecoverableCommit(self.output_dir)
        abort_for_rollback = False
        for idx, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            img_result: ImageProcessingResult
            if not is_valid_image(img_path):
                img_result = ImageProcessingResult(input_path=img_path, success=False, status=STATUS_FAILED, error='Image failed pre-check verification (likely corrupt or unsupported format)')
                report.failed += 1
                report.results.append(img_result)
                self._emit_progress(idx + 1, report.total, img_result)
                continue
            if abort_for_rollback:
                img_result = ImageProcessingResult(input_path=img_path, success=False, status=STATUS_ROLLED_BACK, error='Skipped: batch aborted after an output write failed; staged outputs were rolled back')
                report.skipped += 1
                report.results.append(img_result)
                self._emit_progress(idx + 1, report.total, img_result)
                continue
            img_result = self._process_one_image(img_path, filename, idx, commit)
            if img_result.status == STATUS_ROLLED_BACK:
                # Environmental write failure while staging: stop touching the
                # disk; the shared journal restores everything below.
                abort_for_rollback = True
                report.failed += 1
            elif img_result.success:
                report.succeeded += 1
            else:
                report.failed += 1
            report.results.append(img_result)
            self._emit_progress(idx + 1, report.total, img_result)

        if abort_for_rollback:
            restored = self._safe_rollback(commit)
            commit.discard()
            rollback_note = (
                f'Output staging failed; rolled back {len(restored)} file(s) '
                'and preserved the previous valid outputs. Fix the disk '
                'problem and rerun safely.'
            )
            report.status = STATUS_ROLLED_BACK
            for r in report.results:
                if r.status != STATUS_FAILED:
                    r.status = STATUS_ROLLED_BACK
                r.success = False
                r.outputs = []
                r.output_path = None
                if r.status == STATUS_ROLLED_BACK and not r.error:
                    r.error = rollback_note
            report.succeeded = 0
            report.failed = sum(1 for r in report.results if r.status == STATUS_FAILED)
            report.skipped = report.total - report.failed
        else:
            try:
                commit.commit_pending()
                commit.discard()
                report.status = STATUS_COMPLETE if report.failed == 0 else STATUS_FAILED
            except Exception as e:
                restored = self._safe_rollback(commit)
                commit.discard()
                note = (
                    f'Atomic commit failed while writing batch outputs: {e}. '
                    f'Rolled back {len(restored)} file(s); previous valid outputs preserved.'
                )
                report.status = STATUS_ROLLED_BACK
                for r in report.results:
                    r.success = False
                    r.status = STATUS_ROLLED_BACK
                    r.outputs = []
                    r.output_path = None
                    if not r.error:
                        r.error = note
                report.succeeded = 0
                report.failed = 0
                report.skipped = report.total
        report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
        return report

    def _emit_progress(self, done: int, total: int, result: ImageProcessingResult) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(done, total, result)
            except Exception:
                pass

    def _process_one_image(self, img_path: str, filename: str, idx: int, commit: RecoverableCommit) -> ImageProcessingResult:
        """Run one image through the graph, staging outputs in the batch journal."""
        write_error: Optional[str] = None

        def output_sink(node_id: str, rel_name: str, data: bytes) -> str:
            nonlocal write_error
            try:
                commit.stage_output(rel_name, data)
            except Exception as e:
                # Destinations are still untouched but the environment cannot
                # accept writes: classify as a write rollback, not a bad image.
                write_error = f"node '{node_id}' could not stage '{rel_name}': {e}"
                raise
            return os.path.join(self.output_dir, rel_name)

        context: Dict[str, Any] = {
            'input_path': img_path,
            'input_filename': filename,
            'output_dir': self.output_dir,
            'image_index': idx,
            'output_sink': output_sink,
        }
        try:
            img_result = self.executor.run(context)
        except Exception as e:
            img_result = ImageProcessingResult(
                input_path=img_path,
                success=False,
                status=STATUS_FAILED,
                error=f'Unexpected error during execution: {e}\n{traceback.format_exc()}',
            )
        if not img_result.success and write_error is not None:
            img_result.status = STATUS_ROLLED_BACK
            img_result.outputs = []
            img_result.output_path = None
            img_result.error = (
                f'Write failed before outputs were installed ({write_error}). '
                'Batch outputs will be rolled back; previous valid files preserved.'
            )
        return img_result

    @staticmethod
    def _safe_rollback(commit: RecoverableCommit) -> List[str]:
        try:
            return commit.rollback()
        except Exception:
            return []

    def write_report(self, report: BatchReport, path: str=None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, 'batch_report.json')
        report_dir = os.path.dirname(path)
        if report_dir and (not os.path.exists(report_dir)):
            os.makedirs(report_dir, exist_ok=True)
        data = json.dumps(report.to_dict(), indent=2, ensure_ascii=False).encode('utf-8')
        # Atomic replace: a crash while writing the receipt cannot corrupt the
        # previous valid batch_report.json.
        atomic_write_bytes(path, data)
        return path


_STATUS_LABELS = {
    STATUS_COMPLETE: 'COMPLETE',
    STATUS_PREFLIGHT_REJECTED: 'PREFLIGHT REJECTED',
    STATUS_ROLLED_BACK: 'ROLLED BACK (WRITE FAILURE)',
    STATUS_FAILED: 'FAILED',
}


def print_text_report(report: BatchReport, verbose: bool=False) -> str:
    lines = []
    lines.append('=' * 60)
    lines.append('BATCH PROCESSING REPORT')
    lines.append('=' * 60)
    lines.append(f'Pipeline config : {report.pipeline_config_file}')
    lines.append(f'Input directory : {report.input_dir}')
    lines.append(f'Output directory: {report.output_dir}')
    lines.append(f'Batch status    : {_STATUS_LABELS.get(report.status, report.status)}')
    lines.append('')
    lines.append('--- Summary ---')
    lines.append(f'Total images    : {report.total}')
    lines.append(f'Succeeded       : {report.succeeded}')
    lines.append(f'Failed          : {report.failed}')
    lines.append(f'Skipped         : {report.skipped}')
    lines.append(f'Total duration  : {report.total_duration_ms:.2f} ms')
    if report.total > 0:
        lines.append(f'Avg per image   : {report.total_duration_ms / report.total:.2f} ms')
    lines.append('')
    if report.conflicts:
        lines.append('--- Preflight Conflicts (nothing was written) ---')
        for c in report.conflicts:
            lines.append(f'  ! {c}')
        lines.append('')
        lines.append('Action: fix suffix/format parameters or rename inputs, then rerun safely.')
        lines.append('')
    if verbose:
        lines.append('--- Per-Image Details ---')
        for r in report.results:
            label = _STATUS_LABELS.get(r.status, r.status)
            status = 'OK' if r.success else label
            out = r.output_path or '(no output)'
            err = f'\n    ERROR: {r.error}' if r.error else ''
            lines.append(f'  [{status}] {r.input_path} -> {out} ({r.duration_ms:.2f} ms){err}')
            if r.outputs:
                lines.append('      outputs:')
                for art in r.outputs:
                    size = f'{art.image_size[0]}x{art.image_size[1]}' if art.image_size else '?'
                    lines.append(f'      + node={art.node_id}  {art.path}  pixels={size}  bytes={art.size_bytes}')
            if verbose and r.node_results:
                for nr in r.node_results:
                    nstatus = 'OK' if nr.success else 'FAIL'
                    size = f'{nr.output_size[0]}x{nr.output_size[1]}' if nr.output_size else '?'
                    nerr = f' -> {nr.error}' if nr.error else ''
                    lines.append(f'      + {nstatus} {nr.node_id} ({nr.node_type}, {size}, {nr.duration_ms:.2f} ms){nerr}')
        lines.append('')
    failed_results = [r for r in report.results if not r.success]
    if failed_results and report.status != STATUS_PREFLIGHT_REJECTED:
        lines.append('--- Failed Images ---')
        for r in failed_results:
            lines.append(f'  [{_STATUS_LABELS.get(r.status, r.status)}] {r.input_path}')
            lines.append(f'    Reason: {r.error}')
        lines.append('')
    if report.status == STATUS_ROLLED_BACK:
        lines.append('Action: outputs were rolled back to the previous valid files; fix the disk problem and rerun safely.')
        lines.append('')
    elif report.status == STATUS_COMPLETE:
        lines.append('Result: complete success. Rerun is idempotent.')
        lines.append('')
    return '\n'.join(lines)
