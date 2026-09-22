from typing import Any, Dict, List, Optional, Callable
import os
import json
import time
import traceback
from ..pipeline.engine import PipelineExecutor
from ..utils.types import (BatchReport, ImageProcessingResult, OutputArtifact, ValidationError,
                          IMAGE_FAILED, IMAGE_PREFLIGHT_REJECTED, IMAGE_ROLLED_BACK,
                          BATCH_COMPLETE_SUCCESS, BATCH_PARTIAL_FAILURE, BATCH_TOTAL_FAILURE,
                          BATCH_PREFLIGHT_REJECTED, BATCH_ROLLED_BACK, BATCH_SETUP_ERROR)
from ..utils.image_io import find_images, is_valid_image, write_bytes_atomic
from ..utils.transaction import recover_pending_transactions
from ..nodes.definitions import resolve_output_path


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
            raise ValidationError(f"Input directory does not exist: '{self.input_dir}'")
        return find_images(self.input_dir)

    def _output_nodes(self):
        return self.executor.graph.get_output_nodes()

    def plan_targets(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict every output file: [{input, input_filename, node_id, target}]."""
        plan = []
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            for node in self._output_nodes():
                target = os.path.abspath(resolve_output_path(self.output_dir, filename, node.effective_params()))
                plan.append({'input': img_path, 'input_filename': filename, 'node_id': node.node_id, 'target': target})
        return plan

    def detect_conflicts(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Identify target-path collisions before any pixel is processed.

        Two kinds are reported:
          * within_image - two output nodes of one pipeline resolve to the same
            file for one input image (e.g. identical suffix + same format);
          * cross_image  - output nodes for two different input images resolve
            to the same file (e.g. 'a.png' and 'a.jpg' both exported as PNG
            with an empty suffix).
        Identification is based on the actual output format/extension and the
        concrete file name, not on node IDs alone.
        """
        plan = self.plan_targets(image_paths)
        by_target: Dict[str, List[Dict[str, Any]]] = {}
        for item in plan:
            by_target.setdefault(item['target'], []).append(item)
        conflicts = []
        for target in sorted(by_target.keys()):
            sources = by_target[target]
            if len(sources) < 2:
                continue
            # Within one image: multiple nodes claiming the same target.
            per_image: Dict[str, List[Dict[str, Any]]] = {}
            for s in sources:
                per_image.setdefault(s['input'], []).append(s)
            for img_path, items in per_image.items():
                node_ids = [s['node_id'] for s in items]
                if len(set(node_ids)) >= 2:
                    conflicts.append({'type': 'within_image', 'target': target, 'input': img_path, 'node_ids': sorted(set(node_ids)), 'sources': [{'input': s['input'], 'node_id': s['node_id']} for s in items]})
            # Across different input images.
            distinct_inputs = []
            seen = set()
            for s in sources:
                if s['input'] not in seen:
                    seen.add(s['input'])
                    distinct_inputs.append(s)
            if len(distinct_inputs) >= 2:
                conflicts.append({'type': 'cross_image', 'target': target, 'sources': [{'input': s['input'], 'node_id': s['node_id']} for s in sources]})
        return conflicts

    def run(self) -> BatchReport:
        report = BatchReport(pipeline_config_file=self.config_file, input_dir=self.input_dir, output_dir=self.output_dir)
        overall_start = time.perf_counter()
        try:
            self._ensure_output_dir()
        except Exception as e:
            report.status = BATCH_SETUP_ERROR
            report.failed = 0
            report.total = 0
            report.succeeded = 0
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=f'Failed to create output directory: {e}')
            report.results.append(dummy)
            return report
        # Finish any transaction interrupted by a previous crash.
        recover_pending_transactions(self.output_dir)
        try:
            image_paths = self._collect_input_images()
        except ValidationError as e:
            report.status = BATCH_SETUP_ERROR
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=str(e))
            report.results.append(dummy)
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report
        report.total = len(image_paths)

        # Preflight: refuse the whole batch when output targets collide.
        conflicts = self.detect_conflicts(image_paths)
        if conflicts:
            report.status = BATCH_PREFLIGHT_REJECTED
            report.skipped = report.total
            report.preflight_errors = conflicts
            for c in conflicts:
                report.results.append(self._preflight_result(c))
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report

        for idx, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            img_result: ImageProcessingResult
            if not is_valid_image(img_path):
                img_result = ImageProcessingResult(input_path=img_path, success=False, status=IMAGE_FAILED, error='Image failed pre-check verification (likely corrupt or unsupported format)')
                report.failed += 1
                report.results.append(img_result)
                if self.progress_callback:
                    try:
                        self.progress_callback(idx + 1, report.total, img_result)
                    except Exception:
                        pass
                continue
            context: Dict[str, Any] = {'input_path': img_path, 'input_filename': filename, 'output_dir': self.output_dir, 'image_index': idx}
            try:
                img_result = self.executor.run(context)
                if img_result.success:
                    report.succeeded += 1
                else:
                    report.failed += 1
            except Exception as e:
                img_result = ImageProcessingResult(input_path=img_path, success=False, status=IMAGE_FAILED, error=f'Unexpected error during execution: {e}\n{traceback.format_exc()}')
                report.failed += 1
            report.results.append(img_result)
            if self.progress_callback:
                try:
                    self.progress_callback(idx + 1, report.total, img_result)
                except Exception:
                    pass
        report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
        any_rollback = any((r.status == IMAGE_ROLLED_BACK for r in report.results))
        if report.total > 0 and report.failed == 0:
            report.status = BATCH_COMPLETE_SUCCESS
        elif report.succeeded > 0 and report.failed > 0:
            report.status = BATCH_PARTIAL_FAILURE
        elif any_rollback:
            report.status = BATCH_ROLLED_BACK
        else:
            report.status = BATCH_TOTAL_FAILURE
        return report

    def _preflight_result(self, conflict: Dict[str, Any]) -> ImageProcessingResult:
        node_ids = conflict.get('node_ids') or sorted({s['node_id'] for s in conflict.get('sources', [])})
        if conflict['type'] == 'within_image':
            msg = (f"Preflight rejected: output nodes {node_ids} resolve to the same "
                   f"file '{conflict['target']}' for input '{conflict.get('input')}'. "
                   "Give the nodes distinct suffixes or formats.")
            input_path = conflict.get('input', '')
        else:
            inputs = sorted({s['input'] for s in conflict.get('sources', [])})
            msg = (f"Preflight rejected: {len(inputs)} different input images resolve to "
                   f"the same output file '{conflict['target']}' (inputs: {', '.join(inputs)}). "
                   "Output names must be unique across the batch.")
            input_path = inputs[0] if inputs else ''
        artifacts = [OutputArtifact(node_id=nid, node_type='output', path=conflict['target'], success=False, error='preflight conflict') for nid in node_ids]
        return ImageProcessingResult(input_path=input_path, success=False, status=IMAGE_PREFLIGHT_REJECTED, error=msg, outputs=artifacts)

    def write_report(self, report: BatchReport, path: str=None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, 'batch_report.json')
        report_dir = os.path.dirname(path)
        if report_dir and (not os.path.exists(report_dir)):
            os.makedirs(report_dir, exist_ok=True)
        # Atomic receipt: a crash during serialization can never truncate the
        # last valid batch_report.json.
        data = json.dumps(report.to_dict(), indent=2, ensure_ascii=False).encode('utf-8')
        write_bytes_atomic(data, path)
        return path


def print_text_report(report: BatchReport, verbose: bool=False) -> str:
    from ..utils.types import BATCH_STATUS_LABELS, IMAGE_ROLLED_BACK
    lines = []
    lines.append('=' * 60)
    lines.append('BATCH PROCESSING REPORT')
    lines.append('=' * 60)
    status_label = BATCH_STATUS_LABELS.get(report.status, report.status or 'UNKNOWN')
    lines.append(f'Status          : {status_label}')
    lines.append(f'Pipeline config : {report.pipeline_config_file}')
    lines.append(f'Input directory : {report.input_dir}')
    lines.append(f'Output directory: {report.output_dir}')
    lines.append('')
    if report.status == BATCH_PREFLIGHT_REJECTED:
        lines.append('--- Preflight Conflicts (NOTHING WAS WRITTEN) ---')
        for c in report.preflight_errors:
            if c['type'] == 'within_image':
                node_ids = c.get('node_ids') or sorted({s['node_id'] for s in c['sources']})
                lines.append(f"  [within-image] nodes {node_ids} -> {c['target']}")
                lines.append(f"      input: {c.get('input')}")
            else:
                lines.append(f"  [cross-image] {c['target']}")
                for s in c['sources']:
                    lines.append(f"      {s['input']} via node '{s['node_id']}'")
        lines.append('')
        lines.append('Batch rejected before execution. Fix the naming conflicts above and re-run; no files need cleanup.')
        return '\n'.join(lines)
    lines.append('--- Summary ---')
    lines.append(f'Total images    : {report.total}')
    lines.append(f'Succeeded       : {report.succeeded}')
    lines.append(f'Failed          : {report.failed}')
    lines.append(f'Skipped         : {report.skipped}')
    lines.append(f'Total duration  : {report.total_duration_ms:.2f} ms')
    if report.total > 0:
        lines.append(f'Avg per image   : {report.total_duration_ms / report.total:.2f} ms')
    lines.append('')
    if verbose:
        lines.append('--- Per-Image Details ---')
        for r in report.results:
            if r.success:
                status = 'OK'
            elif r.status == IMAGE_ROLLED_BACK:
                status = 'ROLLBACK'
            else:
                status = 'FAIL'
            out = r.output_path or '(no output)'
            err = f'\n    ERROR: {r.error}' if r.error else ''
            lines.append(f'  [{status}] {r.input_path} -> {out} ({r.duration_ms:.2f} ms){err}')
            for artifact in r.outputs:
                astatus = 'OK' if artifact.success else ('ROLLED-BACK' if artifact.rolled_back else 'FAIL')
                size_txt = f'{artifact.width}x{artifact.height}' if artifact.width else '?'
                lines.append(f'      * [{astatus}] node={artifact.node_id} ({artifact.node_type}) {size_txt} {artifact.bytes_size} bytes -> {artifact.path or "(not written)"}')
                if artifact.error:
                    lines.append(f'          {artifact.error}')
            if verbose and r.node_results:
                for nr in r.node_results:
                    nstatus = 'OK' if nr.success else 'FAIL'
                    size = f'{nr.output_size[0]}x{nr.output_size[1]}' if nr.output_size else '?'
                    nerr = f' -> {nr.error}' if nr.error else ''
                    nbytes = f', {nr.output_bytes} bytes' if nr.output_bytes is not None else ''
                    lines.append(f'      + {nstatus} {nr.node_id} ({nr.node_type}, {size}{nbytes}, {nr.duration_ms:.2f} ms){nerr}')
        lines.append('')
    rolled_back = [r for r in report.results if r.status == IMAGE_ROLLED_BACK]
    if rolled_back:
        lines.append('--- Rolled-Back Images (no partial products left on disk) ---')
        for r in rolled_back:
            lines.append(f'  {r.input_path}')
            lines.append(f'    Reason: {r.error}')
        lines.append('')
        lines.append('Re-run is safe: previous valid output files were restored, staged files discarded.')
        lines.append('')
    if report.failed > 0:
        lines.append('--- Failed Images ---')
        for r in report.results:
            if not r.success and r.status != IMAGE_ROLLED_BACK:
                lines.append(f'  {r.input_path}')
                lines.append(f'    Reason: {r.error}')
        lines.append('')
    if report.status == BATCH_COMPLETE_SUCCESS:
        lines.append('All outputs committed atomically. Re-running reproduces the same result.')
    return '\n'.join(lines)
