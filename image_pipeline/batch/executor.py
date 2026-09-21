from typing import Any, Dict, List, Optional, Callable
import os
import json
import time
import traceback
from ..pipeline.engine import PipelineExecutor
from ..utils.types import BatchReport, ImageProcessingResult, ValidationResult, ValidationError
from ..utils.image_io import find_images, is_valid_image

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

    def run(self) -> BatchReport:
        report = BatchReport(pipeline_config_file=self.config_file, input_dir=self.input_dir, output_dir=self.output_dir)
        overall_start = time.perf_counter()
        try:
            self._ensure_output_dir()
        except Exception as e:
            report.failed = 0
            report.total = 0
            report.succeeded = 0
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=f'Failed to create output directory: {e}')
            report.results.append(dummy)
            return report
        try:
            image_paths = self._collect_input_images()
        except ValidationError as e:
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=str(e))
            report.results.append(dummy)
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report
        report.total = len(image_paths)
        for idx, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            img_result: ImageProcessingResult
            if not is_valid_image(img_path):
                img_result = ImageProcessingResult(input_path=img_path, success=False, error='Image failed pre-check verification (likely corrupt or unsupported format)')
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
                img_result = ImageProcessingResult(input_path=img_path, success=False, error=f'Unexpected error during execution: {e}\n{traceback.format_exc()}')
                report.failed += 1
            report.results.append(img_result)
            if self.progress_callback:
                try:
                    self.progress_callback(idx + 1, report.total, img_result)
                except Exception:
                    pass
        report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
        return report

    def write_report(self, report: BatchReport, path: str=None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, 'batch_report.json')
        report_dir = os.path.dirname(path)
        if report_dir and (not os.path.exists(report_dir)):
            os.makedirs(report_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

def print_text_report(report: BatchReport, verbose: bool=False) -> str:
    lines = []
    lines.append('=' * 60)
    lines.append('BATCH PROCESSING REPORT')
    lines.append('=' * 60)
    lines.append(f'Pipeline config : {report.pipeline_config_file}')
    lines.append(f'Input directory : {report.input_dir}')
    lines.append(f'Output directory: {report.output_dir}')
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
    if verbose:
        lines.append('--- Per-Image Details ---')
        for r in report.results:
            status = 'OK' if r.success else 'FAIL'
            out = r.output_path or '(no output)'
            err = f'\n    ERROR: {r.error}' if r.error else ''
            lines.append(f'  [{status}] {r.input_path} -> {out} ({r.duration_ms:.2f} ms){err}')
            if verbose and r.node_results:
                for nr in r.node_results:
                    nstatus = 'OK' if nr.success else 'FAIL'
                    size = f'{nr.output_size[0]}x{nr.output_size[1]}' if nr.output_size else '?'
                    nerr = f' -> {nr.error}' if nr.error else ''
                    lines.append(f'      + {nstatus} {nr.node_id} ({nr.node_type}, {size}, {nr.duration_ms:.2f} ms){nerr}')
        lines.append('')
    if report.failed > 0:
        lines.append('--- Failed Images ---')
        for r in report.results:
            if not r.success:
                lines.append(f'  {r.input_path}')
                lines.append(f'    Reason: {r.error}')
        lines.append('')
    return '\n'.join(lines)
