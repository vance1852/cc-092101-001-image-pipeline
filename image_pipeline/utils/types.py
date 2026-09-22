from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class NodeType(str, Enum):
    INPUT = 'input'
    GRAYSCALE = 'grayscale'
    BRIGHTNESS = 'brightness'
    CONTRAST = 'contrast'
    THRESHOLD = 'threshold'
    BOX_BLUR = 'box_blur'
    GAUSSIAN_BLUR = 'gaussian_blur'
    SHARPEN = 'sharpen'
    SOBEL = 'sobel'
    PREWITT = 'prewitt'
    CROP = 'crop'
    RESIZE = 'resize'
    OUTPUT = 'output'

# Per-image processing status
IMAGE_SUCCESS = 'success'
IMAGE_FAILED = 'failed'
IMAGE_ROLLED_BACK = 'rolled_back'
IMAGE_PREFLIGHT_REJECTED = 'preflight_rejected'

# Batch-level status (surfaced in both JSON and text receipts)
BATCH_COMPLETE_SUCCESS = 'complete_success'
BATCH_PARTIAL_FAILURE = 'partial_failure'
BATCH_TOTAL_FAILURE = 'total_failure'
BATCH_PREFLIGHT_REJECTED = 'preflight_rejected'
BATCH_ROLLED_BACK = 'rolled_back'
BATCH_SETUP_ERROR = 'setup_error'

BATCH_STATUS_LABELS = {
    BATCH_COMPLETE_SUCCESS: 'COMPLETE SUCCESS',
    BATCH_PARTIAL_FAILURE: 'PARTIAL FAILURE',
    BATCH_TOTAL_FAILURE: 'TOTAL FAILURE',
    BATCH_PREFLIGHT_REJECTED: 'PREFLIGHT REJECTED',
    BATCH_ROLLED_BACK: 'ROLLED BACK',
    BATCH_SETUP_ERROR: 'SETUP ERROR',
}

class PipelineError(Exception):
    pass

class ValidationError(PipelineError):
    pass

class ExecutionError(PipelineError):
    pass

@dataclass
class ValidationIssue:
    level: str
    message: str
    node_id: Optional[str] = None
    field: Optional[str] = None

    def __str__(self):
        parts = [f'[{self.level.upper()}]']
        if self.node_id:
            parts.append(f'(node={self.node_id})')
        if self.field:
            parts.append(f'[field={self.field}]')
        parts.append(self.message)
        return ' '.join(parts)

@dataclass
class ValidationResult:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all((i.level != 'error' for i in self.issues))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == 'error']

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == 'warning']

    def add_error(self, msg: str, node_id: str=None, field: str=None):
        self.issues.append(ValidationIssue('error', msg, node_id, field))

    def add_warning(self, msg: str, node_id: str=None, field: str=None):
        self.issues.append(ValidationIssue('warning', msg, node_id, field))

    def extend(self, other: 'ValidationResult'):
        self.issues.extend(other.issues)

    def __str__(self):
        lines = [f"Validation: {('PASS' if self.valid else 'FAIL')}"]
        for i in self.issues:
            lines.append(f'  {str(i)}')
        return '\n'.join(lines)

@dataclass
class OutputArtifact:
    """One concrete file produced by one output node for one input image."""
    node_id: str
    node_type: str
    path: Optional[str] = None
    bytes_size: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    rolled_back: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {'node_id': self.node_id, 'node_type': self.node_type, 'path': self.path, 'bytes': self.bytes_size, 'width': self.width, 'height': self.height, 'success': self.success, 'rolled_back': self.rolled_back, 'error': self.error}

@dataclass
class NodeExecutionResult:
    node_id: str
    node_type: str
    success: bool
    duration_ms: float = 0.0
    error: Optional[str] = None
    output_size: Optional[tuple] = None
    output_path: Optional[str] = None
    output_bytes: Optional[int] = None
    # True once the group commit installed this node's file; False when it was
    # staged but rolled back; None for non-output / non-committing nodes.
    committed: Optional[bool] = None

@dataclass
class ImageProcessingResult:
    input_path: str
    output_path: Optional[str] = None
    success: bool = False
    duration_ms: float = 0.0
    error: Optional[str] = None
    node_results: List[NodeExecutionResult] = field(default_factory=list)
    outputs: List[OutputArtifact] = field(default_factory=list)
    status: str = IMAGE_FAILED

@dataclass
class BatchReport:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0
    results: List[ImageProcessingResult] = field(default_factory=list)
    pipeline_config_file: str = ''
    input_dir: str = ''
    output_dir: str = ''
    status: str = ''
    preflight_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {'status': self.status, 'summary': {'total': self.total, 'succeeded': self.succeeded, 'failed': self.failed, 'skipped': self.skipped, 'total_duration_ms': round(self.total_duration_ms, 2)}, 'config': {'pipeline_file': self.pipeline_config_file, 'input_dir': self.input_dir, 'output_dir': self.output_dir}, 'preflight_errors': self.preflight_errors, 'results': [{'input': r.input_path, 'output': r.output_path, 'status': r.status, 'success': r.success, 'duration_ms': round(r.duration_ms, 2), 'error': r.error, 'outputs': [a.to_dict() for a in r.outputs], 'nodes': [{'node_id': nr.node_id, 'node_type': nr.node_type, 'success': nr.success, 'duration_ms': round(nr.duration_ms, 2), 'error': nr.error, 'output_size': list(nr.output_size) if nr.output_size else None, 'output_path': nr.output_path, 'output_bytes': nr.output_bytes, 'committed': nr.committed} for nr in r.node_results]} for r in self.results]}
