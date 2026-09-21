from typing import Any, Dict, List, Optional, Tuple
import abc
from ..algorithms import core as alg
from ..utils.types import NodeType, ValidationResult, ValidationError, ExecutionError
from ..utils.image_io import read_image, write_image
from ..algorithms.core import Image

class PipelineNode(abc.ABC):
    node_type: NodeType = None
    input_count: int = 1
    output_count: int = 1

    def __init__(self, node_id: str, params: Dict[str, Any]=None):
        self.node_id = node_id
        self.params = dict(params) if params else {}
        self._execution_context: Dict[str, Any] = {}

    def get_default_params(self) -> Dict[str, Any]:
        return {}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def effective_params(self) -> Dict[str, Any]:
        effective = self.get_default_params().copy()
        effective.update(self.params)
        return effective

    def validate_params(self) -> ValidationResult:
        result = ValidationResult()
        schema = self.get_param_schema()
        effective = self.effective_params()
        for name, spec in schema.items():
            value = effective.get(name)
            required = spec.get('required', False)
            expected_type = spec.get('type')
            options = spec.get('options')
            min_val = spec.get('min')
            max_val = spec.get('max')
            validator = spec.get('validator')
            if required and value is None:
                result.add_error(f"Parameter '{name}' is required", self.node_id, name)
                continue
            if value is None and (not required):
                continue
            if expected_type is not None and (not self._check_type(value, expected_type)):
                result.add_error(f"Parameter '{name}' expected type {expected_type.__name__}, got {type(value).__name__}", self.node_id, name)
                continue
            if min_val is not None and value < min_val:
                result.add_error(f"Parameter '{name}' = {value} is below minimum {min_val}", self.node_id, name)
                continue
            if max_val is not None and value > max_val:
                result.add_error(f"Parameter '{name}' = {value} exceeds maximum {max_val}", self.node_id, name)
                continue
            if options is not None and value not in options:
                result.add_error(f"Parameter '{name}' = {value} is not in allowed options: {options}", self.node_id, name)
                continue
            if validator is not None:
                ok, msg = validator(self.node_id, name, value)
                if not ok:
                    result.add_error(msg, self.node_id, name)
                    continue
        return result

    @staticmethod
    def _check_type(value: Any, expected_type) -> bool:
        if isinstance(expected_type, tuple):
            return isinstance(value, expected_type)
        if expected_type is float:
            return isinstance(value, (int, float)) and (not isinstance(value, bool))
        if expected_type is int:
            return isinstance(value, int) and (not isinstance(value, bool))
        return isinstance(value, expected_type)

    def set_context(self, **kwargs) -> None:
        self._execution_context.update(kwargs)

    def clear_context(self) -> None:
        self._execution_context.clear()

    @abc.abstractmethod
    def execute(self, inputs: List[Image]) -> Image:
        ...

    def __repr__(self):
        return f'<{self.__class__.__name__} id={self.node_id}>'

def _validate_odd_int(node_id: str, name: str, value: int) -> Tuple[bool, str]:
    if value % 2 != 1:
        return (False, f"Parameter '{name}' = {value} must be odd integer")
    return (True, '')

def _validate_kernel_2d(node_id: str, name: str, value: list) -> Tuple[bool, str]:
    if not isinstance(value, list) or len(value) == 0:
        return (False, f"Parameter '{name}' must be non-empty 2D list")
    rows = len(value)
    if rows % 2 == 0:
        return (False, f"Parameter '{name}' has {rows} rows, must be odd")
    cols = None
    for i, row in enumerate(value):
        if not isinstance(row, list):
            return (False, f"Parameter '{name}' row {i} must be a list")
        if cols is None:
            cols = len(row)
            if cols % 2 == 0:
                return (False, f"Parameter '{name}' has {cols} columns, must be odd")
        elif len(row) != cols:
            return (False, f"Parameter '{name}' row {i} has {len(row)} elements, expected {cols}")
        for j, v in enumerate(row):
            if not isinstance(v, (int, float)):
                return (False, f"Parameter '{name}[{i}][{j}]' must be numeric, got {type(v).__name__}")
    return (True, '')

class InputNode(PipelineNode):
    node_type = NodeType.INPUT
    input_count = 0
    output_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'hint': {'type': str, 'required': False, 'description': 'Optional description for this input'}}

    def execute(self, inputs: List[Image]) -> Image:
        if inputs:
            raise ExecutionError(f'[{self.node_id}] InputNode should receive no inputs, got {len(inputs)}')
        input_path = self._execution_context.get('input_path')
        if not input_path:
            raise ExecutionError(f'[{self.node_id}] No input_path set in execution context')
        try:
            return read_image(input_path)
        except Exception as e:
            raise ExecutionError(f"[{self.node_id}] Failed to read image '{input_path}': {e}") from e

class GrayscaleNode(PipelineNode):
    node_type = NodeType.GRAYSCALE
    input_count = 1
    output_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        return alg.to_grayscale(inputs[0])

class BrightnessNode(PipelineNode):
    node_type = NodeType.BRIGHTNESS
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'value': 0}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'value': {'type': int, 'required': False, 'default': 0, 'min': -255, 'max': 255}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.adjust_brightness(inputs[0], p['value'])

class ContrastNode(PipelineNode):
    node_type = NodeType.CONTRAST
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'value': 1.0}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'value': {'type': float, 'required': False, 'default': 1.0, 'min': 0.0, 'max': 10.0}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.adjust_contrast(inputs[0], float(p['value']))

class ThresholdNode(PipelineNode):
    node_type = NodeType.THRESHOLD
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'value': 128, 'mode': 'binary'}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'value': {'type': int, 'required': False, 'default': 128, 'min': 0, 'max': 255}, 'mode': {'type': str, 'required': False, 'default': 'binary', 'options': ['binary', 'binary_inv', 'truncate', 'tozero', 'tozero_inv']}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.threshold(inputs[0], int(p['value']), p['mode'])

class BoxBlurNode(PipelineNode):
    node_type = NodeType.BOX_BLUR
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'size': 3}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'size': {'type': int, 'required': False, 'default': 3, 'min': 3, 'max': 31, 'validator': _validate_odd_int}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.box_blur(inputs[0], int(p['size']))

class GaussianBlurNode(PipelineNode):
    node_type = NodeType.GAUSSIAN_BLUR
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'size': 3, 'sigma': 1.0}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'size': {'type': int, 'required': False, 'default': 3, 'min': 3, 'max': 31, 'validator': _validate_odd_int}, 'sigma': {'type': float, 'required': False, 'default': 1.0, 'min': 0.01, 'max': 100.0}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.gaussian_blur(inputs[0], int(p['size']), float(p['sigma']))

class SharpenNode(PipelineNode):
    node_type = NodeType.SHARPEN
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'amount': 1.0}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'amount': {'type': float, 'required': False, 'default': 1.0, 'min': 0.0, 'max': 10.0}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.sharpen(inputs[0], float(p['amount']))

class SobelNode(PipelineNode):
    node_type = NodeType.SOBEL
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'direction': 'both'}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'direction': {'type': str, 'required': False, 'default': 'both', 'options': ['x', 'y', 'both']}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.sobel_edges(inputs[0], p['direction'])

class PrewittNode(PipelineNode):
    node_type = NodeType.PREWITT
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'direction': 'both'}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'direction': {'type': str, 'required': False, 'default': 'both', 'options': ['x', 'y', 'both']}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        return alg.prewitt_edges(inputs[0], p['direction'])

class CropNode(PipelineNode):
    node_type = NodeType.CROP
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'x': 0, 'y': 0, 'width': -1, 'height': -1}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'x': {'type': int, 'required': True, 'min': 0}, 'y': {'type': int, 'required': True, 'min': 0}, 'width': {'type': int, 'required': True, 'min': 1}, 'height': {'type': int, 'required': True, 'min': 1}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        img = inputs[0]
        H = len(img)
        W = len(img[0]) if H else 0
        width = p['width']
        height = p['height']
        if width == -1:
            width = W - p['x']
        if height == -1:
            height = H - p['y']
        return alg.crop(img, int(p['x']), int(p['y']), int(width), int(height))

class ResizeNode(PipelineNode):
    node_type = NodeType.RESIZE
    input_count = 1

    def get_default_params(self) -> Dict[str, Any]:
        return {'scale': None, 'scale_x': None, 'scale_y': None, 'width': None, 'height': None, 'method': 'bilinear'}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'scale': {'type': float, 'required': False, 'min': 0.01, 'max': 100.0}, 'scale_x': {'type': float, 'required': False, 'min': 0.01, 'max': 100.0}, 'scale_y': {'type': float, 'required': False, 'min': 0.01, 'max': 100.0}, 'width': {'type': int, 'required': False, 'min': 1, 'max': 100000}, 'height': {'type': int, 'required': False, 'min': 1, 'max': 100000}, 'method': {'type': str, 'required': False, 'default': 'bilinear', 'options': ['nearest', 'bilinear']}}

    def validate_params(self) -> ValidationResult:
        result = super().validate_params()
        p = self.effective_params()
        scale_given = any((p[k] is not None for k in ('scale', 'scale_x', 'scale_y')))
        dim_given = p['width'] is not None or p['height'] is not None
        if not scale_given and (not dim_given):
            result.add_error('Must specify at least one of: scale, scale_x, scale_y, width, height', self.node_id)
        return result

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        p = self.effective_params()
        scale = p.get('scale')
        kwargs = {'method': p['method']}
        if scale is not None:
            kwargs['scale_x'] = float(scale)
            kwargs['scale_y'] = float(scale)
        if p.get('scale_x') is not None:
            kwargs['scale_x'] = float(p['scale_x'])
        if p.get('scale_y') is not None:
            kwargs['scale_y'] = float(p['scale_y'])
        if p.get('width') is not None:
            kwargs['target_width'] = int(p['width'])
        if p.get('height') is not None:
            kwargs['target_height'] = int(p['height'])
        return alg.resize(inputs[0], **kwargs)

class OutputNode(PipelineNode):
    node_type = NodeType.OUTPUT
    input_count = 1
    output_count = 0

    def get_default_params(self) -> Dict[str, Any]:
        return {'format': None, 'quality': 90, 'suffix': ''}

    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        return {'format': {'type': str, 'required': False, 'options': [None, 'PNG', 'JPEG', 'BMP', 'TIFF', 'WEBP']}, 'quality': {'type': int, 'required': False, 'default': 90, 'min': 1, 'max': 100}, 'suffix': {'type': str, 'required': False, 'default': ''}}

    def execute(self, inputs: List[Image]) -> Image:
        if len(inputs) != 1:
            raise ExecutionError(f'[{self.node_id}] Expected 1 input, got {len(inputs)}')
        img = inputs[0]
        output_dir = self._execution_context.get('output_dir')
        input_filename = self._execution_context.get('input_filename')
        params = self.effective_params()
        if output_dir and input_filename:
            import os
            stem, ext = os.path.splitext(input_filename)
            suffix = params.get('suffix', '')
            fmt = params.get('format')
            if fmt:
                fmt_to_ext = {'PNG': '.png', 'JPEG': '.jpg', 'BMP': '.bmp', 'TIFF': '.tif', 'WEBP': '.webp'}
                out_ext = fmt_to_ext.get(fmt.upper(), ext)
            else:
                out_ext = ext if ext else '.png'
            out_name = f'{stem}{suffix}{out_ext}'
            out_path = os.path.join(output_dir, out_name)
            try:
                write_image(img, out_path, fmt=fmt, quality=int(params.get('quality', 90)))
                self._execution_context[f'_output_{self.node_id}_path'] = out_path
            except Exception as e:
                raise ExecutionError(f"[{self.node_id}] Failed to write output to '{out_path}': {e}") from e
        return img
NODE_TYPE_MAP: Dict[str, type] = {NodeType.INPUT.value: InputNode, NodeType.GRAYSCALE.value: GrayscaleNode, NodeType.BRIGHTNESS.value: BrightnessNode, NodeType.CONTRAST.value: ContrastNode, NodeType.THRESHOLD.value: ThresholdNode, NodeType.BOX_BLUR.value: BoxBlurNode, NodeType.GAUSSIAN_BLUR.value: GaussianBlurNode, NodeType.SHARPEN.value: SharpenNode, NodeType.SOBEL.value: SobelNode, NodeType.PREWITT.value: PrewittNode, NodeType.CROP.value: CropNode, NodeType.RESIZE.value: ResizeNode, NodeType.OUTPUT.value: OutputNode}

def create_node(node_type: str, node_id: str, params: Dict[str, Any]=None) -> PipelineNode:
    cls = NODE_TYPE_MAP.get(node_type)
    if cls is None:
        raise ValidationError(f"Unknown node type: '{node_type}'")
    return cls(node_id, params)

def get_all_node_types() -> List[str]:
    return list(NODE_TYPE_MAP.keys())
