from typing import Any, Dict, List, Tuple, Optional
import json
import os
from ..pipeline.engine import PipelineGraph, PipelineExecutor
from ..nodes.definitions import PipelineNode, create_node, get_all_node_types, NODE_TYPE_MAP
from ..utils.types import ValidationResult, ValidationError, NodeType

class PipelineConfig:

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        self.version = raw.get('version', '1.0')
        self.name = raw.get('name', 'unnamed_pipeline')
        self.description = raw.get('description', '')
        self.defaults = raw.get('defaults', {}) or {}
        self.graph: Optional[PipelineGraph] = None
        self.executor: Optional[PipelineExecutor] = None

    def build_graph(self) -> Tuple[PipelineGraph, ValidationResult]:
        result = ValidationResult()
        graph = PipelineGraph()
        nodes_raw = self.raw.get('nodes', [])
        edges_raw = self.raw.get('edges', [])
        if not isinstance(nodes_raw, list):
            result.add_error("'nodes' must be a list")
            return (graph, result)
        if not isinstance(edges_raw, list):
            result.add_error("'edges' must be a list")
            return (graph, result)
        if len(nodes_raw) == 0:
            result.add_error("Pipeline has no nodes (empty 'nodes' list)")
            return (graph, result)
        seen_ids = set()
        for idx, node_cfg in enumerate(nodes_raw):
            if not isinstance(node_cfg, dict):
                result.add_error(f'nodes[{idx}] must be an object')
                continue
            nid = node_cfg.get('id')
            ntype = node_cfg.get('type')
            nparams = node_cfg.get('params', {}) or {}
            if not nid or not isinstance(nid, str):
                result.add_error(f"nodes[{idx}] missing or invalid 'id' (must be non-empty string)")
                continue
            if nid in seen_ids:
                result.add_error(f"nodes[{idx}]: duplicate node id '{nid}'", nid)
                continue
            seen_ids.add(nid)
            if not ntype or not isinstance(ntype, str):
                result.add_error(f"nodes[{idx}] missing or invalid 'type' (must be string)", nid)
                continue
            if ntype not in NODE_TYPE_MAP:
                result.add_error(f"nodes[{idx}] unknown type '{ntype}'. Allowed types: {sorted(get_all_node_types())}", nid, 'type')
                continue
            if not isinstance(nparams, dict):
                result.add_error(f"nodes[{idx}] 'params' must be an object", nid, 'params')
                nparams = {}
            try:
                node = create_node(ntype, nid, nparams)
                graph.add_node(node)
            except ValidationError as e:
                result.add_error(str(e), nid)
            except Exception as e:
                result.add_error(f"Failed to create node '{nid}': {e}", nid)
        for idx, edge_cfg in enumerate(edges_raw):
            if not isinstance(edge_cfg, dict):
                result.add_error(f'edges[{idx}] must be an object')
                continue
            src = edge_cfg.get('from')
            tgt = edge_cfg.get('to')
            slot = edge_cfg.get('slot', 0)
            if not src or not isinstance(src, str):
                result.add_error(f"edges[{idx}] missing or invalid 'from' (source node id)")
                continue
            if not tgt or not isinstance(tgt, str):
                result.add_error(f"edges[{idx}] missing or invalid 'to' (target node id)")
                continue
            if not isinstance(slot, int) or slot < 0:
                result.add_error(f"edges[{idx}] 'slot' must be non-negative integer")
                continue
            try:
                graph.add_edge(src, tgt, slot)
            except ValidationError as e:
                result.add_error(f'edges[{idx}]: {e}')
        graph_validation = graph.validate()
        result.extend(graph_validation)
        self.graph = graph
        return (graph, result)

    def build_executor(self) -> Tuple[Optional[PipelineExecutor], ValidationResult]:
        if self.graph is None:
            _, result = self.build_graph()
            if not result.valid:
                return (None, result)
        else:
            result = self.graph.validate()
            if not result.valid:
                return (None, result)
        try:
            executor = PipelineExecutor(self.graph)
            self.executor = executor
            return (executor, result)
        except ValidationError as e:
            result.add_error(str(e))
            return (None, result)

def load_config_file(path: str) -> PipelineConfig:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Pipeline config file not found: {path}')
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in pipeline config '{path}': {e}") from e
    except Exception as e:
        raise ValidationError(f"Failed to read pipeline config '{path}': {e}") from e
    if not isinstance(raw, dict):
        raise ValidationError(f'Pipeline config must be a JSON object (dict), got {type(raw).__name__}')
    return PipelineConfig(raw)

def validate_config_file(path: str) -> ValidationResult:
    try:
        cfg = load_config_file(path)
    except FileNotFoundError as e:
        result = ValidationResult()
        result.add_error(str(e))
        return result
    except ValidationError as e:
        result = ValidationResult()
        result.add_error(str(e))
        return result
    _, result = cfg.build_graph()
    try:
        cfg.build_executor()
    except ValidationError as e:
        result.add_error(str(e))
    return result

def sample_pipeline_config() -> Dict[str, Any]:
    return {'version': '1.0', 'name': 'sample_grayscale_threshold_pipeline', 'description': 'Example: convert to grayscale, threshold at 128, light box blur, write as PNG', 'nodes': [{'id': 'in', 'type': 'input'}, {'id': 'gray', 'type': 'grayscale'}, {'id': 'thresh', 'type': 'threshold', 'params': {'value': 128, 'mode': 'binary'}}, {'id': 'blur', 'type': 'box_blur', 'params': {'size': 3}}, {'id': 'out', 'type': 'output', 'params': {'format': 'PNG', 'suffix': '_processed'}}], 'edges': [{'from': 'in', 'to': 'gray'}, {'from': 'gray', 'to': 'thresh'}, {'from': 'thresh', 'to': 'blur'}, {'from': 'blur', 'to': 'out'}], 'defaults': {'output_format': 'PNG'}}

def sample_complex_pipeline_config() -> Dict[str, Any]:
    return {'version': '1.0', 'name': 'sample_complex_pipeline', 'description': 'Example demonstrating parallel graph: grayscale feeds into sobel edge detection and also a brightness/contrast path. (Each branch writes to its own output.)', 'nodes': [{'id': 'in', 'type': 'input'}, {'id': 'gray', 'type': 'grayscale'}, {'id': 'sobel', 'type': 'sobel', 'params': {'direction': 'both'}}, {'id': 'bright', 'type': 'brightness', 'params': {'value': 20}}, {'id': 'contrast', 'type': 'contrast', 'params': {'value': 1.5}}, {'id': 'edges_out', 'type': 'output', 'params': {'format': 'PNG', 'suffix': '_edges'}}, {'id': 'enhanced_out', 'type': 'output', 'params': {'format': 'JPEG', 'quality': 90, 'suffix': '_enhanced'}}], 'edges': [{'from': 'in', 'to': 'gray'}, {'from': 'gray', 'to': 'sobel'}, {'from': 'sobel', 'to': 'edges_out'}, {'from': 'gray', 'to': 'bright'}, {'from': 'bright', 'to': 'contrast'}, {'from': 'contrast', 'to': 'enhanced_out'}], 'defaults': {}}

def cycle_pipeline_config() -> Dict[str, Any]:
    return {'version': '1.0', 'name': 'invalid_cyclic_pipeline', 'description': 'This pipeline contains a cycle and should be rejected.', 'nodes': [{'id': 'in', 'type': 'input'}, {'id': 'gray', 'type': 'grayscale'}, {'id': 'loopA', 'type': 'grayscale'}, {'id': 'loopB', 'type': 'brightness', 'params': {'value': 10}}, {'id': 'loopC', 'type': 'contrast', 'params': {'value': 1.1}}, {'id': 'out', 'type': 'output'}], 'edges': [{'from': 'in', 'to': 'gray'}, {'from': 'gray', 'to': 'out'}, {'from': 'loopA', 'to': 'loopB'}, {'from': 'loopB', 'to': 'loopC'}, {'from': 'loopC', 'to': 'loopA'}]}

def invalid_params_pipeline_config() -> Dict[str, Any]:
    return {'version': '1.0', 'name': 'invalid_params_pipeline', 'description': 'Contains multiple parameter errors for validation testing.', 'nodes': [{'id': 'in', 'type': 'input'}, {'id': 'bad_blur', 'type': 'box_blur', 'params': {'size': 4}}, {'id': 'bad_resize', 'type': 'resize', 'params': {'scale': -1.0}}, {'id': 'bad_crop', 'type': 'crop'}, {'id': 'out', 'type': 'output'}], 'edges': [{'from': 'in', 'to': 'bad_blur'}, {'from': 'bad_blur', 'to': 'bad_resize'}, {'from': 'bad_resize', 'to': 'bad_crop'}, {'from': 'bad_crop', 'to': 'out'}]}
