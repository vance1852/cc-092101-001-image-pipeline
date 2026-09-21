from typing import Any, Dict, List, Optional, Set, Tuple
import time
from collections import deque
from ..nodes.definitions import PipelineNode, OutputNode
from ..utils.types import NodeType, ValidationResult, ValidationError, ExecutionError, NodeExecutionResult, ImageProcessingResult
from ..algorithms.core import Image

class PipelineGraph:

    def __init__(self):
        self.nodes: Dict[str, PipelineNode] = {}
        self.upstream: Dict[str, List[Tuple[str, int]]] = {}
        self.downstream: Dict[str, List[str]] = {}

    def add_node(self, node: PipelineNode) -> None:
        nid = node.node_id
        if nid in self.nodes:
            raise ValidationError(f"Duplicate node ID: '{nid}'")
        self.nodes[nid] = node
        self.upstream.setdefault(nid, [])
        self.downstream.setdefault(nid, [])

    def add_edge(self, source_id: str, target_id: str, slot: int=0) -> None:
        if source_id not in self.nodes:
            raise ValidationError(f"Edge source node not found: '{source_id}'")
        if target_id not in self.nodes:
            raise ValidationError(f"Edge target node not found: '{target_id}'")
        if source_id == target_id:
            raise ValidationError(f"Self-loop not allowed on node '{source_id}'")
        target_node = self.nodes[target_id]
        if slot < 0 or slot >= target_node.input_count:
            raise ValidationError(f"Invalid input slot {slot} for node '{target_id}' (has {target_node.input_count} inputs)")
        for existing_src, existing_slot in self.upstream.get(target_id, []):
            if existing_slot == slot:
                raise ValidationError(f"Node '{target_id}' input slot {slot} is already connected to '{existing_src}'")
        self.upstream[target_id].append((source_id, slot))
        self.downstream[source_id].append(target_id)

    def get_input_nodes(self) -> List[PipelineNode]:
        return [self.nodes[nid] for nid in self.nodes if not self.upstream.get(nid)]

    def get_output_nodes(self) -> List[OutputNode]:
        result = []
        for nid, node in self.nodes.items():
            if isinstance(node, OutputNode):
                result.append(node)
        return result

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def detect_cycle(self) -> Optional[List[str]]:
        WHITE, GRAY, BLACK = (0, 1, 2)
        color = {nid: WHITE for nid in self.nodes}
        parent = {}
        cycle_found = []

        def _dfs(u: str, path: list) -> bool:
            color[u] = GRAY
            path.append(u)
            for v in self.downstream.get(u, []):
                if color[v] == GRAY:
                    if v in path:
                        idx = path.index(v)
                        cycle_found.extend(path[idx:] + [v])
                    else:
                        cycle_found.append(v)
                    return True
                elif color[v] == WHITE:
                    parent[v] = u
                    if _dfs(v, path):
                        return True
            path.pop()
            color[u] = BLACK
            return False
        for nid in sorted(self.nodes.keys()):
            if color[nid] == WHITE:
                if _dfs(nid, []):
                    return cycle_found
        return None

    def topological_order(self) -> List[str]:
        in_degree: Dict[str, int] = {nid: len(self.upstream.get(nid, [])) for nid in self.nodes}
        queue = deque(sorted([nid for nid, deg in in_degree.items() if deg == 0]))
        result = []
        while queue:
            u = queue.popleft()
            result.append(u)
            for v in sorted(self.downstream.get(u, [])):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        if len(result) != len(self.nodes):
            cycle = self.detect_cycle()
            cycle_msg = f" Cycle: {' -> '.join(cycle)}" if cycle else ''
            raise ValidationError(f'Graph contains a cycle - topological sort impossible.{cycle_msg}')
        return result

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        if not self.nodes:
            result.add_error('Pipeline graph is empty (no nodes)')
            return result
        input_nodes = self.get_input_nodes()
        if not input_nodes:
            result.add_error("Pipeline has no source nodes (nodes with no inputs). At least one 'input' node is required.")
        explicit_inputs = [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]
        if not explicit_inputs:
            result.add_error("Pipeline has no 'input' type nodes. At least one input node is required.")
        output_nodes = self.get_output_nodes()
        if not output_nodes:
            result.add_warning("Pipeline has no 'output' type nodes. Processed results will not be written to disk.")
        try:
            order = self.topological_order()
        except ValidationError as e:
            result.add_error(str(e))
            order = []
        for nid, node in self.nodes.items():
            if node.node_type == NodeType.INPUT:
                continue
            needed = node.input_count
            connected_slots = set((slot for _, slot in self.upstream.get(nid, [])))
            missing = [s for s in range(needed) if s not in connected_slots]
            if missing:
                result.add_error(f"Node '{nid}' has unconnected input slots: {missing}", nid)
        for nid, node in self.nodes.items():
            node_result = node.validate_params()
            for issue in node_result.issues:
                if issue.node_id is None:
                    issue.node_id = nid
            result.extend(node_result)
        return result

class PipelineExecutor:

    def __init__(self, graph: PipelineGraph):
        self.graph = graph
        self._order = graph.topological_order()

    @property
    def execution_order(self) -> List[str]:
        return list(self._order)

    def run(self, context: Dict[str, Any]=None) -> ImageProcessingResult:
        context = context or {}
        overall_start = time.perf_counter()
        img_result = ImageProcessingResult(input_path=context.get('input_path', ''))
        cache: Dict[str, Image] = {}
        try:
            for nid in self._order:
                node = self.graph.nodes[nid]
                node.set_context(**context)
                upstream = self.graph.upstream.get(nid, [])
                upstream_sorted = sorted(upstream, key=lambda x: x[1])
                inputs: List[Image] = []
                for src_id, slot in upstream_sorted:
                    if src_id not in cache:
                        raise ExecutionError(f"Node '{nid}' depends on '{src_id}' which has not been computed (topological order error)")
                    inputs.append(cache[src_id])
                node_start = time.perf_counter()
                node_result = NodeExecutionResult(node_id=nid, node_type=node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type), success=False)
                try:
                    output_img = node.execute(inputs)
                    cache[nid] = output_img
                    if output_img and output_img[0]:
                        node_result.output_size = (len(output_img[0]), len(output_img))
                    if isinstance(node, OutputNode):
                        out_path = node._execution_context.get(f'_output_{nid}_path')
                        if out_path and img_result.output_path is None:
                            img_result.output_path = out_path
                    node_result.success = True
                except Exception as e:
                    node_result.error = str(e)
                    raise ExecutionError(f"Node '{nid}' execution failed: {e}") from e
                finally:
                    node_result.duration_ms = (time.perf_counter() - node_start) * 1000.0
                    img_result.node_results.append(node_result)
                    node.clear_context()
            img_result.success = True
        except ExecutionError as e:
            img_result.success = False
            img_result.error = str(e)
        except Exception as e:
            img_result.success = False
            img_result.error = f'Unexpected error: {e}'
        finally:
            for nid in self._order:
                self.graph.nodes[nid].clear_context()
            img_result.duration_ms = (time.perf_counter() - overall_start) * 1000.0
        return img_result

    def clear_cache(self):
        pass
