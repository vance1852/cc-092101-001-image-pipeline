import pytest
from image_pipeline.pipeline.engine import PipelineGraph, PipelineExecutor
from image_pipeline.nodes.definitions import InputNode, OutputNode, GrayscaleNode, BrightnessNode, ThresholdNode, BoxBlurNode, ResizeNode, SobelNode, CropNode
from image_pipeline.nodes.definitions import create_node, get_all_node_types
from image_pipeline.utils.types import ValidationError

class TestGraphConstruction:

    def test_add_and_retrieve_node(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        assert g.has_node('in')
        assert 'in' in g.nodes

    def test_duplicate_node_raises(self):
        g = PipelineGraph()
        g.add_node(InputNode('same'))
        with pytest.raises(ValidationError, match='Duplicate'):
            g.add_node(GrayscaleNode('same'))

    def test_edge_to_missing_node_raises(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        with pytest.raises(ValidationError):
            g.add_edge('in', 'ghost')

    def test_edge_from_missing_node_raises(self):
        g = PipelineGraph()
        g.add_node(GrayscaleNode('gray'))
        with pytest.raises(ValidationError):
            g.add_edge('ghost', 'gray')

    def test_self_loop_raises(self):
        g = PipelineGraph()
        g.add_node(GrayscaleNode('A'))
        with pytest.raises(ValidationError):
            g.add_edge('A', 'A')

    def test_duplicate_slot_raises(self):
        g = PipelineGraph()
        g.add_node(InputNode('in1'))
        g.add_node(InputNode('in2'))
        g.add_node(GrayscaleNode('gray'))
        g.add_edge('in1', 'gray', slot=0)
        with pytest.raises(ValidationError, match='already connected'):
            g.add_edge('in2', 'gray', slot=0)

    def test_input_and_output_nodes_lists(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out'))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'out')
        inputs = g.get_input_nodes()
        outputs = g.get_output_nodes()
        assert len(inputs) == 1
        assert inputs[0].node_id == 'in'
        assert len(outputs) == 1
        assert outputs[0].node_id == 'out'

class TestCycleDetection:

    def test_acyclic_graph_no_cycle(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out'))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'out')
        assert g.detect_cycle() is None

    def test_three_node_cycle_detected(self):
        g = PipelineGraph()
        g.add_node(GrayscaleNode('A'))
        g.add_node(BrightnessNode('B', {'value': 10}))
        g.add_node(GrayscaleNode('C'))
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        cycle = g.detect_cycle()
        assert cycle is not None
        cycle_set = set(cycle)
        assert {'A', 'B', 'C'}.issubset(cycle_set)

    def test_cycle_alongside_valid_branch(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out'))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'out')
        g.add_node(GrayscaleNode('X'))
        g.add_node(BrightnessNode('Y', {'value': 1}))
        g.add_node(GrayscaleNode('Z'))
        g.add_edge('X', 'Y')
        g.add_edge('Y', 'Z')
        g.add_edge('Z', 'X')
        cycle = g.detect_cycle()
        assert cycle is not None

    def test_topological_sort_raises_on_cycle(self):
        g = PipelineGraph()
        g.add_node(GrayscaleNode('A'))
        g.add_node(BrightnessNode('B', {'value': 5}))
        g.add_node(GrayscaleNode('C'))
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        with pytest.raises(ValidationError, match='cycle'):
            g.topological_order()

class TestTopologicalOrder:

    def test_simple_chain_order(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(ThresholdNode('thresh'))
        g.add_node(OutputNode('out'))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'thresh')
        g.add_edge('thresh', 'out')
        order = g.topological_order()
        idx = {nid: i for i, nid in enumerate(order)}
        assert idx['in'] < idx['gray'] < idx['thresh'] < idx['out']

    def test_diamond_branch_order(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(BoxBlurNode('blur', {'size': 3}))
        g.add_node(BrightnessNode('bright', {'value': 10}))
        g.add_node(OutputNode('out1'))
        g.add_node(OutputNode('out2'))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'blur')
        g.add_edge('gray', 'bright')
        g.add_edge('blur', 'out1')
        g.add_edge('bright', 'out2')
        order = g.topological_order()
        idx = {nid: i for i, nid in enumerate(order)}
        assert idx['gray'] < idx['blur']
        assert idx['gray'] < idx['bright']
        assert idx['in'] < idx['gray']
        assert idx['blur'] < idx['out1']
        assert idx['bright'] < idx['out2']

    def test_executor_has_order(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out'))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'out')
        executor = PipelineExecutor(g)
        assert len(executor.execution_order) == 3

    def test_deterministic_order(self):

        def build_and_order():
            g = PipelineGraph()
            g.add_node(InputNode('in'))
            g.add_node(GrayscaleNode('b'))
            g.add_node(GrayscaleNode('a'))
            g.add_node(GrayscaleNode('c'))
            g.add_edge('in', 'a')
            g.add_edge('in', 'b')
            g.add_edge('in', 'c')
            return g.topological_order()
        order1 = build_and_order()
        order2 = build_and_order()
        assert order1 == order2

class TestGraphValidation:

    def test_valid_graph_passes(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out'))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'out')
        result = g.validate()
        assert result.valid

    def test_empty_graph_fails(self):
        g = PipelineGraph()
        result = g.validate()
        assert not result.valid
        assert any(('empty' in str(i).lower() for i in result.errors))

    def test_no_input_node_fails(self):
        g = PipelineGraph()
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out'))
        g.add_edge('gray', 'out')
        result = g.validate()
        assert not result.valid
        assert any(('input' in str(i).lower() for i in result.errors))

    def test_dangling_input_slot_fails(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        result = g.validate()
        assert not result.valid
        issues = [str(i) for i in result.errors]
        assert any(('unconnected' in i.lower() for i in issues))

    def test_no_output_node_warns(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_edge('in', 'gray')
        result = g.validate()
        assert result.valid
        assert len(result.warnings) > 0
        assert any(('output' in str(w).lower() for w in result.warnings))

    def test_cyclic_graph_fails_validation(self):
        g = PipelineGraph()
        g.add_node(GrayscaleNode('A'))
        g.add_node(BrightnessNode('B', {'value': 1}))
        g.add_node(GrayscaleNode('C'))
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        result = g.validate()
        assert not result.valid
        assert any(('cycle' in str(i).lower() for i in result.errors))

class TestNodeParamValidation:

    def test_resize_missing_params_fails(self):
        node = ResizeNode('r1', {})
        result = node.validate_params()
        assert not result.valid

    def test_resize_invalid_mode_fails(self):
        from image_pipeline.nodes.definitions import ThresholdNode
        node = ThresholdNode('t1', {'mode': 'weird'})
        result = node.validate_params()
        assert not result.valid

    def test_box_blur_even_size_fails(self):
        node = BoxBlurNode('b1', {'size': 4})
        result = node.validate_params()
        assert not result.valid
        assert any(('odd' in str(i).lower() for i in result.errors))

    def test_sobel_invalid_direction_fails(self):
        node = SobelNode('s1', {'direction': 'diagonal'})
        result = node.validate_params()
        assert not result.valid

    def test_crop_missing_required_fails(self):
        node = CropNode('c1', {})
        result = node.validate_params()
        assert not result.valid

    def test_unknown_node_type_raises(self):
        with pytest.raises(Exception):
            create_node('nonexistent_type', 'id1', {})

    def test_get_all_node_types_not_empty(self):
        types = get_all_node_types()
        assert len(types) > 0
        assert 'input' in types
        assert 'output' in types
        assert 'grayscale' in types
