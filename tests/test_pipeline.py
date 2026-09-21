import os
import json
import pytest
from image_pipeline.pipeline.engine import PipelineGraph, PipelineExecutor
from image_pipeline.nodes.definitions import InputNode, OutputNode, GrayscaleNode, BrightnessNode, ThresholdNode, BoxBlurNode, SobelNode
from image_pipeline.config.loader import PipelineConfig, load_config_file, validate_config_file, sample_pipeline_config
from image_pipeline.batch.executor import BatchExecutor, print_text_report
from image_pipeline.utils.image_io import write_image, is_valid_image
from image_pipeline.algorithms import core as alg

class TestPipelineExecutor:

    def _build_simple_graph(self):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(ThresholdNode('thresh', {'value': 128}))
        g.add_node(OutputNode('out', {'suffix': '_result'}))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'thresh')
        g.add_edge('thresh', 'out')
        return g

    def test_executor_runs_successfully(self, tmpdir_path):
        g = self._build_simple_graph()
        executor = PipelineExecutor(g)
        img_path = os.path.join(tmpdir_path, 'test.png')
        img = alg.generate_gradient_image(8, 8)
        write_image(img, img_path, fmt='PNG')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        result = executor.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out_dir})
        assert result.success
        assert result.output_path is not None
        assert os.path.isfile(result.output_path)

    def test_executor_per_node_results(self, tmpdir_path):
        g = self._build_simple_graph()
        executor = PipelineExecutor(g)
        img_path = os.path.join(tmpdir_path, 'test.png')
        write_image(alg.generate_gradient_image(6, 6), img_path, fmt='PNG')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        result = executor.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out_dir})
        assert len(result.node_results) == 4
        for nr in result.node_results:
            assert nr.success
            assert nr.duration_ms >= 0

    def test_deterministic_output(self, tmpdir_path):
        g = self._build_simple_graph()
        exec1 = PipelineExecutor(g)
        exec2 = PipelineExecutor(g)
        img_path = os.path.join(tmpdir_path, 'test.png')
        write_image(alg.generate_checkerboard(16, 16, 4), img_path, fmt='PNG')
        out1 = os.path.join(tmpdir_path, 'out1')
        out2 = os.path.join(tmpdir_path, 'out2')
        os.makedirs(out1, exist_ok=True)
        os.makedirs(out2, exist_ok=True)
        r1 = exec1.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out1})
        r2 = exec2.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out2})
        assert r1.success and r2.success
        from image_pipeline.utils.image_io import read_image
        img1 = read_image(r1.output_path)
        img2 = read_image(r2.output_path)
        assert img1 == img2

    def test_cached_upstream_reuse(self, tmpdir_path):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(SobelNode('sobel', {'direction': 'x'}))
        g.add_node(BrightnessNode('bright', {'value': 20}))
        g.add_node(OutputNode('out1', {'suffix': '_sobel'}))
        g.add_node(OutputNode('out2', {'suffix': '_bright'}))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'sobel')
        g.add_edge('gray', 'bright')
        g.add_edge('sobel', 'out1')
        g.add_edge('bright', 'out2')
        executor = PipelineExecutor(g)
        order = executor.execution_order
        assert order.count('gray') == 1
        img_path = os.path.join(tmpdir_path, 'test.png')
        write_image(alg.generate_gradient_image(8, 8), img_path, fmt='PNG')
        out_dir = os.path.join(tmpdir_path, 'out_branch')
        os.makedirs(out_dir, exist_ok=True)
        result = executor.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out_dir})
        assert result.success
        assert len(result.node_results) == 6
        assert os.path.isfile(os.path.join(out_dir, 'test_sobel.png'))
        assert os.path.isfile(os.path.join(out_dir, 'test_bright.png'))

    def test_missing_input_file_raises(self, tmpdir_path):
        g = self._build_simple_graph()
        executor = PipelineExecutor(g)
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        result = executor.run({'input_path': '/nonexistent/file.png', 'input_filename': 'file.png', 'output_dir': out_dir})
        assert not result.success
        assert result.error is not None

class TestBatchExecutor:

    def _make_simple_pipeline(self):
        cfg = PipelineConfig(sample_pipeline_config())
        executor, _ = cfg.build_executor()
        return executor

    def test_batch_all_good_images(self, tmpdir_path):
        executor = self._make_simple_pipeline()
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        for i in range(3):
            img = alg.generate_gradient_image(8 + i, 8 + i)
            write_image(img, os.path.join(in_dir, f'img{i}.png'), fmt='PNG')
        batch = BatchExecutor(executor, in_dir, out_dir, 'test_pipeline')
        report = batch.run()
        assert report.total == 3
        assert report.succeeded == 3
        assert report.failed == 0
        assert len(report.results) == 3

    def test_batch_with_corrupt_images(self, mixed_input_dir, tmpdir_path):
        executor = self._make_simple_pipeline()
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        batch = BatchExecutor(executor, mixed_input_dir, out_dir)
        report = batch.run()
        assert report.total == 3
        assert report.succeeded == 2
        assert report.failed == 1
        corrupt_results = [r for r in report.results if 'bad.png' in r.input_path]
        assert len(corrupt_results) == 1
        assert not corrupt_results[0].success
        assert corrupt_results[0].error is not None

    def test_batch_report_json_serializable(self, tmpdir_path):
        executor = self._make_simple_pipeline()
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'test.png'), fmt='PNG')
        batch = BatchExecutor(executor, in_dir, out_dir)
        report = batch.run()
        data = report.to_dict()
        json_str = json.dumps(data, indent=2)
        assert 'summary' in data
        assert 'results' in data
        assert data['summary']['total'] == 1

    def test_batch_writes_report_file(self, tmpdir_path):
        executor = self._make_simple_pipeline()
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'test.png'), fmt='PNG')
        batch = BatchExecutor(executor, in_dir, out_dir)
        report = batch.run()
        path = batch.write_report(report)
        assert os.path.isfile(path)
        with open(path, 'r') as f:
            data = json.load(f)
        assert data['summary']['total'] == 1

    def test_text_report_contains_summary(self, tmpdir_path):
        executor = self._make_simple_pipeline()
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(6, 6), os.path.join(in_dir, 'a.png'), fmt='PNG')
        batch = BatchExecutor(executor, in_dir, out_dir)
        report = batch.run()
        text = print_text_report(report)
        assert 'Total images' in text
        assert 'Succeeded' in text
        assert '1' in text

    def test_missing_input_dir_raises(self, tmpdir_path):
        from image_pipeline.utils.types import ValidationError
        executor = self._make_simple_pipeline()
        batch = BatchExecutor(executor, os.path.join(tmpdir_path, 'does_not_exist'), os.path.join(tmpdir_path, 'out'))
        report = batch.run()
        assert len(report.results) == 1
        assert not report.results[0].success

    def test_empty_input_dir(self, tmpdir_path):
        executor = self._make_simple_pipeline()
        in_dir = os.path.join(tmpdir_path, 'in_empty')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        batch = BatchExecutor(executor, in_dir, out_dir)
        report = batch.run()
        assert report.total == 0
        assert report.succeeded == 0
        assert report.failed == 0
