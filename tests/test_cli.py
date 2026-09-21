import os
import json
import pytest
from image_pipeline.cli.main import build_parser, main

def _run_cli(args_list):
    import io
    import sys
    parser = build_parser()
    parsed = parser.parse_args(args_list)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    try:
        rc = parsed.func(parsed)
    except SystemExit as e:
        rc = e.code
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return (rc, stdout_buf.getvalue(), stderr_buf.getvalue())

class TestBOMConfig:

    def test_load_config_with_bom_succeeds(self, bom_config_path):
        from image_pipeline.config.loader import load_config_file
        cfg = load_config_file(bom_config_path)
        assert cfg is not None
        assert cfg.name == 'test_pipeline'

    def test_validate_with_bom_config_passes(self, bom_config_path):
        rc, out, err = _run_cli(['validate', '-c', bom_config_path, '-q'])
        assert rc == 0

    def test_run_with_bom_config_works(self, bom_config_path, tmpdir_path):
        from image_pipeline.algorithms import core as alg
        from image_pipeline.utils.image_io import write_image
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'test.png'), fmt='PNG')
        rc, out, err = _run_cli(['run', '-c', bom_config_path, '-i', in_dir, '-o', out_dir, '-q', '--no-report'])
        assert rc == 0
        assert os.path.isfile(os.path.join(out_dir, 'test_out.png'))

class TestRunExitCodes:

    @pytest.fixture
    def simple_config_path(self, tmpdir_path, simple_pipeline_config):
        p = os.path.join(tmpdir_path, 'pipeline.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(simple_pipeline_config, f, indent=2)
        return p

    def test_exit_code_0_all_success(self, simple_config_path, tmpdir_path):
        from image_pipeline.algorithms import core as alg
        from image_pipeline.utils.image_io import write_image
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        write_image(alg.generate_checkerboard(10, 10, 2), os.path.join(in_dir, 'b.png'), fmt='PNG')
        rc, out, err = _run_cli(['run', '-c', simple_config_path, '-i', in_dir, '-o', out_dir, '-q', '--no-report'])
        assert rc == 0, f'Expected 0 (ALL_SUCCESS), got {rc}. stderr={err}'

    def test_exit_code_1_partial_fail(self, simple_config_path, mixed_input_dir, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        rc, out, err = _run_cli(['run', '-c', simple_config_path, '-i', mixed_input_dir, '-o', out_dir, '-q', '--no-report'])
        assert rc == 1, f'Expected 1 (PARTIAL_FAIL), got {rc}. stderr={err}'

    def test_exit_code_2_total_fail(self, simple_config_path, all_corrupt_dir, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        rc, out, err = _run_cli(['run', '-c', simple_config_path, '-i', all_corrupt_dir, '-o', out_dir, '-q', '--no-report'])
        assert rc == 2, f'Expected 2 (TOTAL_FAIL), got {rc}. stderr={err}'

    def test_exit_code_3_bad_config(self, tmpdir_path):
        bad_path = os.path.join(tmpdir_path, 'bad.json')
        with open(bad_path, 'w') as f:
            f.write('this is not json at all!!!')
        out_dir = os.path.join(tmpdir_path, 'out')
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir, exist_ok=True)
        rc, out, err = _run_cli(['run', '-c', bad_path, '-i', in_dir, '-o', out_dir, '-q', '--no-report'])
        assert rc == 3, f'Expected 3 (SETUP_ERROR), got {rc}. stderr={err}'

    def test_exit_code_3_missing_input_dir(self, simple_config_path, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        rc, out, err = _run_cli(['run', '-c', simple_config_path, '-i', os.path.join(tmpdir_path, 'no_such_dir'), '-o', out_dir, '-q', '--no-report'])
        assert rc == 3, f'Expected 3 (SETUP_ERROR), got {rc}. stderr={err}'

    def test_exit_code_3_empty_input_dir(self, simple_config_path, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'empty')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        rc, out, err = _run_cli(['run', '-c', simple_config_path, '-i', in_dir, '-o', out_dir, '-q', '--no-report'])
        assert rc == 3, f'Expected 3 (SETUP_ERROR), got {rc}. stderr={err}'

class TestValidateCommand:

    @pytest.fixture
    def valid_config_path(self, tmpdir_path, sample_data_dir):
        return os.path.join(sample_data_dir['pipeline_dir'], 'pipeline_simple.json')

    @pytest.fixture
    def cyclic_config_path(self, sample_data_dir):
        return os.path.join(sample_data_dir['pipeline_dir'], 'pipeline_cyclic_INVALID.json')

    def test_validate_valid_config_exit_0(self, valid_config_path):
        rc, out, err = _run_cli(['validate', '-c', valid_config_path, '-q'])
        assert rc == 0

    def test_validate_cyclic_config_exit_1(self, cyclic_config_path):
        rc, out, err = _run_cli(['validate', '-c', cyclic_config_path, '-q'])
        assert rc == 1

    def test_validate_output_mentions_cycle(self, cyclic_config_path):
        rc, out, err = _run_cli(['validate', '-c', cyclic_config_path])
        assert 'cycle' in out.lower() or 'cycle' in err.lower()

    def test_validate_valid_shows_execution_order(self, valid_config_path):
        rc, out, err = _run_cli(['validate', '-c', valid_config_path])
        assert 'Execution order' in out
        assert 'input' in out
        assert 'output' in out

class TestDryRunCommand:

    @pytest.fixture
    def valid_config_path(self, sample_data_dir):
        return os.path.join(sample_data_dir['pipeline_dir'], 'pipeline_simple.json')

    @pytest.fixture
    def input_dir(self, sample_data_dir):
        return sample_data_dir['input_dir']

    def test_dry_run_exit_0(self, valid_config_path, input_dir, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        rc, out, err = _run_cli(['dry-run', '-c', valid_config_path, '-i', input_dir, '-o', out_dir])
        assert rc == 0

    def test_dry_run_shows_execution_order(self, valid_config_path, input_dir, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        rc, out, err = _run_cli(['dry-run', '-c', valid_config_path, '-i', input_dir, '-o', out_dir])
        assert 'Node Execution Order' in out
        assert 'in' in out
        assert 'out' in out

    def test_dry_run_shows_predicted_outputs(self, valid_config_path, input_dir, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        rc, out, err = _run_cli(['dry-run', '-c', valid_config_path, '-i', input_dir, '-o', out_dir])
        assert 'Predicted Output Files' in out
        assert '_processed' in out

    def test_dry_run_does_not_write_outputs(self, valid_config_path, input_dir, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out_dry')
        _run_cli(['dry-run', '-c', valid_config_path, '-i', input_dir, '-o', out_dir])
        if os.path.exists(out_dir):
            files = os.listdir(out_dir)
            assert len(files) == 0, 'Dry run should not produce output images'

class TestGenerateSampleCommand:

    def test_generate_sample_creates_files(self, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'samples')
        rc, out, err = _run_cli(['generate-sample', '-o', out_dir, '-q'])
        assert rc == 0
        assert os.path.isdir(os.path.join(out_dir, 'input_images'))
        assert os.path.isdir(os.path.join(out_dir, 'pipelines'))
        assert os.path.isdir(os.path.join(out_dir, 'output'))
        img_dir = os.path.join(out_dir, 'input_images')
        images = os.listdir(img_dir)
        assert len(images) >= 3
        pipe_dir = os.path.join(out_dir, 'pipelines')
        pipes = os.listdir(pipe_dir)
        assert any(('simple' in p for p in pipes))
        assert any(('INVALID' in p for p in pipes))

    def test_generate_sample_non_quiet_prints_summary(self, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'samples2')
        rc, out, err = _run_cli(['generate-sample', '-o', out_dir])
        assert rc == 0
        assert 'Generated' in out
        assert 'sample' in out.lower()

class TestReportFile:

    @pytest.fixture
    def simple_config_path(self, tmpdir_path, simple_pipeline_config):
        p = os.path.join(tmpdir_path, 'pipeline.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(simple_pipeline_config, f, indent=2)
        return p

    def test_report_json_is_written(self, simple_config_path, tmpdir_path):
        from image_pipeline.algorithms import core as alg
        from image_pipeline.utils.image_io import write_image
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        rc, out, err = _run_cli(['run', '-c', simple_config_path, '-i', in_dir, '-o', out_dir, '-q'])
        assert rc == 0
        report_path = os.path.join(out_dir, 'batch_report.json')
        assert os.path.isfile(report_path)
        with open(report_path, 'r') as f:
            data = json.load(f)
        assert data['summary']['total'] == 1
        assert data['summary']['succeeded'] == 1
