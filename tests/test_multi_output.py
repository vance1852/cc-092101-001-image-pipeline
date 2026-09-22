import io
import json
import os
import sys
import pytest

from image_pipeline.algorithms import core as alg
from image_pipeline.utils.image_io import write_image
from image_pipeline.config.loader import PipelineConfig
from image_pipeline.batch.executor import BatchExecutor
from image_pipeline.batch import executor as exec_mod
from image_pipeline.cli.main import build_parser
from image_pipeline.utils.types import (
    STATUS_COMPLETE,
    STATUS_PREFLIGHT_REJECTED,
    STATUS_ROLLED_BACK,
)


def _multi_config(suffix_a='_edges', suffix_b='_thumb', fmt_a='PNG', fmt_b='JPEG'):
    return {
        'version': '1.0',
        'name': 'multi_out',
        'nodes': [
            {'id': 'in', 'type': 'input'},
            {'id': 'gray', 'type': 'grayscale'},
            {'id': 'edges_out', 'type': 'output',
             'params': {'suffix': suffix_a, 'format': fmt_a}},
            {'id': 'thumb_out', 'type': 'output',
             'params': {'suffix': suffix_b, 'format': fmt_b}},
        ],
        'edges': [
            {'from': 'in', 'to': 'gray'},
            {'from': 'gray', 'to': 'edges_out'},
            {'from': 'gray', 'to': 'thumb_out'},
        ],
    }


def _single_output_config(fmt='PNG', suffix=''):
    return {
        'version': '1.0',
        'name': 'single_out',
        'nodes': [
            {'id': 'in', 'type': 'input'},
            {'id': 'o', 'type': 'output', 'params': {'format': fmt, 'suffix': suffix}},
        ],
        'edges': [{'from': 'in', 'to': 'o'}],
    }


def _build(config):
    cfg = PipelineConfig(config)
    executor, validation = cfg.build_executor()
    assert validation.valid, validation.errors
    return executor


def _run_cli(args_list):
    parser = build_parser()
    parsed = parser.parse_args(args_list)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        rc = parsed.func(parsed)
        out, err = sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return rc, out, err


class TestMultiOutputArtifacts:

    def test_each_output_node_gets_independent_artifact(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        batch = BatchExecutor(_build(_multi_config()), in_dir, out_dir)
        report = batch.run()
        assert report.status == STATUS_COMPLETE
        result = report.results[0]
        assert result.success
        assert len(result.outputs) == 2
        by_node = {o.node_id: o for o in result.outputs}
        assert set(by_node) == {'edges_out', 'thumb_out'}
        assert by_node['edges_out'].path.endswith('a_edges.png')
        assert by_node['thumb_out'].path.endswith('a_thumb.jpg')
        assert os.path.isfile(by_node['edges_out'].path)
        assert os.path.isfile(by_node['thumb_out'].path)
        assert by_node['edges_out'].size_bytes > 0
        assert by_node['thumb_out'].size_bytes > 0
        assert by_node['edges_out'].checksum != by_node['thumb_out'].checksum
        assert by_node['edges_out'].image_size == (8, 8)
        # backwards-compatible scalar output_path points at the first artifact
        assert result.output_path == result.outputs[0].path

    def test_json_report_lists_all_outputs_with_identity(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        batch = BatchExecutor(_build(_multi_config()), in_dir, out_dir)
        report = batch.run()
        path = batch.write_report(report)
        data = json.load(open(path, encoding='utf-8'))
        assert data['status'] == 'complete'
        outs = data['results'][0]['outputs']
        assert {o['node_id'] for o in outs} == {'edges_out', 'thumb_out'}
        assert all(o['size_bytes'] > 0 and o['path'] for o in outs)


class TestWithinImageConflict:

    def test_conflicting_nodes_rejected_before_writes(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        batch = BatchExecutor(
            _build(_multi_config(suffix_a='_x', suffix_b='_x', fmt_a='PNG', fmt_b='PNG')),
            in_dir, out_dir,
        )
        report = batch.run()
        assert report.status == STATUS_PREFLIGHT_REJECTED
        assert report.skipped == 1
        assert any('Within-image conflict' in c for c in report.conflicts)
        # nothing at all was written for the rejected batch
        produced = [f for f in os.listdir(out_dir)]
        assert not any(f.endswith('.png') for f in produced)
        assert report.results[0].status == STATUS_PREFLIGHT_REJECTED
        path = batch.write_report(report)
        data = json.load(open(path, encoding='utf-8'))
        assert data['status'] == 'preflight_rejected'
        assert data['conflicts']

    def test_cli_exit_code_4_for_preflight_reject(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        cfg_path = os.path.join(tmpdir_path, 'p.json')
        json.dump(_multi_config(suffix_a='_x', suffix_b='_x', fmt_a='PNG', fmt_b='PNG'),
                  open(cfg_path, 'w'))
        rc, out, err = _run_cli(['run', '-c', cfg_path, '-i', in_dir, '-o', out_dir, '-q'])
        assert rc == 4
        assert 'PREFLIGHT CONFLICT' in err

    def test_dry_run_predicts_conflict(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        cfg_path = os.path.join(tmpdir_path, 'p.json')
        json.dump(_multi_config(suffix_a='_x', suffix_b='_x', fmt_a='PNG', fmt_b='PNG'),
                  open(cfg_path, 'w'))
        rc, out, err = _run_cli(['dry-run', '-c', cfg_path, '-i', in_dir, '-o', out_dir])
        assert rc == 0
        assert 'PREDICTED TARGET CONFLICTS' in out


class TestCrossInputConflict:

    def test_different_source_extensions_forced_to_same_target(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        write_image(alg.generate_checkerboard(8, 8, 2), os.path.join(in_dir, 'a.jpg'), fmt='JPEG')
        batch = BatchExecutor(_build(_single_output_config(fmt='PNG')), in_dir, out_dir)
        report = batch.run()
        assert report.status == STATUS_PREFLIGHT_REJECTED
        assert any('Cross-input conflict' in c for c in report.conflicts)
        assert not os.path.isfile(os.path.join(out_dir, 'a.png'))
        assert report.skipped == 2

    def test_distinct_stems_do_not_conflict(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        write_image(alg.generate_checkerboard(8, 8, 2), os.path.join(in_dir, 'b.png'), fmt='PNG')
        batch = BatchExecutor(_build(_single_output_config(fmt='PNG')), in_dir, out_dir)
        report = batch.run()
        assert report.status == STATUS_COMPLETE
        assert os.path.isfile(os.path.join(out_dir, 'a.png'))
        assert os.path.isfile(os.path.join(out_dir, 'b.png'))


class TestAtomicRollback:

    def test_commit_failure_restores_previous_outputs(self, tmpdir_path, monkeypatch):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        batch = BatchExecutor(_build(_multi_config()), in_dir, out_dir)
        assert batch.run().status == STATUS_COMPLETE
        old_edges = open(os.path.join(out_dir, 'a_edges.png'), 'rb').read()
        old_thumb = open(os.path.join(out_dir, 'a_thumb.jpg'), 'rb').read()

        real_commit = exec_mod.RecoverableCommit.commit_pending

        def flaky(self):
            # install the first replacement, then simulate a disk failure
            import os as _os
            entry = self._entries[0]
            _os.replace(self._journal_path(entry['staged']),
                        _os.path.join(self.output_dir, entry['dest']))
            open(self._done_marker(entry['idx']), 'wb').close()
            raise OSError('simulated disk full')

        monkeypatch.setattr(exec_mod.RecoverableCommit, 'commit_pending', flaky)
        report = batch.run()
        assert report.status == STATUS_ROLLED_BACK
        result = report.results[0]
        assert not result.success
        assert result.outputs == []
        # previous valid files are byte-for-byte intact
        assert open(os.path.join(out_dir, 'a_edges.png'), 'rb').read() == old_edges
        assert open(os.path.join(out_dir, 'a_thumb.jpg'), 'rb').read() == old_thumb
        # no journal leaks
        assert not [f for f in os.listdir(out_dir) if f.startswith('.imgpipe-journal-')]
        monkeypatch.undo()
        # safe rerun after recovery
        assert batch.run().status == STATUS_COMPLETE

    def test_cli_exit_code_5_on_rollback(self, tmpdir_path, monkeypatch):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        cfg_path = os.path.join(tmpdir_path, 'p.json')
        json.dump(_multi_config(), open(cfg_path, 'w'))
        rc, _, _ = _run_cli(['run', '-c', cfg_path, '-i', in_dir, '-o', out_dir, '-q'])
        assert rc == 0

        def flaky_commit(self):
            raise OSError('simulated disk full')

        monkeypatch.setattr(exec_mod.RecoverableCommit, 'commit_pending', flaky_commit)
        rc, _, _ = _run_cli(['run', '-c', cfg_path, '-i', in_dir, '-o', out_dir, '-q', '--no-report'])
        assert rc == 5

    def test_commit_failure_across_multiple_images_leaves_no_half_set(self, tmpdir_path, monkeypatch):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        for name in ('a.png', 'b.png', 'c.png'):
            write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, name), fmt='PNG')
        batch = BatchExecutor(_build(_multi_config()), in_dir, out_dir)
        assert batch.run().status == STATUS_COMPLETE
        old = {
            name: open(os.path.join(out_dir, name), 'rb').read()
            for name in os.listdir(out_dir) if name.endswith(('.png', '.jpg'))
        }

        # Fail after installing some (but not all) batch outputs.
        def flaky_commit(self):
            import os as _os
            for entry in self._entries[:2]:
                _os.replace(self._journal_path(entry['staged']),
                            _os.path.join(self.output_dir, entry['dest']))
                open(self._done_marker(entry['idx']), 'wb').close()
            raise OSError('simulated disk full')

        monkeypatch.setattr(exec_mod.RecoverableCommit, 'commit_pending', flaky_commit)
        report = batch.run()
        assert report.status == STATUS_ROLLED_BACK
        assert report.succeeded == 0
        # Every image's receipt is marked rolled back and claims no files
        assert all(r.status == STATUS_ROLLED_BACK for r in report.results)
        assert all(r.outputs == [] for r in report.results)
        # Disk holds exactly the previous run's byte-for-byte files
        current = {
            name: open(os.path.join(out_dir, name), 'rb').read()
            for name in os.listdir(out_dir) if name.endswith(('.png', '.jpg'))
        }
        assert current == old
        assert not [f for f in os.listdir(out_dir) if f.startswith('.imgpipe-journal-')]
        monkeypatch.undo()
        assert batch.run().status == STATUS_COMPLETE

    def test_stale_journal_is_recovered_on_next_run(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        batch = BatchExecutor(_build(_multi_config()), in_dir, out_dir)
        batch.run()
        good = open(os.path.join(out_dir, 'a_edges.png'), 'rb').read()
        # fabricate a crashed mid-commit journal
        import shutil
        jdir = os.path.join(out_dir, '.imgpipe-journal-deadbeef')
        os.makedirs(jdir)
        shutil.copy(os.path.join(out_dir, 'a_edges.png'), os.path.join(jdir, 'backup-1'))
        open(os.path.join(jdir, 'new-1'), 'wb').write(b'half-written')
        open(os.path.join(jdir, 'done-1'), 'wb').close()
        json.dump({'journal_id': 'deadbeef', 'entries': [
            {'idx': 1, 'dest': 'a_edges.png', 'staged': 'new-1', 'backup': 'backup-1'}]},
            open(os.path.join(jdir, 'manifest.json'), 'w'))
        open(os.path.join(out_dir, 'a_edges.png'), 'wb').write(b'half-written')

        report = batch.run()
        assert report.status == STATUS_COMPLETE
        assert open(os.path.join(out_dir, 'a_edges.png'), 'rb').read() == good
        assert not os.path.exists(jdir)

    def test_report_write_is_atomic_and_keeps_last_receipt_on_rerun(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        batch = BatchExecutor(_build(_multi_config()), in_dir, out_dir)
        batch.write_report(batch.run())
        receipt = os.path.join(out_dir, 'batch_report.json')
        first = open(receipt, 'rb').read()
        assert json.loads(first)['status'] == 'complete'
        # rerunning reproduces a valid receipt (temp files never linger)
        batch.write_report(batch.run())
        assert json.load(open(receipt, encoding='utf-8'))['status'] == 'complete'
        assert not [f for f in os.listdir(out_dir) if f.startswith('.imgpipe-tmp-')]


class TestReportDistinguishability:

    def test_text_report_mentions_status_and_conflicts(self, tmpdir_path):
        from image_pipeline.batch.executor import print_text_report
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(8, 8), os.path.join(in_dir, 'a.png'), fmt='PNG')
        rejected = BatchExecutor(
            _build(_multi_config(suffix_a='_x', suffix_b='_x', fmt_a='PNG', fmt_b='PNG')),
            in_dir, out_dir).run()
        text = print_text_report(rejected, verbose=True)
        assert 'PREFLIGHT REJECTED' in text
        assert 'Preflight Conflicts' in text
        assert 'rerun' in text.lower()

        out_dir2 = os.path.join(tmpdir_path, 'out2')
        complete = BatchExecutor(_build(_multi_config()), in_dir, out_dir2).run()
        text_ok = print_text_report(complete, verbose=True)
        assert 'Batch status    : COMPLETE' in text_ok
        assert 'edges_out' in text_ok and 'bytes=' in text_ok
