import json
import os
import pytest
from image_pipeline.pipeline.engine import PipelineGraph, PipelineExecutor
from image_pipeline.nodes.definitions import (InputNode, OutputNode, GrayscaleNode,
                                              BrightnessNode, SobelNode, resolve_output_path,
                                              resolve_output_filename)
from image_pipeline.config.loader import PipelineConfig
from image_pipeline.batch.executor import BatchExecutor, print_text_report
from image_pipeline.utils.image_io import write_image, write_bytes_atomic
from image_pipeline.utils.transaction import (OutputTransaction, recover_pending_transactions,
                                              JOURNAL_DIRNAME, STAGING_DIRNAME)
from image_pipeline.algorithms import core as alg


def _two_output_graph(suffix_a='_a', suffix_b='_b', fmt_a='PNG', fmt_b='PNG'):
    g = PipelineGraph()
    g.add_node(InputNode('in'))
    g.add_node(GrayscaleNode('gray'))
    g.add_node(SobelNode('sobel', {'direction': 'x'}))
    g.add_node(BrightnessNode('bright', {'value': 20}))
    g.add_node(OutputNode('edges_out', {'suffix': suffix_a, 'format': fmt_a}))
    g.add_node(OutputNode('bright_out', {'suffix': suffix_b, 'format': fmt_b}))
    g.add_edge('in', 'gray')
    g.add_edge('gray', 'sobel')
    g.add_edge('gray', 'bright')
    g.add_edge('sobel', 'edges_out')
    g.add_edge('bright', 'bright_out')
    return g


def _conflict_graph():
    # Both output nodes resolve to the same file (same suffix + same format).
    g = PipelineGraph()
    g.add_node(InputNode('in'))
    g.add_node(GrayscaleNode('gray'))
    g.add_node(OutputNode('edge_out', {'suffix': '_v', 'format': 'PNG'}))
    g.add_node(OutputNode('thumb_out', {'suffix': '_v', 'format': 'PNG'}))
    g.add_edge('in', 'gray')
    g.add_edge('gray', 'edge_out')
    g.add_edge('gray', 'thumb_out')
    return g


def _write_input(in_dir, name='test.png', size=8):
    os.makedirs(in_dir, exist_ok=True)
    p = os.path.join(in_dir, name)
    write_image(alg.generate_gradient_image(size, size), p, fmt='PNG')
    return p


class TestConflictDetection:

    def test_within_image_conflict_detected(self, tmpdir_path):
        executor = PipelineExecutor(_conflict_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _write_input(in_dir)
        batch = BatchExecutor(executor, in_dir, out_dir)
        images = [os.path.join(in_dir, 'test.png')]
        conflicts = batch.detect_conflicts(images)
        assert len(conflicts) == 1
        assert conflicts[0]['type'] == 'within_image'
        assert set(conflicts[0]['node_ids']) == {'edge_out', 'thumb_out'}
        assert conflicts[0]['target'].endswith('test_v.png')

    def test_cross_image_conflict_uses_actual_format(self, tmpdir_path):
        # a.png and a.jpg both exported as PNG with an empty suffix -> a.png.
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out', {'suffix': '', 'format': 'PNG'}))
        g.add_edge('in', 'gray')
        g.add_edge('in', 'out')
        executor = PipelineExecutor(g)
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir, exist_ok=True)
        write_image(alg.generate_gradient_image(6, 6), os.path.join(in_dir, 'a.png'), fmt='PNG')
        write_image(alg.generate_gradient_image(6, 6), os.path.join(in_dir, 'a.jpg'), fmt='JPEG')
        batch = BatchExecutor(executor, in_dir, os.path.join(tmpdir_path, 'out'))
        conflicts = batch.detect_conflicts([os.path.join(in_dir, 'a.png'), os.path.join(in_dir, 'a.jpg')])
        assert any(c['type'] == 'cross_image' for c in conflicts)

    def test_no_conflict_for_distinct_suffixes(self, tmpdir_path):
        executor = PipelineExecutor(_two_output_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        _write_input(in_dir)
        batch = BatchExecutor(executor, in_dir, os.path.join(tmpdir_path, 'out'))
        assert batch.detect_conflicts([os.path.join(in_dir, 'test.png')]) == []


class TestPreflightRejection:

    def test_batch_rejected_writes_nothing(self, tmpdir_path):
        executor = PipelineExecutor(_conflict_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        _write_input(in_dir)
        batch = BatchExecutor(executor, in_dir, out_dir)
        report = batch.run()
        assert report.status == 'preflight_rejected'
        assert report.skipped == 1
        assert report.succeeded == 0 and report.failed == 0
        assert len(report.preflight_errors) == 1
        # No image products and no journal/staging leftovers.
        leftovers = [n for n in os.listdir(out_dir) if n.endswith('.png')]
        assert leftovers == []
        assert not os.path.isdir(os.path.join(out_dir, JOURNAL_DIRNAME))
        assert not os.path.isdir(os.path.join(out_dir, STAGING_DIRNAME))

    def test_preflight_json_and_text_distinguishable(self, tmpdir_path):
        executor = PipelineExecutor(_conflict_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _write_input(in_dir)
        batch = BatchExecutor(executor, in_dir, out_dir)
        report = batch.run()
        data = report.to_dict()
        assert data['status'] == 'preflight_rejected'
        assert data['preflight_errors'][0]['type'] == 'within_image'
        text = print_text_report(report, verbose=True)
        assert 'PREFLIGHT REJECTED' in text
        assert 'NOTHING WAS WRITTEN' in text

    def test_cli_exit_code_4_on_conflict(self, tmpdir_path):
        from image_pipeline.cli.main import build_parser
        import io
        import sys

        # Build a conflict pipeline config on disk.
        cfg = {'version': '1.0', 'name': 'conflict',
               'nodes': [{'id': 'in', 'type': 'input'}, {'id': 'gray', 'type': 'grayscale'},
                         {'id': 'o1', 'type': 'output', 'params': {'suffix': '_v', 'format': 'PNG'}},
                         {'id': 'o2', 'type': 'output', 'params': {'suffix': '_v', 'format': 'PNG'}}],
               'edges': [{'from': 'in', 'to': 'gray'}, {'from': 'gray', 'to': 'o1'}, {'from': 'gray', 'to': 'o2'}]}
        cfg_path = os.path.join(tmpdir_path, 'p.json')
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f)
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _write_input(in_dir)
        parser = build_parser()
        parsed = parser.parse_args(['run', '-c', cfg_path, '-i', in_dir, '-o', out_dir, '-q'])
        old_err, old_out = sys.stderr, sys.stdout
        sys.stderr, sys.stdout = io.StringIO(), io.StringIO()
        try:
            rc = parsed.func(parsed)
        finally:
            sys.stderr, sys.stdout = old_err, old_out
        assert rc == 4
        # Receipt still written and marked correctly.
        with open(os.path.join(out_dir, 'batch_report.json')) as f:
            data = json.load(f)
        assert data['status'] == 'preflight_rejected'
        # No image files produced.
        assert not any(n.endswith('.png') for n in os.listdir(out_dir))


class TestMultiOutputSuccess:

    def test_each_output_node_has_independent_artifact(self, tmpdir_path):
        executor = PipelineExecutor(_two_output_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _write_input(in_dir)
        report = BatchExecutor(executor, in_dir, out_dir).run()
        assert report.status == 'complete_success'
        r = report.results[0]
        assert r.success and r.status == 'success'
        assert len(r.outputs) == 2
        by_node = {a.node_id: a for a in r.outputs}
        for nid, fname in (('edges_out', 'test_a.png'), ('bright_out', 'test_b.png')):
            art = by_node[nid]
            assert art.success
            assert art.path.endswith(fname)
            assert art.bytes_size and art.bytes_size > 0
            assert art.width and art.height
            assert os.path.isfile(art.path)
            assert os.path.getsize(art.path) == art.bytes_size
        # Node-level results carry identity and committed state.
        node_ids = {nr.node_id: nr for nr in r.node_results}
        for nid in ('edges_out', 'bright_out'):
            assert node_ids[nid].output_path is not None
            assert node_ids[nid].output_bytes > 0
            assert node_ids[nid].committed is True
        data = report.to_dict()
        assert data['status'] == 'complete_success'
        assert len(data['results'][0]['outputs']) == 2

    def test_text_reports_complete_success(self, tmpdir_path):
        executor = PipelineExecutor(_two_output_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _write_input(in_dir)
        report = BatchExecutor(executor, in_dir, out_dir).run()
        text = print_text_report(report, verbose=True)
        assert 'COMPLETE SUCCESS' in text
        assert 'edges_out' in text and 'bright_out' in text
        assert 'bytes' in text


class TestRollback:

    def test_failed_second_output_rolls_back_first(self, tmpdir_path, monkeypatch):
        executor = PipelineExecutor(_two_output_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(out_dir, exist_ok=True)
        img_path = _write_input(in_dir)
        # A previous valid run's file for the first output must survive.
        prev_path = os.path.join(out_dir, 'test_a.png')
        write_bytes_atomic(b'PREVIOUS VALID CONTENT', prev_path)

        calls = {'n': 0}
        orig_stage = OutputTransaction.stage

        def flaky_stage(self, final_path, data):
            calls['n'] += 1
            if calls['n'] == 2:
                raise OSError('simulated disk full while staging second output')
            return orig_stage(self, final_path, data)

        monkeypatch.setattr(OutputTransaction, 'stage', flaky_stage)
        result = executor.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out_dir})

        assert not result.success
        assert result.status == 'rolled_back'
        # No half set: second target never created; first target untouched.
        assert not os.path.exists(os.path.join(out_dir, 'test_b.png'))
        with open(prev_path, 'rb') as f:
            assert f.read() == b'PREVIOUS VALID CONTENT'
        # Artifacts identify what was rolled back.
        assert any(a.rolled_back and not a.success for a in result.outputs)
        # No staging/journal leftovers.
        assert not os.path.isdir(os.path.join(out_dir, STAGING_DIRNAME))

    def test_batch_text_report_marks_rollback(self, tmpdir_path, monkeypatch):
        from image_pipeline.utils.types import BatchReport, BATCH_ROLLED_BACK
        executor = PipelineExecutor(_two_output_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        img_path = _write_input(in_dir)
        monkeypatch.setattr(OutputTransaction, 'commit', lambda self: (_ for _ in ()).throw(OSError('disk full')))
        result = executor.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out_dir})
        assert result.status == 'rolled_back'
        report = BatchReport(total=1, failed=1, succeeded=0, status=BATCH_ROLLED_BACK,
                             input_dir=in_dir, output_dir=out_dir, results=[result])
        text = print_text_report(report, verbose=True)
        assert 'ROLLED BACK' in text
        assert 'Re-run is safe' in text


class TestTransactionPrimitive:

    def test_commit_installs_group(self, tmpdir_path):
        out = os.path.join(tmpdir_path, 'out')
        os.makedirs(out)
        txn = OutputTransaction(out)
        p1 = os.path.join(out, 'a.bin')
        p2 = os.path.join(out, 'b.bin')
        txn.stage(p1, b'one')
        txn.stage(p2, b'two')
        txn.commit()
        assert open(p1, 'rb').read() == b'one'
        assert open(p2, 'rb').read() == b'two'
        assert not os.path.isdir(os.path.join(out, JOURNAL_DIRNAME))
        assert not os.path.isdir(os.path.join(out, STAGING_DIRNAME))

    def test_stage_twice_same_path_rejected(self, tmpdir_path):
        out = os.path.join(tmpdir_path, 'out')
        os.makedirs(out)
        txn = OutputTransaction(out)
        p = os.path.join(out, 'a.bin')
        txn.stage(p, b'one')
        with pytest.raises(RuntimeError):
            txn.stage(p, b'two')
        txn.rollback()
        assert not os.path.exists(p)

    def test_install_failure_restores_previous_files(self, tmpdir_path, monkeypatch):
        out = os.path.join(tmpdir_path, 'out')
        os.makedirs(out)
        p1 = os.path.join(out, 'a.bin')
        p2 = os.path.join(out, 'b.bin')
        write_bytes_atomic(b'OLD-A', p1)
        txn = OutputTransaction(out)
        txn.stage(p1, b'NEW-A')
        txn.stage(p2, b'NEW-B')
        real_replace = os.replace

        def fail_second_install(src, dst):
            if os.path.basename(dst) == 'b.bin':
                raise OSError('simulated replace failure')
            return real_replace(src, dst)

        monkeypatch.setattr('image_pipeline.utils.transaction.os.replace', fail_second_install)
        with pytest.raises(OSError):
            txn.commit()
        assert open(p1, 'rb').read() == b'OLD-A'
        assert not os.path.exists(p2)
        assert not os.path.isdir(os.path.join(out, JOURNAL_DIRNAME))


class TestCrashRecovery:

    def test_recovers_backed_up_journal(self, tmpdir_path):
        out = os.path.join(tmpdir_path, 'out')
        os.makedirs(out)
        target = os.path.join(out, 'a.png')
        backup = target + '.imgpipe-bak.abc123'
        write_bytes_atomic(b'OLD', target)
        os.replace(target, backup)                     # crash window: old moved away
        write_bytes_atomic(b'PARTIAL-NEW', target)     # ... partial install present
        jdir = os.path.join(out, JOURNAL_DIRNAME)
        os.makedirs(jdir)
        with open(os.path.join(jdir, 'txn-abc123.json'), 'w') as f:
            json.dump({'txn_id': 'abc123', 'state': 'backed_up',
                       'operations': [{'target': target, 'backup': backup}]}, f)
        recovered = recover_pending_transactions(out)
        assert len(recovered) == 1
        assert open(target, 'rb').read() == b'OLD'
        assert not os.path.exists(backup)

    def test_recovers_committed_journal_keeps_new_files(self, tmpdir_path):
        out = os.path.join(tmpdir_path, 'out')
        os.makedirs(out)
        target = os.path.join(out, 'a.png')
        backup = target + '.imgpipe-bak.def456'
        write_bytes_atomic(b'NEW', target)
        write_bytes_atomic(b'OLD', backup)
        jdir = os.path.join(out, JOURNAL_DIRNAME)
        os.makedirs(jdir)
        with open(os.path.join(jdir, 'txn-def456.json'), 'w') as f:
            json.dump({'txn_id': 'def456', 'state': 'committed',
                       'operations': [{'target': target, 'backup': backup}]}, f)
        recover_pending_transactions(out)
        assert open(target, 'rb').read() == b'NEW'
        assert not os.path.exists(backup)

    def test_stale_staging_cleaned(self, tmpdir_path):
        out = os.path.join(tmpdir_path, 'out')
        stale = os.path.join(out, STAGING_DIRNAME, 'deadbeef')
        os.makedirs(stale)
        write_bytes_atomic(b'x', os.path.join(stale, 'stage-000-a.bin'))
        recover_pending_transactions(out)
        assert not os.path.exists(os.path.join(out, STAGING_DIRNAME))


class TestAtomicWrites:

    def test_failed_write_preserves_existing_file(self, tmpdir_path, monkeypatch):
        import image_pipeline.utils.image_io as image_io
        path = os.path.join(tmpdir_path, 'keep.bin')
        write_bytes_atomic(b'GOOD RECEIPT', path)

        def boom(src, dst):
            raise OSError('simulated replace failure')

        monkeypatch.setattr(image_io.os, 'replace', boom)
        with pytest.raises(OSError):
            write_bytes_atomic(b'TRUNCATED', path)
        assert open(path, 'rb').read() == b'GOOD RECEIPT'

    def test_report_write_failure_keeps_last_receipt(self, tmpdir_path, monkeypatch):
        executor = PipelineExecutor(_two_output_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _write_input(in_dir)
        batch = BatchExecutor(executor, in_dir, out_dir)
        report = batch.run()
        path = batch.write_report(report)
        assert os.path.isfile(path)
        first = open(path, 'rb').read()

        def boom(data, p):
            raise OSError('disk full while writing receipt')

        monkeypatch.setattr('image_pipeline.batch.executor.write_bytes_atomic', boom)
        with pytest.raises(OSError):
            batch.write_report(report)
        assert open(path, 'rb').read() == first


class TestRerunSafety:

    def test_rerun_is_idempotent(self, tmpdir_path):
        executor = PipelineExecutor(_two_output_graph())
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        _write_input(in_dir)
        r1 = BatchExecutor(executor, in_dir, out_dir).run()
        first = {a.node_id: open(a.path, 'rb').read() for a in r1.results[0].outputs}
        r2 = BatchExecutor(executor, in_dir, out_dir).run()
        second = {a.node_id: open(a.path, 'rb').read() for a in r2.results[0].outputs}
        assert r1.status == r2.status == 'complete_success'
        assert first == second


class TestSingleOutputCompatibility:

    def test_single_output_output_path_still_set(self, tmpdir_path):
        g = PipelineGraph()
        g.add_node(InputNode('in'))
        g.add_node(GrayscaleNode('gray'))
        g.add_node(OutputNode('out', {'suffix': '_result'}))
        g.add_edge('in', 'gray')
        g.add_edge('gray', 'out')
        executor = PipelineExecutor(g)
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        img_path = _write_input(in_dir, 'test.png')
        result = executor.run({'input_path': img_path, 'input_filename': 'test.png', 'output_dir': out_dir})
        assert result.success
        assert result.output_path.endswith('test_result.png')
        assert os.path.isfile(result.output_path)
        assert len(result.outputs) == 1
        assert result.outputs[0].node_id == 'out'

    def test_resolve_output_filename_matches_dry_run_logic(self):
        assert resolve_output_filename('photo.JPG', {'suffix': '_thumb', 'format': 'PNG'}) == 'photo_thumb.png'
        assert resolve_output_filename('photo.png', {'suffix': '', 'format': None}) == 'photo.png'
        assert resolve_output_filename('photo', {'suffix': '_x', 'format': 'JPEG'}) == 'photo_x.jpg'
