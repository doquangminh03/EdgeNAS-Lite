import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.benchmarking import cpu_benchmark as benchmark
from src.candidate_runner import runner


class TestHardwareMetadata(unittest.TestCase):
    def test_invalid_ids(self):
        for value in ('', ' ', ' mac', 'two machines', 42, False):
            with self.subTest(value=value), self.assertRaises(benchmark.BenchmarkConfigError):
                benchmark.validate_hardware_id(value)

    def test_legacy_missing_id_is_not_invented(self):
        self.assertIsNone(benchmark.resolve_hardware_id({}, None))

    def test_explicit_id(self):
        self.assertEqual(benchmark.resolve_hardware_id({}, 'mac01'), 'mac01')

    def test_config_id(self):
        self.assertEqual(benchmark.resolve_hardware_id({'hardware_id': 'mac01'}, None), 'mac01')

    def test_conflicting_config_rejected(self):
        with self.assertRaises(benchmark.BenchmarkConfigError):
            benchmark.resolve_hardware_id({'hardware_id': 'other'}, 'mac01')

    def test_session_config_preserves_base(self):
        base = {'model': {}, 'protocol': {}, 'output': {}}
        original = copy.deepcopy(base)
        result = runner.build_session_benchmark_config(
            base, {'fixed_configuration': {'model': {'checkpoint': 'best.pt'}}},
            {'candidate_id': 'candidate', 'image_size': 448}, 1, Path('run.json'), 'mac01')
        self.assertEqual(result['hardware_id'], 'mac01')
        self.assertEqual(base, original)

    def test_raw_result_records_operator_id(self):
        config = dict(schema_version='1.0', benchmark_id='b', candidate_id='c', hardware_id='mac01',
                      model={'checkpoint': 'best.pt'}, dataset={'name': 'KITTI', 'split': 'val', 'images_dir': 'images'},
                      selection={'sample_size': 1, 'seed': 42}, output={'result_file': 'out.json'},
                      protocol=dict(device='cpu', image_size=448, batch_size=1, warmup_runs=0,
                                    repetitions_per_image=1, confidence_threshold=.25, iou_threshold=.7,
                                    max_detections=300, timing_scope='preprocess_inference_postprocess', disk_io_included=False))
        values = benchmark.read_config_values(config)
        result = benchmark.build_result(values, 1, [Path('images/1.png')], [{'latency_ms': 10.0}])
        self.assertEqual(result['hardware_id'], 'mac01')
        del config['hardware_id']
        result = benchmark.build_result(benchmark.read_config_values(config), 1, [Path('images/1.png')], [{'latency_ms': 10.0}])
        self.assertNotIn('hardware_id', result)

    def test_conflict_fails_before_validation(self):
        search = {'evaluation': {'benchmark_config': 'benchmark.yaml'}}
        with patch.object(runner, 'load_search_space', return_value=search), patch.object(runner, 'expand_candidates', return_value=[{'candidate_id': 'c', 'status': 'pending_evaluation'}]), patch.object(runner, 'load_benchmark_config', return_value={'hardware_id': 'other'}), patch.object(runner, 'run_validation') as validation:
            with self.assertRaises(benchmark.BenchmarkConfigError):
                runner.run_candidates(Path('configs/pool.yaml'), None, False, False, hardware_id='mac01')
            validation.assert_not_called()

    def test_reused_candidate_wrong_machine_rejected(self):
        search = {'evaluation': {'benchmark_config': 'benchmark.yaml'}}
        with patch.object(runner, 'load_search_space', return_value=search), patch.object(runner, 'expand_candidates', return_value=[{'candidate_id': 'c', 'status': 'reuse_existing'}]), patch.object(runner, 'load_benchmark_config', return_value={}), patch.object(runner, 'load_json', return_value={'benchmark': {'hardware_id': 'other'}}):
            with self.assertRaises(runner.CandidateRunnerError):
                runner.run_candidates(Path('configs/pool.yaml'), None, True, False, hardware_id='mac01')

    def sessions(self):
        return [dict(candidate_id='candidate', hardware_id='mac01',
                     protocol_status='standardized',
                     protocol=dict(device='cpu', image_size=448, batch_size=1,
                                   warmup_runs=10, repetitions_per_image=3,
                                   timing_scope='preprocess_inference_postprocess', disk_io_included=False),
                     dataset=dict(name='KITTI', split='val', selected_images=1, selection_seed=42),
                     environment=dict(system='test', architecture='arm64'),
                     selected_image_files=['1.png'], raw_samples=[{'latency_ms': 10.0}],
                     summary={'sample_count': 1}) for _ in range(3)]

    def test_pool_retains_id(self):
        result = runner.pool_benchmark_results(self.sessions(), [], [], 'candidate', 448, Path('.'))
        self.assertEqual(result['hardware_id'], 'mac01')
        self.assertEqual(result['total_latency_samples'], 3)

    def test_mixed_ids_rejected(self):
        sessions = self.sessions()
        sessions[1]['hardware_id'] = 'other'
        with self.assertRaises(runner.CandidateRunnerError):
            runner.validate_benchmark_sessions(sessions, 'candidate', 448)

    def test_partially_missing_ids_rejected(self):
        sessions = self.sessions()
        del sessions[1]['hardware_id']
        with self.assertRaises(runner.CandidateRunnerError):
            runner.validate_benchmark_sessions(sessions, 'candidate', 448)

    def test_legacy_pool_does_not_invent_id(self):
        sessions = self.sessions()
        for session in sessions:
            del session['hardware_id']
        result = runner.pool_benchmark_results(sessions, [], [], 'candidate', 448, Path('.'))
        self.assertNotIn('hardware_id', result)

    def test_five_sessions_receive_and_return_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = {'candidate_id': 'candidate', 'image_size': 448}
            search = {'evaluation': {'benchmark_config': 'benchmark.yaml'},
                      'fixed_configuration': {'model': {'checkpoint': 'best.pt'}}}
            base = {'model': {}, 'protocol': {}, 'output': {}}
            configs = []
            def run(command, **kwargs):
                config = runner.yaml.safe_load(Path(command[-1]).read_text())
                configs.append(config)
                result = self.sessions()[0]
                result['hardware_id'] = config['hardware_id']
                runner.save_json(result, Path(config['output']['result_file']))
            with patch.object(runner, 'load_benchmark_config', return_value=base), patch.object(runner.subprocess, 'run', side_effect=run):
                result = runner.run_benchmark_sessions(search, candidate, root, False, 'mac01')
            self.assertEqual(len(configs), 5)
            self.assertTrue(all(config['hardware_id'] == 'mac01' for config in configs))
            self.assertEqual(result['hardware_id'], 'mac01')
            self.assertEqual(len(result['source_results']), 3)
            self.assertEqual(len(result['excluded_stabilization_runs']), 2)

    def test_session_wrong_id_stops_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            search = {'evaluation': {'benchmark_config': 'benchmark.yaml'},
                      'fixed_configuration': {'model': {'checkpoint': 'best.pt'}}}
            with patch.object(runner, 'load_benchmark_config', return_value={'model': {}, 'protocol': {}, 'output': {}}), patch.object(runner.subprocess, 'run') as process, patch.object(runner, 'load_json', return_value={'hardware_id': 'other'}):
                with self.assertRaises(runner.CandidateRunnerError):
                    runner.run_benchmark_sessions(search, {'candidate_id': 'c', 'image_size': 448}, Path(directory), False, 'mac01')
                self.assertEqual(process.call_count, 1)


if __name__ == '__main__':
    unittest.main()
