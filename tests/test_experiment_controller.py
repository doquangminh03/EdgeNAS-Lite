import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import yaml

from src.llm_agent.experiment_controller import run_experiment


class TestExperimentController(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.previous = Path.cwd()
        os.chdir(self.root)
        for folder in ('configs', 'knowledge', 'results/candidates'):
            (self.root / folder).mkdir(parents=True)
        self.index = self.root / 'knowledge/index.yaml'
        self.original = 'schema_version: "1.0"\ncandidate_records: []\n'
        self.index.write_text(self.original)
        self.pool = {'fixed_configuration': {'model': {'family': 'YOLO26', 'scale': 'n'},
                     'task': 'object_detection', 'deployment': {'device': 'cpu'}, 'dataset': {'name': 'KITTI'}},
                     'evaluation': {'benchmark_config': 'configs/benchmark.yaml'}, 'candidate_naming': {}}
        self.requirement = {'request_id': 'demo', 'target': {'task': 'object_detection', 'device': 'cpu', 'dataset': 'KITTI'}}
        self.context = {'measured_evidence': [{'image_size': 416}], 'available_image_sizes': [480]}
        self.record = {'candidate_id': 'new480', 'accuracy': {'image_size': 480}, 'benchmark': {'image_size': 480}}
        def write_pair(config_path, plan, report_path, report):
            config_path.write_text(yaml.safe_dump(plan))
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report))
        def execute(*args, **kwargs):
            if not args[2]:
                (self.root / 'results/candidates/new480.json').write_text(json.dumps(self.record))
            return {'completed': []}
        proposer = SimpleNamespace(load_search_space=Mock(return_value=self.pool),
            collect_context=Mock(return_value=self.context), request_proposal=Mock(return_value={'image_size': 480, 'rationale': 'test'}),
            build_single_search_space=Mock(return_value={'plan': 'singleton'}), candidate_id_for=Mock(return_value='new480'),
            write_new_pair=Mock(side_effect=write_pair), PROMPT_VERSION='test', SYSTEM_PROMPT='test')
        runner = SimpleNamespace(load_template_candidate=Mock(), load_benchmark_config=Mock(return_value={}),
            resolve_hardware_id=Mock(), benchmark_result_paths=Mock(return_value=[]),
            run_candidates=Mock(side_effect=execute), load_json=lambda path: json.loads(path.read_text()))
        self.api = SimpleNamespace(proposer=proposer, runner=runner, load=Mock(return_value={}), query=Mock(return_value=[]),
            check=Mock(return_value={'status': 'compatible'}), rank=Mock(return_value={'requirement': self.requirement}),
            provider=Mock(return_value=SimpleNamespace(model='fake')))
        self.args = dict(root=self.root, requirement_file='configs/request.yaml', hardware_id='mac01',
                         template_candidate_id='template', run_id='run1', backend=self.api)

    def tearDown(self):
        os.chdir(self.previous)
        self.temporary.cleanup()

    def report(self):
        return json.loads((self.root / 'results/llm_experiments/run1.json').read_text())

    def test_dry_run_no_calls_or_writes(self):
        before = sorted(str(p) for p in self.root.rglob('*'))
        result = run_experiment(**self.args, dry_run=True)
        self.assertEqual(result['status'], 'planned')
        self.api.provider.assert_not_called()
        self.api.runner.run_candidates.assert_not_called()
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob('*')))

    def test_success_one_measurement_and_registration(self):
        result = run_experiment(**self.args)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['experiments_attempted'], 1)
        self.assertTrue(result['index_updated'])
        calls = self.api.runner.run_candidates.call_args_list
        self.assertEqual([c.args[2] for c in calls], [True, False])
        self.assertTrue(all(c.args[3] is False for c in calls))
        self.assertTrue(all(c.kwargs['hardware_id'] == 'mac01' for c in calls))
        self.api.proposer.request_proposal.assert_called_once()
        self.assertEqual(len(yaml.safe_load(self.index.read_text())['candidate_records']), 1)
        self.assertEqual(self.api.rank.call_count, 2)
        self.assertFalse((self.root / 'results/.llm_experiment.lock').exists())

    def test_budget_rejected_before_provider(self):
        with self.assertRaises(ValueError):
            run_experiment(**self.args, budget=2)
        self.api.provider.assert_not_called()

    def test_incompatible_not_registered(self):
        self.api.check.return_value = {'status': 'insufficient_metadata'}
        result = run_experiment(**self.args)
        self.assertEqual(result['status'], 'rejected_incompatible')
        self.assertEqual(self.index.read_text(), self.original)
        self.assertEqual(self.api.rank.call_count, 1)

    def test_benchmark_failure_preserves_index_and_report(self):
        self.api.runner.run_candidates.side_effect = [{}, RuntimeError('benchmark failed')]
        with self.assertRaises(RuntimeError):
            run_experiment(**self.args)
        self.assertEqual(self.report()['status'], 'failed')
        self.assertEqual(self.report()['experiments_attempted'], 1)
        self.assertEqual(self.index.read_text(), self.original)
        self.assertFalse((self.root / 'results/.llm_experiment.lock').exists())

    def test_existing_run_blocks_api(self):
        (self.root / 'configs/llm_experiment_run1.yaml').write_text('existing')
        with self.assertRaises(FileExistsError):
            run_experiment(**self.args)
        self.api.provider.assert_not_called()

    def test_existing_measurement_not_overwritten(self):
        path = self.root / 'results/candidates/new480.json'
        path.write_text('existing')
        with self.assertRaises(FileExistsError):
            run_experiment(**self.args)
        self.api.runner.run_candidates.assert_not_called()
        self.assertEqual(path.read_text(), 'existing')

    def test_lock_blocks_execution(self):
        lock = self.root / 'results/.llm_experiment.lock'
        lock.write_text('other run')
        with self.assertRaises(FileExistsError):
            run_experiment(**self.args)
        self.assertEqual(lock.read_text(), 'other run')
        self.api.provider.assert_not_called()

    def test_index_changed_during_measurement_preserved(self):
        original_execute = self.api.runner.run_candidates.side_effect
        def execute(*args, **kwargs):
            result = original_execute(*args, **kwargs)
            if not args[2]:
                self.index.write_text(self.original + '# external edit\n')
            return result
        self.api.runner.run_candidates.side_effect = execute
        with self.assertRaises(ValueError):
            run_experiment(**self.args)
        self.assertIn('# external edit', self.index.read_text())
        self.assertEqual(self.report()['status'], 'failed')

    def test_no_available_candidate_no_api(self):
        self.context['available_image_sizes'] = []
        result = run_experiment(**self.args)
        self.assertEqual(result['status'], 'no_available_candidate')
        self.api.provider.assert_not_called()

    def test_wrong_measured_resolution_rejected(self):
        self.record['accuracy']['image_size'] = 512
        with self.assertRaises(ValueError):
            run_experiment(**self.args)
        self.assertEqual(self.index.read_text(), self.original)

    def test_invalid_proposal_no_measurement(self):
        self.api.proposer.request_proposal.side_effect = ValueError('invalid proposal')
        with self.assertRaises(ValueError):
            run_experiment(**self.args)
        self.api.runner.run_candidates.assert_not_called()
        self.assertEqual(self.report()['experiments_attempted'], 0)


if __name__ == '__main__':
    unittest.main()
