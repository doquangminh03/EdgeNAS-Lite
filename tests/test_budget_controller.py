import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.llm_agent import budget_controller as controller


class TestBudgetController(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = dict(
            root=self.root,
            batch_id='batch',
            budget=2,
            requirement_file='request.yaml',
            hardware_id='mac01',
            template_candidate_id='template',
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_budget_limits_execution(self):
        with patch.object(
            controller, 'run_experiment',
            return_value={'status': 'completed'},
        ) as run:
            result = controller.run_budget(**self.args)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(result['status'], 'budget_finished')
            self.assertEqual(
                len({c.kwargs['run_id'] for c in run.call_args_list}),
                2,
            )

    def test_resume_does_not_repeat_completed(self):
        with patch.object(
            controller, 'run_experiment',
            return_value={'status': 'completed'},
        ) as run:
            controller.run_budget(**self.args)
            controller.run_budget(**self.args, resume=True)
            self.assertEqual(run.call_count, 2)

    def test_failure_stops_and_is_not_retried(self):
        with patch.object(
            controller, 'run_experiment',
            side_effect=RuntimeError('failure'),
        ) as run:
            with self.assertRaises(RuntimeError):
                controller.run_budget(**self.args)
            with self.assertRaises(ValueError):
                controller.run_budget(**self.args, resume=True)
            self.assertEqual(run.call_count, 1)

    def test_pool_exhaustion_stops(self):
        with patch.object(
            controller, 'run_experiment',
            return_value={'status': 'no_available_candidate'},
        ) as run:
            result = controller.run_budget(**self.args)
            self.assertEqual(result['status'], 'exhausted')
            self.assertEqual(run.call_count, 1)

    def test_dry_run_writes_nothing(self):
        with patch.object(
            controller, 'run_experiment',
            return_value={'status': 'planned'},
        ) as run:
            controller.run_budget(**self.args, dry_run=True)
            self.assertTrue(run.call_args.kwargs['dry_run'])
            self.assertEqual(list(self.root.iterdir()), [])

    def test_reconcile_finished_child_then_continue(self):
        folder = self.root / 'results/llm_batches'
        folder.mkdir(parents=True)
        settings = {
            key: self.args[key]
            for key in (
                'budget', 'requirement_file',
                'hardware_id', 'template_candidate_id',
            )
        }
        settings['model'] = None
        state = dict(
            settings=settings,
            status='running',
            slots=[{'run_id': 'batch_001', 'status': 'running'}],
        )
        (folder / 'batch.json').write_text(json.dumps(state))
        child = self.root / 'results/llm_experiments'
        child.mkdir()
        (child / 'batch_001.json').write_text(
            json.dumps({'status': 'completed'})
        )

        with patch.object(
            controller, 'run_experiment',
            return_value={'status': 'completed'},
        ) as run:
            controller.run_budget(**self.args, resume=True)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(
                run.call_args.kwargs['run_id'], 'batch_002'
            )


if __name__ == '__main__':
    unittest.main()
