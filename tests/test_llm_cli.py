import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.llm_agent.cli import main
from src.llm_agent.gemini_provider import GeminiProviderError


class TestLLMCLI(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

        index = self.root / "knowledge" / "index.yaml"
        index.parent.mkdir(parents=True)
        index.write_text("{}\n", encoding="utf-8")
        self.index = index

        self.output = self.root / "reports" / "demo.json"
        self.user_text = "Yêu cầu thử nghiệm CLI."

        self.requirement = {
            "schema_version": "1.0",
            "request_id": "cli_test",
            "target": {
                "task": "object_detection",
                "device": "cpu",
                "dataset": "KITTI",
            },
            "constraints": {
                "minimum_map50_95": 0.20,
                "maximum_median_latency_ms": 12.0,
                "maximum_model_size_mb": 6.0,
            },
            "preferences": {
                "optimization_goal": "balanced",
            },
        }

        # Replace the provider; keep the real interpreter and validation.
        provider_patch = patch("src.llm_agent.cli.GeminiProvider")
        self.provider_factory = provider_patch.start()
        self.addCleanup(provider_patch.stop)
        self.provider = self.provider_factory.return_value

        self.set_interpretation({
            "status": "ready",
            "requirement": self.requirement,
            "questions": [],
            "reasons": [],
        })

        # Existing proposal tests cover candidate evaluation and ranking.
        # These tests check how the CLI calls and handles that pipeline.
        proposal_patch = patch(
            "src.llm_agent.cli.propose_from_knowledge"
        )
        self.propose = proposal_patch.start()
        self.addCleanup(proposal_patch.stop)

        self.propose.return_value = {
            "proposal_status": "selected",
            "selected_source": {
                "candidate_id": "candidate_416",
                "record_path": "results/candidates/candidate_416.json",
            },
        }

        # Extra guard against accidental real network calls.
        network_patch = patch(
            "src.llm_agent.gemini_provider.urlopen",
            side_effect=AssertionError("Real HTTP is forbidden in tests."),
        )
        network_patch.start()
        self.addCleanup(network_patch.stop)

    def set_interpretation(self, interpretation):
        self.provider.complete.return_value = json.dumps(
            interpretation,
            ensure_ascii=False,
        )

    def arguments(self):
        return [
            "--text", self.user_text,
            "--request-id", "cli_test",
            "--hardware-id", "local_mac_cpu_01",
            "--project-root", str(self.root),
            "--output", "reports/demo.json",
        ]

    def run_cli(self, arguments=None):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                self.arguments() if arguments is None else arguments
            )

        return code, stdout.getvalue(), stderr.getvalue()

    def read_report(self):
        return json.loads(self.output.read_text(encoding="utf-8"))

    def test_ready_passes_requirement_to_proposal_and_saves_report(self):
        observed_paths = []
        expected_proposal = self.propose.return_value

        def inspect_requirement(requirement_path, **kwargs):
            path = Path(requirement_path)
            observed_paths.append(path)

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                self.requirement,
            )
            self.assertEqual(kwargs["project_root"], self.root)
            self.assertEqual(kwargs["index_file"], self.index)
            self.assertEqual(kwargs["model_family"], "YOLO26")
            self.assertEqual(
                kwargs["expected_hardware_id"],
                "local_mac_cpu_01",
            )
            return expected_proposal

        self.propose.side_effect = inspect_requirement
        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("candidate_416", stdout)
        self.provider.complete.assert_called_once()
        self.propose.assert_called_once()

        report = self.read_report()
        self.assertEqual(report["user_text"], self.user_text)
        self.assertEqual(report["request_id"], "cli_test")
        self.assertEqual(
            report["interpretation"]["requirement"],
            self.requirement,
        )
        self.assertEqual(report["proposal"], expected_proposal)
        self.assertEqual(
            report["expected_hardware_id"], "local_mac_cpu_01"
        )

        # Temporary requirement must be removed after the call.
        self.assertFalse(observed_paths[0].exists())

    def test_clarification_saves_questions_without_proposal(self):
        question = "Median latency tối đa là bao nhiêu ms?"
        self.set_interpretation({
            "status": "needs_clarification",
            "requirement": None,
            "questions": [question],
            "reasons": [],
        })

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 1)
        self.assertEqual(stderr, "")
        self.assertIn(question, stdout)
        self.propose.assert_not_called()

        report = self.read_report()
        self.assertIsNone(report["proposal"])
        self.assertIsNone(report["interpretation"]["requirement"])
        self.assertEqual(
            report["interpretation"]["questions"], [question]
        )

    def test_unsupported_saves_reasons_without_proposal(self):
        reason = "Instance segmentation chưa được hỗ trợ."
        self.set_interpretation({
            "status": "unsupported",
            "requirement": None,
            "questions": [],
            "reasons": [reason],
        })

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 1)
        self.assertEqual(stderr, "")
        self.assertIn(reason, stdout)
        self.propose.assert_not_called()

        report = self.read_report()
        self.assertIsNone(report["proposal"])
        self.assertEqual(
            report["interpretation"]["reasons"], [reason]
        )

    def test_no_feasible_candidate_saves_report_and_returns_one(self):
        self.propose.return_value = {
            "proposal_status": "no_feasible_candidate",
            "selected_source": None,
        }

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 1)
        self.assertEqual(stderr, "")
        self.assertIn("Selected: None", stdout)
        self.assertEqual(
            self.read_report()["proposal"],
            self.propose.return_value,
        )

    def test_provider_error_does_not_create_report_or_proposal(self):
        self.provider.complete.side_effect = GeminiProviderError(
            "Gemini request timed out."
        )

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 2)
        self.assertIn("timed out", stderr)
        self.propose.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_response_does_not_create_report_or_proposal(self):
        self.provider.complete.return_value = "not valid JSON"

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 2)
        self.assertIn("Error:", stderr)
        self.propose.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_preserved_without_calling_provider(self):
        self.output.parent.mkdir(parents=True)
        original = '{"keep": "existing report"}\n'
        self.output.write_text(original, encoding="utf-8")

        code, stdout, stderr = self.run_cli()

        self.assertEqual(code, 2)
        self.assertIn("already exists", stderr)
        self.assertEqual(
            self.output.read_text(encoding="utf-8"), original
        )
        self.provider_factory.assert_not_called()
        self.propose.assert_not_called()

    def test_missing_hardware_argument_stops_before_provider(self):
        arguments = self.arguments()
        position = arguments.index("--hardware-id")
        del arguments[position:position + 2]

        with self.assertRaises(SystemExit) as caught:
            self.run_cli(arguments)

        self.assertEqual(caught.exception.code, 2)
        self.provider_factory.assert_not_called()
        self.propose.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()