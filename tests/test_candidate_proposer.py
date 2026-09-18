import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from src.search_space.parser import (
    load_search_space,
    expand_candidates,
)
from src.llm_agent.candidate_proposer import (
    parse_proposal,
    collect_context,
    request_proposal,
    build_single_search_space,
    write_new_pair,
)


class TestCandidateProposer(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]

        self.pool = deepcopy(
            load_search_space(
                root / "configs/search_space_llm_pool.yaml"
            )
        )

        self.valid = {
            "schema_version": "1.0",
            "image_size": 448,
            "rationale": "Measure the intermediate resolution.",
        }

        self.available = [448, 480, 576, 608]

        self.checker = Mock(
            return_value={"status": "compatible"}
        )

    def entry(self, size=416, checkpoint=None):
        model = deepcopy(
            self.pool["fixed_configuration"]["model"]
        )
        model["model_size_mb"] = 5.102

        if checkpoint is not None:
            model["checkpoint"] = checkpoint

        return {
            "record_path": "results/candidates/demo.json",
            "record": {
                "candidate_id": "demo_" + str(size),
                "model": model,
                "accuracy": {
                    "image_size": size,
                    "map50_95": 0.2,
                },
                "benchmark": {
                    "image_size": size,
                    "median_latency_ms": 9.0,
                },
            },
        }

    def test_accepts_available_integer(self):
        result = parse_proposal(
            json.dumps(self.valid),
            self.available,
        )

        self.assertEqual(result, self.valid)

    def test_rejects_measured_outside_and_wrong_types(self):
        for value in (
            416,
            640,
            999,
            True,
            448.0,
            "448",
            None,
        ):
            with self.subTest(value=value):
                data = dict(
                    self.valid,
                    image_size=value,
                )

                with self.assertRaises(ValueError):
                    parse_proposal(
                        json.dumps(data),
                        self.available,
                    )

    def test_rejects_metrics_extra_keys_and_bad_rationale(self):
        cases = (
            dict(self.valid, map50_95=0.3),
            dict(self.valid, rationale=" "),
            dict(self.valid, rationale=42),
            dict(self.valid, schema_version="2.0"),
        )

        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    parse_proposal(
                        json.dumps(data),
                        self.available,
                    )

    def test_rejects_duplicate_keys_nonfinite_and_markdown(self):
        cases = (
            '{"image_size":448,"image_size":480}',
            '{"image_size":NaN}',
            "```json\n{}\n```",
            "[]",
            "null",
        )

        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_proposal(
                        text,
                        self.available,
                    )

    def test_context_excludes_different_checkpoint_and_incompatible_evidence(self):
        with TemporaryDirectory() as directory:
            entries = [
                self.entry(416),
                self.entry(448, "other.pt"),
                self.entry(480),
            ]

            self.checker.side_effect = [
                {"status": "compatible"},
                {"status": "compatible"},
                {"status": "incompatible"},
            ]

            context = collect_context(
                self.pool,
                entries,
                "mac",
                directory,
                self.checker,
            )

            self.assertEqual(
                context["measured_image_sizes"],
                [416],
            )
            self.assertEqual(
                len(context["excluded_evidence"]),
                2,
            )
            self.assertIn(
                448,
                context["available_image_sizes"],
            )
            self.assertIn(
                480,
                context["available_image_sizes"],
            )

    def test_context_blocks_existing_candidate_path_without_index_entry(self):
        with TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "results"
                / "candidates"
                / "yolo26n_kitti_pilot_imgsz448_cpu.json"
            )

            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")

            context = collect_context(
                self.pool,
                [self.entry()],
                "mac",
                directory,
                self.checker,
            )

            self.assertNotIn(
                448,
                context["available_image_sizes"],
            )
            self.assertEqual(
                context["blocked_candidates"][0]["image_size"],
                448,
            )

    def test_context_detects_sizes_and_does_not_mutate_inputs(self):
        entries = [
            self.entry(size)
            for size in (640, 416, 512)
        ]
        original = deepcopy(entries)

        with TemporaryDirectory() as directory:
            context = collect_context(
                self.pool,
                entries,
                "mac",
                directory,
                self.checker,
            )

        self.assertEqual(
            context["measured_image_sizes"],
            [416, 512, 640],
        )
        self.assertEqual(
            context["available_image_sizes"],
            self.available,
        )
        self.assertEqual(entries, original)

    def test_provider_receives_context_once_and_output_is_validated(self):
        provider = Mock()
        provider.complete.return_value = json.dumps(
            self.valid
        )

        context = {
            "available_image_sizes": self.available,
            "measured_evidence": [
                {"image_size": 416}
            ],
        }

        result = request_proposal(
            provider,
            {"request_id": "demo"},
            self.pool,
            context,
        )

        self.assertEqual(result, self.valid)
        provider.complete.assert_called_once()

        messages = provider.complete.call_args.args[0]

        self.assertEqual(
            [item["role"] for item in messages],
            ["system", "user"],
        )

        payload = json.loads(
            messages[1]["content"]
        )

        self.assertEqual(
            payload["available_image_sizes"],
            self.available,
        )

    def test_empty_available_or_evidence_stops_before_provider(self):
        provider = Mock()

        contexts = (
            {
                "available_image_sizes": [],
                "measured_evidence": [{}],
            },
            {
                "available_image_sizes": [448],
                "measured_evidence": [],
            },
        )

        for context in contexts:
            with self.assertRaises(ValueError):
                request_proposal(
                    provider,
                    {},
                    self.pool,
                    context,
                )

        provider.complete.assert_not_called()

    def test_one_candidate_plan_preserves_fixed_settings_and_original_pool(self):
        original = deepcopy(self.pool)

        plan = build_single_search_space(
            self.pool,
            self.valid,
            self.available,
            "demo",
        )

        self.assertEqual(self.pool, original)
        self.assertEqual(
            plan["search"]["experiment_budget"],
            1,
        )
        self.assertEqual(
            plan["existing_candidates"],
            [],
        )
        self.assertEqual(
            plan["fixed_configuration"],
            original["fixed_configuration"],
        )
        self.assertEqual(
            plan["evaluation"],
            original["evaluation"],
        )

        candidates = expand_candidates(plan)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["candidate_id"],
            "yolo26n_kitti_pilot_imgsz448_cpu",
        )
        self.assertEqual(
            candidates[0]["status"],
            "pending_evaluation",
        )

    def test_rejects_unsafe_request_id(self):
        with self.assertRaises(ValueError):
            build_single_search_space(
                self.pool,
                self.valid,
                self.available,
                "../demo",
            )

    def test_saves_new_pair_and_preserves_existing_files(self):
        with TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            report = Path(directory) / "report.json"

            plan = build_single_search_space(
                self.pool,
                self.valid,
                self.available,
                "demo",
            )

            write_new_pair(
                config,
                plan,
                report,
                {"status": "proposed_unmeasured"},
            )

            loaded = load_search_space(config)

            self.assertEqual(
                loaded["search"]["experiment_budget"],
                1,
            )

            original = report.read_bytes()

            with self.assertRaises(FileExistsError):
                write_new_pair(
                    config,
                    plan,
                    report,
                    {},
                )

            self.assertEqual(
                report.read_bytes(),
                original,
            )

            other = Path(directory) / "other.yaml"

            with self.assertRaises(FileExistsError):
                write_new_pair(
                    other,
                    plan,
                    report,
                    {},
                )

            self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()