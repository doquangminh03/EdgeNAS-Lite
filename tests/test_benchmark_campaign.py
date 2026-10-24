import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_campaign import (
    CampaignError,
    build_session_schedule,
    make_plan,
    summarize_sessions,
)


class TestBenchmarkCampaign(unittest.TestCase):
    def test_schedule_is_seeded_and_contains_each_resolution_once(self):
        first = build_session_schedule((416, 448, 640), 4, 19)
        repeated = build_session_schedule((416, 448, 640), 4, 19)

        self.assertEqual(first, repeated)
        for session in first:
            self.assertCountEqual(
                session["resolution_order"],
                (416, 448, 640),
            )
        self.assertGreater(len({
            tuple(session["resolution_order"])
            for session in first
        }), 1)

    def test_summary_excludes_stabilization_and_keeps_session_units(self):
        sessions = [
            {
                "session": 1,
                "raw_samples": [{"latency_ms": 100.0}],
                "median_latency_ms": 100.0,
                "p95_latency_ms": 100.0,
            },
            {
                "session": 2,
                "raw_samples": [{"latency_ms": 10.0}, {"latency_ms": 20.0}],
                "median_latency_ms": 15.0,
                "p95_latency_ms": 19.5,
            },
            {
                "session": 3,
                "raw_samples": [{"latency_ms": 30.0}, {"latency_ms": 40.0}],
                "median_latency_ms": 35.0,
                "p95_latency_ms": 39.5,
            },
        ]

        summary = summarize_sessions(sessions, [1])

        self.assertEqual(summary["pooled_sample_count"], 4)
        self.assertEqual(summary["included_sessions"], [2, 3])
        self.assertEqual(summary["pooled_median_latency_ms"], 25.0)
        self.assertEqual(summary["session_median_mean_ms"], 25.0)
        self.assertIn("not independent", summary["inference_note"])

    def test_preflight_hashes_assets_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            image_dir = root / "images"
            image_dir.mkdir()
            for image_number in range(3):
                (image_dir / f"image_{image_number}.jpg").write_bytes(
                    f"image fixture {image_number}".encode("ascii")
                )
            config = root / "benchmark.yaml"
            config.write_text(
                "schema_version: '1.0'\n"
                "benchmark_id: test\n"
                "candidate_id: test_candidate\n"
                f"model:\n  checkpoint: {checkpoint}\n"
                "dataset:\n  name: KITTI\n  split: val\n"
                f"  images_dir: {image_dir}\n"
                "selection:\n  sample_size: 3\n  seed: 42\n"
                "protocol:\n  image_size: 640\n  warmup_runs: 10\n"
                "  repetitions_per_image: 3\n"
                "  timing_scope: preprocess_inference_postprocess\n"
                "  disk_io_included: false\n",
                encoding="utf-8",
            )
            output_dir = root / "campaign"

            plan = make_plan(
                config,
                root=root,
                run_id="test_campaign",
                hardware_id="test_cpu",
                output_dir=output_dir,
                resolutions=(416, 640),
                session_count=3,
                stabilization_sessions=1,
            )

            self.assertEqual(plan["status"], "planned")
            self.assertEqual(len(plan["dataset"]["images"]), 3)
            self.assertEqual(len(plan["checkpoint"]["sha256"]), 64)
            self.assertEqual(len(plan["dataset"]["manifest_sha256"]), 64)
            self.assertFalse(output_dir.exists())

    def test_rejects_invalid_session_exclusion(self):
        with self.assertRaisesRegex(CampaignError, "session"):
            build_session_schedule((416,), 1, 1)
            summarize_sessions([], [1])


    def test_thread_evidence_rejects_missing_or_mismatched_observations(self):
        import copy
        from scripts.benchmark_campaign import validate_thread_evidence
        record = {'protocol': {'version': 'cpu_threads_v2', 'requested_cpu_threads': 1},
                  'environment': {'requested_cpu_threads': 1, 'torch_num_threads': 1,
                                  'thread_observations': dict.fromkeys(('before_warmup', 'after_warmup', 'before_timing', 'after_timing'), 1)},
                  'raw_samples': [{'torch_num_threads_before': 1, 'torch_num_threads_after': 1}]}
        validate_thread_evidence(record, 1)
        for path, value in ((('environment', 'torch_num_threads'), 7),
                            (('protocol', 'requested_cpu_threads'), None),
                            (('environment', 'thread_observations', 'after_warmup'), 7),
                            (('raw_samples', 0, 'torch_num_threads_after'), None),
                            (('raw_samples', 0, 'torch_num_threads_before'), True)):
            mutated = copy.deepcopy(record)
            target = mutated
            for key in path[:-1]: target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(CampaignError):
                validate_thread_evidence(mutated, 1)


if __name__ == "__main__":
    unittest.main()