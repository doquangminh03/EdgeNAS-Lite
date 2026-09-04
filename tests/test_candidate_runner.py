import copy
import unittest
from pathlib import Path

from src.candidate_runner.runner import (
    CandidateRunnerError,
    build_candidate_record,
    build_session_benchmark_config,
    build_validation_kwargs,
    extract_accuracy_result,
    pool_benchmark_results,
    select_candidates,
)


SEARCH_SPACE = {
    "schema_version": "1.0",
    "search_space_id": "resolution_search_v1",
    "fixed_configuration": {
        "model": {
            "checkpoint": "runs/best.pt",
        },
        "dataset": {
            "name": "KITTI",
            "data_config": "kitti.yaml",
            "validation_split": "val",
            "validation_images": 1496,
            "classes": [
                "car",
                "pedestrian",
                "cyclist",
            ],
            "class_ids": [0, 3, 5],
        },
        "deployment": {
            "device": "cpu",
            "batch_size": 1,
        },
        "prediction": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "max_detections": 300,
        },
    },
    "evaluation": {
        "validation_output_dir": (
            "runs/val/search_candidates"
        ),
    },
}

CANDIDATE = {
    "candidate_id": "candidate_imgsz416_cpu",
    "image_size": 416,
    "status": "pending_evaluation",
}

BASE_BENCHMARK_CONFIG = {
    "schema_version": "1.0",
    "benchmark_id": "base",
    "candidate_id": "base_candidate",
    "model": {"checkpoint": "old.pt"},
    "dataset": {
        "name": "KITTI",
        "split": "val",
        "images_dir": "datasets/kitti/images/val",
    },
    "selection": {"sample_size": 100, "seed": 42},
    "protocol": {
        "device": "cpu",
        "image_size": 640,
        "batch_size": 1,
        "warmup_runs": 10,
        "repetitions_per_image": 3,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.7,
        "max_detections": 300,
        "timing_scope": (
            "preprocess_inference_postprocess"
        ),
        "disk_io_included": False,
    },
    "output": {"result_file": "old.json"},
}


class FakeBoxMetrics:
    mp = 0.5
    mr = 0.4
    map50 = 0.6
    map = 0.3
    ap_class_index = [0, 3, 5]

    def class_result(self, position):
        results = [
            (0.7, 0.8, 0.9, 0.5),
            (0.4, 0.3, 0.35, 0.2),
            (0.4, 0.1, 0.2, 0.1),
        ]
        return results[position]


class FakeMetrics:
    box = FakeBoxMetrics()
    nt_per_class = [100, 0, 0, 20, 0, 10]


def make_benchmark_result(
    candidate_id="candidate_imgsz416_cpu",
    image_size=416,
    latencies=None,
):
    if latencies is None:
        latencies = [10.0, 20.0]

    samples = [
        {
            "repetition": 1,
            "image": f"image_{index}.png",
            "latency_ms": latency,
        }
        for index, latency in enumerate(latencies)
    ]

    return {
        "candidate_id": candidate_id,
        "protocol_status": "standardized",
        "dataset": {
            "name": "KITTI",
            "split": "val",
            "selected_images": 100,
            "selection_seed": 42,
        },
        "protocol": {
            "device": "cpu",
            "image_size": image_size,
            "batch_size": 1,
            "warmup_runs": 10,
            "repetitions_per_image": 3,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.7,
            "max_detections": 300,
            "timing_scope": (
                "preprocess_inference_postprocess"
            ),
            "disk_io_included": False,
        },
        "environment": {
            "system": "test-system",
            "architecture": "arm64",
        },
        "summary": {
            "sample_count": len(samples),
        },
        "selected_image_files": ["same_image.png"],
        "raw_samples": samples,
    }


class TestCandidateRunner(unittest.TestCase):
    def test_validation_uses_low_default_confidence(self):
        arguments = build_validation_kwargs(
            SEARCH_SPACE,
            CANDIDATE,
            Path("/project"),
        )

        self.assertNotIn("conf", arguments)
        self.assertEqual(arguments["imgsz"], 416)
        self.assertEqual(arguments["classes"], [0, 3, 5])

    def test_extracts_overall_and_per_class_accuracy(self):
        accuracy = extract_accuracy_result(
            FakeMetrics(),
            SEARCH_SPACE,
            CANDIDATE,
        )

        self.assertEqual(accuracy["map50_95"], 0.3)
        self.assertEqual(accuracy["validation_instances"], 130)
        self.assertEqual(
            accuracy["per_class"]["pedestrian"]["class_id"],
            3,
        )

    def test_benchmark_config_is_candidate_specific(self):
        result_path = Path("result_run1.json")
        config = build_session_benchmark_config(
            BASE_BENCHMARK_CONFIG,
            SEARCH_SPACE,
            CANDIDATE,
            1,
            result_path,
        )

        self.assertEqual(
            config["candidate_id"],
            "candidate_imgsz416_cpu",
        )
        self.assertEqual(config["protocol"]["image_size"], 416)
        self.assertEqual(
            config["output"]["result_file"],
            str(result_path),
        )
        self.assertEqual(
            BASE_BENCHMARK_CONFIG["protocol"]["image_size"],
            640,
        )

    def test_pools_three_benchmark_sessions(self):
        results = [
            make_benchmark_result(latencies=[10.0, 20.0]),
            make_benchmark_result(latencies=[20.0, 30.0]),
            make_benchmark_result(latencies=[30.0, 40.0]),
        ]
        root = Path("/project")
        included_paths = [
            root / f"run{index}.json"
            for index in (3, 4, 5)
        ]
        excluded_paths = [
            root / f"run{index}.json"
            for index in (1, 2)
        ]

        benchmark = pool_benchmark_results(
            results,
            included_paths,
            excluded_paths,
            "candidate_imgsz416_cpu",
            416,
            root,
        )

        self.assertEqual(benchmark["session_count"], 3)
        self.assertEqual(benchmark["total_latency_samples"], 6)
        self.assertEqual(benchmark["median_latency_ms"], 25.0)
        self.assertEqual(
            benchmark["source_results"],
            ["run3.json", "run4.json", "run5.json"],
        )

    def test_rejects_wrong_benchmark_image_size(self):
        results = [
            make_benchmark_result(image_size=512)
            for _ in range(3)
        ]

        with self.assertRaisesRegex(
            CandidateRunnerError,
            "image size does not match",
        ):
            pool_benchmark_results(
                results,
                [Path("run3.json")] * 3,
                [Path("run1.json"), Path("run2.json")],
                "candidate_imgsz416_cpu",
                416,
                Path("."),
            )

    def test_candidate_record_preserves_training_metadata(self):
        source = {
            "model": {"model_size_mb": 5.102},
            "training": {"image_size": 640},
        }
        accuracy = {"map50_95": 0.25}
        benchmark = {"median_latency_ms": 12.0}

        record = build_candidate_record(
            source,
            CANDIDATE,
            accuracy,
            benchmark,
        )

        self.assertEqual(record["training"]["image_size"], 640)
        self.assertEqual(
            record["candidate_id"],
            "candidate_imgsz416_cpu",
        )

    def test_rejects_candidate_outside_search_space(self):
        candidates = [copy.deepcopy(CANDIDATE)]

        with self.assertRaisesRegex(
            CandidateRunnerError,
            "not in the search space",
        ):
            select_candidates(candidates, "unknown_candidate")


if __name__ == "__main__":
    unittest.main()
