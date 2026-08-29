import copy
import tempfile
import unittest
from pathlib import Path

from src.benchmarking.cpu_benchmark import (
    BenchmarkConfigError,
    collect_images,
    read_config_values,
    select_images,
    summarize_latencies,
)


VALID_CONFIG = {
    "schema_version": "1.0",
    "benchmark_id": "test_cpu",
    "candidate_id": "test_candidate",
    "model": {
        "checkpoint": "model.pt",
    },
    "dataset": {
        "name": "KITTI",
        "split": "val",
        "images_dir": "datasets/kitti/images/val",
    },
    "selection": {
        "sample_size": 2,
        "seed": 42,
    },
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
    "output": {
        "result_file": "result.json",
    },
}


class TestCpuBenchmark(unittest.TestCase):
    def test_select_images_is_deterministic(self) -> None:
        images = [
            Path(f"image_{index}.png")
            for index in range(10)
        ]

        first_selection = select_images(
            images,
            sample_size=4,
            seed=42,
        )
        second_selection = select_images(
            images,
            sample_size=4,
            seed=42,
        )

        self.assertEqual(
            first_selection,
            second_selection,
        )
        self.assertEqual(
            len(first_selection),
            4,
        )

    def test_rejects_excessive_sample_size(self) -> None:
        images = [
            Path("one.png"),
            Path("two.png"),
        ]

        with self.assertRaisesRegex(
            BenchmarkConfigError,
            "only 2 are available",
        ):
            select_images(
                images,
                sample_size=3,
                seed=42,
            )

    def test_collect_images_filters_extensions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (root / "first.jpg").touch()
            (root / "second.png").touch()
            (root / "notes.txt").touch()

            images = collect_images(root)

            self.assertEqual(
                len(images),
                2,
            )
            self.assertEqual(
                {
                    path.suffix
                    for path in images
                },
                {".jpg", ".png"},
            )

    def test_summarizes_latency_samples(self) -> None:
        samples = [
            {"latency_ms": 10.0},
            {"latency_ms": 20.0},
            {"latency_ms": 30.0},
            {"latency_ms": 40.0},
        ]

        summary = summarize_latencies(samples)

        self.assertEqual(
            summary["sample_count"],
            4,
        )
        self.assertEqual(
            summary["average_latency_ms"],
            25.0,
        )
        self.assertEqual(
            summary["median_latency_ms"],
            25.0,
        )
        self.assertEqual(
            summary["p95_latency_ms"],
            38.5,
        )
        self.assertEqual(
            summary["fps_from_median"],
            40.0,
        )

    def test_rejects_non_cpu_device(self) -> None:
        config = copy.deepcopy(VALID_CONFIG)
        config["protocol"]["device"] = "mps"

        with self.assertRaisesRegex(
            BenchmarkConfigError,
            "only supports device: cpu",
        ):
            read_config_values(config)

    def test_rejects_batch_size_above_one(
        self,
    ) -> None:
        config = copy.deepcopy(VALID_CONFIG)
        config["protocol"]["batch_size"] = 4

        with self.assertRaisesRegex(
            BenchmarkConfigError,
            "requires batch_size: 1",
        ):
            read_config_values(config)


if __name__ == "__main__":
    unittest.main()