import copy
import itertools
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from unittest.mock import patch

import numpy as np
from src.benchmarking import cpu_benchmark as benchmark
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


class TestCpuThreadControl(unittest.TestCase):
    def run_cli(self, directory, *, reset_at=None, config_threads=None):
        root = Path(directory)
        checkpoint = root / "model.pt"
        checkpoint.touch()
        config = copy.deepcopy(VALID_CONFIG)
        config["model"]["checkpoint"] = str(checkpoint)
        config["selection"]["sample_size"] = 1
        config["protocol"].update(warmup_runs=1, repetitions_per_image=1)
        if config_threads is not None:
            config["protocol"]["cpu_threads"] = config_threads
        output = root / "result.json"
        config["output"]["result_file"] = str(output)
        image_path = Path("datasets/kitti/images/val/one.png")
        state = {"threads": 1, "calls": []}
        def predict(model, image, values):
            state["calls"].append(state["threads"])
            if len(state["calls"]) == 1 or len(state["calls"]) == reset_at:
                state["threads"] = 7
        with patch.object(benchmark, "load_config", return_value=config), \
             patch.object(benchmark, "YOLO"), \
             patch.object(benchmark, "collect_images", return_value=[image_path]), \
             patch.object(benchmark, "preload_images", return_value=[(image_path, np.zeros((8, 8, 3), dtype=np.uint8))]), \
             patch.object(benchmark, "run_prediction", side_effect=predict), \
             patch.object(benchmark.torch, "get_num_threads", side_effect=lambda: state["threads"]), \
             patch.object(benchmark.torch, "set_num_threads", side_effect=lambda count: state.update(threads=count)), \
             patch.object(benchmark.time, "perf_counter_ns", side_effect=itertools.count(step=1000000)), \
             patch("sys.argv", ["cpu_benchmark", "fixture.yaml", "--threads", "1"]), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()) as stderr:
            try:
                benchmark.main()
            except SystemExit:
                if reset_at is not None:
                    stage = "after_warmup_prediction" if reset_at == 2 else "after_timed_prediction"
                    self.assertIn("CPU thread mismatch at " + stage, stderr.getvalue())
                elif config_threads is not None:
                    self.assertIn("conflicts with --threads", stderr.getvalue())
                raise
        return json.loads(output.read_text()), state

    def test_backend_reset_is_overridden_before_warmup_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state = self.run_cli(directory)
        self.assertEqual(state["calls"], [1, 1, 1])
        self.assertEqual(result["environment"]["thread_observations"], {
            "after_backend_setup": 7, "before_warmup": 1,
            "after_warmup": 1, "before_timing": 1, "after_timing": 1,
        })
        self.assertEqual(result["protocol"]["requested_cpu_threads"], 1)
        self.assertEqual(result["environment"]["requested_cpu_threads"], 1)
        self.assertEqual(result["environment"]["torch_num_threads"], 1)
        self.assertEqual(len(result["raw_samples"]), 1)
        self.assertEqual(result["raw_samples"][0]["torch_num_threads_before"], 1)
        self.assertEqual(result["raw_samples"][0]["torch_num_threads_after"], 1)
        self.assertEqual(result["raw_samples"][0]["latency_ms"], 1)

    def test_later_warmup_or_timed_reset_exits_without_success_file(self):
        for call in (2, 3):
            with self.subTest(reset_at=call), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit) as error:
                    self.run_cli(directory, reset_at=call)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse((Path(directory) / "result.json").exists())

    def test_ineffective_thread_setter_fails_before_warmup(self):
        with patch.object(benchmark, "run_prediction"), \
             patch.object(benchmark.torch, "get_num_threads", return_value=7), \
             patch.object(benchmark.torch, "set_num_threads"):
            with self.assertRaisesRegex(BenchmarkConfigError, "before_warmup.*requested 1, observed 7"):
                benchmark.initialize_backend(object(), np.zeros((8, 8, 3)), {"requested_cpu_threads": 1})

    def test_timing_rejects_drift_before_first_prediction(self):
        with patch.object(benchmark.torch, "get_num_threads", return_value=7), \
             patch.object(benchmark, "run_prediction") as predict:
            with self.assertRaisesRegex(BenchmarkConfigError, "before_timing"):
                benchmark.measure_latency(object(), [], {"requested_cpu_threads": 1})
            predict.assert_not_called()

    def test_invalid_config_threads_and_cli_conflict_are_rejected(self):
        for value in (0, -1, True, 1.5, "1"):
            config = copy.deepcopy(VALID_CONFIG)
            config["protocol"]["cpu_threads"] = value
            with self.subTest(value=value), self.assertRaises(BenchmarkConfigError):
                read_config_values(config)
        config["protocol"]["cpu_threads"] = 2
        self.assertEqual(read_config_values(config)["requested_cpu_threads"], 2)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as error:
                self.run_cli(directory, config_threads=2)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((Path(directory) / "result.json").exists())


if __name__ == "__main__":
    unittest.main()