import argparse
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import ultralytics
import yaml
from PIL import Image
from ultralytics import YOLO


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


class BenchmarkConfigError(ValueError):
    """Raised when the benchmark configuration is invalid."""


def load_config(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Benchmark config not found: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise BenchmarkConfigError(
            f"Invalid benchmark YAML: {error}"
        ) from error

    if not isinstance(config, dict):
        raise BenchmarkConfigError(
            "Benchmark config root must be a mapping/object."
        )

    return config


def read_config_values(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        values = {
            "schema_version": str(
                config["schema_version"]
            ),
            "benchmark_id": str(
                config["benchmark_id"]
            ),
            "candidate_id": str(
                config["candidate_id"]
            ),
            "checkpoint": Path(
                config["model"]["checkpoint"]
            ),
            "dataset_name": str(
                config["dataset"]["name"]
            ),
            "dataset_split": str(
                config["dataset"]["split"]
            ),
            "images_dir": Path(
                config["dataset"]["images_dir"]
            ),
            "sample_size": int(
                config["selection"]["sample_size"]
            ),
            "seed": int(
                config["selection"]["seed"]
            ),
            "device": str(
                config["protocol"]["device"]
            ),
            "image_size": int(
                config["protocol"]["image_size"]
            ),
            "batch_size": int(
                config["protocol"]["batch_size"]
            ),
            "warmup_runs": int(
                config["protocol"]["warmup_runs"]
            ),
            "repetitions_per_image": int(
                config["protocol"][
                    "repetitions_per_image"
                ]
            ),
            "confidence_threshold": float(
                config["protocol"][
                    "confidence_threshold"
                ]
            ),
            "iou_threshold": float(
                config["protocol"]["iou_threshold"]
            ),
            "max_detections": int(
                config["protocol"]["max_detections"]
            ),
            "timing_scope": str(
                config["protocol"]["timing_scope"]
            ),
            "disk_io_included": bool(
                config["protocol"]["disk_io_included"]
            ),
            "result_file": Path(
                config["output"]["result_file"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkConfigError(
            f"Missing or invalid benchmark config value: {error}"
        ) from error

    if values["device"].lower() != "cpu":
        raise BenchmarkConfigError(
            "This benchmark only supports device: cpu."
        )

    if values["batch_size"] != 1:
        raise BenchmarkConfigError(
            "Standard CPU protocol requires batch_size: 1."
        )

    if values["sample_size"] <= 0:
        raise BenchmarkConfigError(
            "sample_size must be greater than 0."
        )

    if values["warmup_runs"] < 0:
        raise BenchmarkConfigError(
            "warmup_runs cannot be negative."
        )

    if values["repetitions_per_image"] <= 0:
        raise BenchmarkConfigError(
            "repetitions_per_image must be greater than 0."
        )

    return values


def collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Validation image directory not found: {images_dir}"
        )

    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not images:
        raise BenchmarkConfigError(
            f"No supported images found in: {images_dir}"
        )

    return images


def select_images(
    images: List[Path],
    sample_size: int,
    seed: int,
) -> List[Path]:
    if sample_size > len(images):
        raise BenchmarkConfigError(
            f"Requested {sample_size} images, "
            f"but only {len(images)} are available."
        )

    random_generator = random.Random(seed)

    return sorted(
        random_generator.sample(
            images,
            sample_size,
        )
    )


def preload_images(
    image_paths: List[Path],
) -> List[Tuple[Path, np.ndarray]]:
    preloaded_images = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_array = np.asarray(
                image.convert("RGB")
            ).copy()

        preloaded_images.append(
            (image_path, rgb_array)
        )

    return preloaded_images


def run_prediction(
    model: YOLO,
    image: np.ndarray,
    values: Dict[str, Any],
) -> None:
    model.predict(
        source=image,
        device="cpu",
        imgsz=values["image_size"],
        conf=values["confidence_threshold"],
        iou=values["iou_threshold"],
        max_det=values["max_detections"],
        verbose=False,
        save=False,
    )


def run_warmup(
    model: YOLO,
    images: List[Tuple[Path, np.ndarray]],
    values: Dict[str, Any],
) -> None:
    for run_index in range(values["warmup_runs"]):
        _, image = images[
            run_index % len(images)
        ]

        run_prediction(
            model,
            image,
            values,
        )


def measure_latency(
    model: YOLO,
    images: List[Tuple[Path, np.ndarray]],
    values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    samples = []

    for repetition in range(
        values["repetitions_per_image"]
    ):
        ordered_images = list(images)

        random.Random(
            values["seed"] + repetition
        ).shuffle(ordered_images)

        for image_path, image in ordered_images:
            start_time = time.perf_counter_ns()

            run_prediction(
                model,
                image,
                values,
            )

            end_time = time.perf_counter_ns()

            latency_ms = (
                end_time - start_time
            ) / 1_000_000

            samples.append(
                {
                    "repetition": repetition + 1,
                    "image": str(
                        image_path.relative_to(
                            values["images_dir"]
                        )
                    ),
                    "latency_ms": round(
                        latency_ms,
                        3,
                    ),
                }
            )

    return samples


def summarize_latencies(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    latencies = [
        sample["latency_ms"]
        for sample in samples
    ]

    median_latency = statistics.median(
        latencies
    )

    return {
        "sample_count": len(latencies),
        "average_latency_ms": round(
            statistics.mean(latencies),
            3,
        ),
        "median_latency_ms": round(
            median_latency,
            3,
        ),
        "p95_latency_ms": round(
            float(np.percentile(latencies, 95)),
            3,
        ),
        "standard_deviation_ms": round(
            statistics.pstdev(latencies),
            3,
        ),
        "min_latency_ms": round(
            min(latencies),
            3,
        ),
        "max_latency_ms": round(
            max(latencies),
            3,
        ),
        "fps_from_median": round(
            1000.0 / median_latency,
            3,
        ),
    }


def build_result(
    values: Dict[str, Any],
    available_image_count: int,
    selected_images: List[Path],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "benchmark_id": values["benchmark_id"],
        "candidate_id": values["candidate_id"],
        "protocol_status": "standardized",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": {
            "checkpoint": str(
                values["checkpoint"]
            ),
        },
        "dataset": {
            "name": values["dataset_name"],
            "split": values["dataset_split"],
            "images_dir": str(
                values["images_dir"]
            ),
            "available_images": (
                available_image_count
            ),
            "selected_images": len(
                selected_images
            ),
            "selection_seed": values["seed"],
        },
        "protocol": {
            "device": values["device"],
            "image_size": values["image_size"],
            "batch_size": values["batch_size"],
            "warmup_runs": values["warmup_runs"],
            "repetitions_per_image": values[
                "repetitions_per_image"
            ],
            "confidence_threshold": values[
                "confidence_threshold"
            ],
            "iou_threshold": values[
                "iou_threshold"
            ],
            "max_detections": values[
                "max_detections"
            ],
            "timing_scope": values[
                "timing_scope"
            ],
            "disk_io_included": values[
                "disk_io_included"
            ],
            "raw_samples_saved": True,
        },
        "environment": {
            "system": platform.platform(),
            "architecture": platform.machine(),
            "python_version": (
                platform.python_version()
            ),
            "torch_version": torch.__version__,
            "ultralytics_version": (
                ultralytics.__version__
            ),
            "numpy_version": np.__version__,
            "torch_num_threads": (
                torch.get_num_threads()
            ),
        },
        "summary": summarize_latencies(
            samples
        ),
        "selected_image_files": [
            str(
                path.relative_to(
                    values["images_dir"]
                )
            )
            for path in selected_images
        ],
        "raw_samples": samples,
    }


def save_result(
    result: Dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Run the standardized "
            "EdgeNAS-Lite CPU benchmark."
        )
    )

    argument_parser.add_argument(
        "config_file",
        help="Path to benchmark YAML config.",
    )

    arguments = argument_parser.parse_args()

    try:
        config = load_config(
            Path(arguments.config_file)
        )
        values = read_config_values(config)

        if not values["checkpoint"].exists():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{values['checkpoint']}"
            )

        all_images = collect_images(
            values["images_dir"]
        )

        selected_images = select_images(
            all_images,
            values["sample_size"],
            values["seed"],
        )

        print(
            f"Loading model: {values['checkpoint']}"
        )
        model = YOLO(
            str(values["checkpoint"])
        )

        print(
            f"Preloading {len(selected_images)} images..."
        )
        preloaded_images = preload_images(
            selected_images
        )

        print(
            f"Running {values['warmup_runs']} warm-up runs..."
        )
        run_warmup(
            model,
            preloaded_images,
            values,
        )

        measurement_count = (
            len(preloaded_images)
            * values["repetitions_per_image"]
        )

        print(
            f"Measuring {measurement_count} predictions..."
        )
        samples = measure_latency(
            model,
            preloaded_images,
            values,
        )

        result = build_result(
            values,
            len(all_images),
            selected_images,
            samples,
        )

        save_result(
            result,
            values["result_file"],
        )

    except (
        FileNotFoundError,
        BenchmarkConfigError,
    ) as error:
        argument_parser.error(str(error))

    summary = result["summary"]

    print("\nBenchmark completed")
    print(
        f"Median latency: "
        f"{summary['median_latency_ms']} ms"
    )
    print(
        f"P95 latency: "
        f"{summary['p95_latency_ms']} ms"
    )
    print(
        f"FPS from median: "
        f"{summary['fps_from_median']}"
    )
    print(
        f"Result saved to: "
        f"{values['result_file']}"
    )


if __name__ == "__main__":
    main()