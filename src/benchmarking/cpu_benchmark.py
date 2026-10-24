import argparse
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def validate_hardware_id(value: Any) -> Optional[str]:
    """Validate an operator-supplied label; this is not hardware discovery."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BenchmarkConfigError("hardware_id must be a non-empty string without surrounding whitespace.")
    if any(char.isspace() for char in value):
        raise BenchmarkConfigError("hardware_id cannot contain whitespace.")
    return value


def resolve_hardware_id(config: Dict[str, Any], hardware_id: Optional[str]) -> Optional[str]:
    configured = validate_hardware_id(config.get("hardware_id"))
    supplied = validate_hardware_id(hardware_id)
    if configured is not None and supplied is not None and configured != supplied:
        raise BenchmarkConfigError("Configured hardware_id conflicts with --hardware-id.")
    return supplied if supplied is not None else configured


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

    values["hardware_id"] = validate_hardware_id(config.get("hardware_id"))
    values["requested_cpu_threads"] = validate_cpu_threads(
        config["protocol"].get("cpu_threads")
    )
    return values


def validate_cpu_threads(value: Any) -> Optional[int]:
    if value is not None and (type(value) is not int or value <= 0):
        raise BenchmarkConfigError("cpu_threads must be a positive integer.")
    return value


def observe_cpu_threads(values: Dict[str, Any], stage: str) -> int:
    """Check outside the timed predict call; never silently repair later drift."""
    observed = torch.get_num_threads()
    requested = values.get("requested_cpu_threads")
    if requested is not None and observed != requested:
        raise BenchmarkConfigError(
            f"CPU thread mismatch at {stage}: requested {requested}, observed {observed}."
        )
    return observed


def initialize_backend(
    model: YOLO,
    image: np.ndarray,
    values: Dict[str, Any],
) -> None:
    # Public predict initializes the lazy predictor/backend and its internal
    # warmup. This single setup prediction is outside declared warmups/timing.
    run_prediction(model, image, values)
    observations = {"after_backend_setup": torch.get_num_threads()}
    requested = validate_cpu_threads(values.get("requested_cpu_threads"))
    if requested is not None:
        torch.set_num_threads(requested)
    observations["before_warmup"] = observe_cpu_threads(values, "before_warmup")
    values["thread_observations"] = observations


def collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Validation image directory not found: {images_dir}"
        )

    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and not path.name.startswith("._")
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
        observe_cpu_threads(values, "before_warmup_prediction")
        _, image = images[
            run_index % len(images)
        ]

        run_prediction(
            model,
            image,
            values,
        )
        observe_cpu_threads(values, "after_warmup_prediction")

    values.setdefault("thread_observations", {})["after_warmup"] = observe_cpu_threads(
        values, "after_warmup"
    )


def measure_latency(
    model: YOLO,
    images: List[Tuple[Path, np.ndarray]],
    values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    samples = []
    values.setdefault("thread_observations", {})["before_timing"] = observe_cpu_threads(
        values, "before_timing"
    )

    for repetition in range(
        values["repetitions_per_image"]
    ):
        ordered_images = list(images)

        random.Random(
            values["seed"] + repetition
        ).shuffle(ordered_images)

        for image_path, image in ordered_images:
            threads_before = observe_cpu_threads(values, "before_timed_prediction")
            start_time = time.perf_counter_ns()

            run_prediction(
                model,
                image,
                values,
            )

            end_time = time.perf_counter_ns()
            threads_after = observe_cpu_threads(values, "after_timed_prediction")

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
                    "torch_num_threads_before": threads_before,
                    "torch_num_threads_after": threads_after,
                }
            )

    values["thread_observations"]["after_timing"] = observe_cpu_threads(values, "after_timing")
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
    hardware_id = validate_hardware_id(values.get("hardware_id"))
    observed_threads = observe_cpu_threads(values, "build_result")
    return {
        **({"hardware_id": hardware_id} if hardware_id is not None else {}),
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
            "version": "cpu_threads_v2",
            "backend_setup_predictions": 1,
            "requested_cpu_threads": values.get("requested_cpu_threads"),
            "thread_check_scope": "warmup_and_each_timed_prediction_boundaries_outside_timer",
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
            "torch_num_threads": observed_threads,
            "requested_cpu_threads": values.get("requested_cpu_threads"),
            "thread_observations": dict(values.get("thread_observations", {})),
            "campaign_id": values.get("campaign_id"),
            "session_id": values.get("session_id"),
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

    argument_parser.add_argument("--hardware-id", help="Operator-supplied identity of the machine running this benchmark.")
    argument_parser.add_argument("--threads", type=int, default=None)
    argument_parser.add_argument("--campaign-id")
    argument_parser.add_argument("--session-id", type=int)

    arguments = argument_parser.parse_args()

    try:
        config = load_config(
            Path(arguments.config_file)
        )
        hardware_id = resolve_hardware_id(config, arguments.hardware_id)
        if hardware_id is not None:
            config["hardware_id"] = hardware_id
        values = read_config_values(config)
        if arguments.threads is not None:
            requested = validate_cpu_threads(arguments.threads)
            if values["requested_cpu_threads"] not in (None, requested):
                raise BenchmarkConfigError("Configured cpu_threads conflicts with --threads.")
            values["requested_cpu_threads"] = requested
        if "campaign_id" in config:
            values["campaign_id"] = config["campaign_id"]
        if "session_id" in config:
            values["session_id"] = config["session_id"]
        if arguments.campaign_id is not None:
            values["campaign_id"] = arguments.campaign_id
        if arguments.session_id is not None:
            values["session_id"] = arguments.session_id

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

        initialize_backend(model, preloaded_images[0][1], values)

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
