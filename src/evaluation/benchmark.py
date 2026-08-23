import json
import platform
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "kitti_yolo26n_pilot"
    / "weights"
    / "best.pt"
)
IMAGE_PATH = PROJECT_ROOT / "bus.jpg"
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pilot_cpu_benchmark.json"
)

IMAGE_SIZE = 640
WARMUP_RUNS = 5
BENCHMARK_RUNS = 30


def validate_files() -> None:
    """Kiểm tra model và ảnh đầu vào có tồn tại hay không."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")


def calculate_model_parameters(model: YOLO) -> int:
    """Tính tổng số parameter của model."""

    return sum(parameter.numel() for parameter in model.model.parameters())


def run_benchmark() -> dict:
    """Đo latency, FPS, model size và số parameters."""

    validate_files()

    print("Loading baseline model...")
    model = YOLO(str(MODEL_PATH))

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError(f"Unable to read image: {IMAGE_PATH}")

    print(f"Running {WARMUP_RUNS} warm-up iterations...")

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )

    print(f"Running {BENCHMARK_RUNS} benchmark iterations...")

    latency_values = []

    for _ in range(BENCHMARK_RUNS):
        start_time = time.perf_counter()

        model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000
        latency_values.append(latency_ms)

    average_latency = statistics.mean(latency_values)
    median_latency = statistics.median(latency_values)
    min_latency = min(latency_values)
    max_latency = max(latency_values)

    parameters = calculate_model_parameters(model)
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    fps = 1000 / median_latency

    return {
        "candidate_id": "yolo26n_kitti_pilot",
        "model": MODEL_PATH.name,
        "device": "cpu",
        "system": platform.platform(),
        "architecture": platform.machine(),
        "image_size": IMAGE_SIZE,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "parameters": parameters,
        "parameters_m": round(parameters / 1_000_000, 3),
        "model_size_mb": round(model_size_mb, 3),
        "average_latency_ms": round(average_latency, 3),
        "median_latency_ms": round(median_latency, 3),
        "min_latency_ms": round(min_latency, 3),
        "max_latency_ms": round(max_latency, 3),
        "fps": round(fps, 3),
    }


def save_results(results: dict) -> None:
    """Lưu kết quả benchmark thành JSON."""

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"\nResults saved to: {RESULT_PATH}")


def print_results(results: dict) -> None:
    """Hiển thị kết quả trên Terminal."""

    print("\nBaseline benchmark completed")
    print("-" * 40)
    print(f"Model:          {results['model']}")
    print(f"Parameters:     {results['parameters_m']} M")
    print(f"Model size:     {results['model_size_mb']} MB")
    print(f"Median latency: {results['median_latency_ms']} ms")
    print(f"Average latency:{results['average_latency_ms']} ms")
    print(f"FPS:            {results['fps']}")
    print("-" * 40)


def main() -> None:
    results = run_benchmark()
    save_results(results)
    print_results(results)


if __name__ == "__main__":
    main()