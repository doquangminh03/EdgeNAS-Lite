"""Plan or execute a provenance-tracked CPU benchmark campaign."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Sequence

import yaml


DEFAULT_RESOLUTIONS = (416, 448, 480, 512, 576, 608, 640)
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class CampaignError(ValueError):
    """Raised when a campaign cannot be planned safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CampaignError("Cannot summarize an empty latency sample.")
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_session_schedule(
    resolutions: Sequence[int],
    session_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    schedule = []
    for session_number in range(1, session_count + 1):
        order = list(resolutions)
        random.Random(seed + session_number).shuffle(order)
        schedule.append({"session": session_number, "resolution_order": order})
    return schedule


def summarize_sessions(
    session_results: Sequence[Dict[str, Any]],
    excluded_sessions: Sequence[int],
) -> Dict[str, Any]:
    included = [
        result for result in session_results
        if result["session"] not in excluded_sessions
    ]
    if not included:
        raise CampaignError("No sessions remain after stabilization exclusion.")

    pooled = [
        sample["latency_ms"]
        for result in included
        for sample in result["raw_samples"]
    ]
    medians = [result["median_latency_ms"] for result in included]
    return {
        "excluded_stabilization_sessions": list(excluded_sessions),
        "included_sessions": [result["session"] for result in included],
        "per_session": [
            {
                "session": result["session"],
                "sample_count": len(result["raw_samples"]),
                "median_latency_ms": result["median_latency_ms"],
                "p95_latency_ms": result["p95_latency_ms"],
            }
            for result in session_results
        ],
        "pooled_sample_count": len(pooled),
        "pooled_median_latency_ms": statistics.median(pooled),
        "pooled_p95_latency_ms": percentile(pooled, 95),
        "session_median_mean_ms": statistics.mean(medians),
        "session_median_population_sd_ms": (
            statistics.pstdev(medians) if len(medians) > 1 else 0.0
        ),
        "inference_note": (
            "Repeated image timings are not independent observations; "
            "use per-session summaries to compare variability."
        ),
    }


def _git_value(root: Path, *arguments: str) -> Any:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def make_plan(
    config_path: Path,
    *,
    root: Path,
    run_id: str,
    hardware_id: str,
    output_dir: Path,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    session_count: int = 5,
    stabilization_sessions: int = 2,
    seed: int = 20261024,
    cpu_threads: int = 1,
) -> Dict[str, Any]:
    root = root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise CampaignError("run_id may contain only letters, digits, '_' and '-'.")
    if not hardware_id.strip() or any(character.isspace() for character in hardware_id):
        raise CampaignError("hardware_id must be a non-empty identifier without whitespace.")
    if not resolutions or len(set(resolutions)) != len(resolutions) or any(type(size) is not int or size <= 0 or size % 32 for size in resolutions):
        raise CampaignError("resolutions must be distinct positive integers divisible by 32.")
    if session_count < 2 or not 0 <= stabilization_sessions < session_count:
        raise CampaignError("Invalid session or stabilization-session count.")
    if cpu_threads <= 0:
        raise CampaignError("cpu_threads must be greater than zero.")

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CampaignError(f"Cannot read benchmark config: {error}") from error
    if not isinstance(config, dict):
        raise CampaignError("Benchmark config must be a YAML mapping.")

    checkpoint = (root / config["model"]["checkpoint"]).resolve()
    image_dir = (root / config["dataset"]["images_dir"]).resolve()
    if not checkpoint.is_file():
        raise CampaignError(f"Checkpoint not found: {checkpoint}")
    images = sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    sample_size = int(config["selection"]["sample_size"])
    if len(images) < sample_size:
        raise CampaignError(
            f"Need {sample_size} images but found {len(images)} in {image_dir}."
        )
    selected = sorted(
        random.Random(int(config["selection"]["seed"])).sample(images, sample_size)
    )
    image_manifest = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for path in selected
    ]
    manifest_bytes = json.dumps(
        image_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    protocol = config["protocol"]
    if protocol.get("device", "cpu") != "cpu" or protocol.get("batch_size", 1) != 1:
        raise CampaignError("Campaign requires CPU and batch_size 1.")
    if protocol["timing_scope"] != "preprocess_inference_postprocess" or protocol["disk_io_included"] is not False:
        raise CampaignError("Campaign requires declared end-to-end timing excluding disk I/O.")
    if sample_size <= 0 or int(protocol["warmup_runs"]) < 0 or int(protocol["repetitions_per_image"]) <= 0:
        raise CampaignError("Invalid sample size, warmup or repetitions.")
    schedule = build_session_schedule(resolutions, session_count, seed)
    return {
        "schema_version": "1.0",
        "campaign_id": run_id,
        "status": "planned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value(root, "rev-parse", "HEAD"),
        "code_dirty": (
            None
            if _git_value(root, "status", "--porcelain") is None
            else bool(_git_value(root, "status", "--porcelain"))
        ),
        "hardware_id": hardware_id,
        "hardware": {
            "system": platform.platform(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        },
        "software": {
            "torch": _package_version("torch"),
            "ultralytics": _package_version("ultralytics"),
            "numpy": _package_version("numpy"),
        },
        "source_sha256": {
            str(path): sha256_file(root / path)
            for path in (Path("scripts/benchmark_campaign.py"), Path("src/benchmarking/cpu_benchmark.py"))
            if (root / path).is_file()
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(root)),
            "sha256": sha256_file(checkpoint),
        },
        "dataset": {
            "name": config["dataset"]["name"],
            "split": config["dataset"]["split"],
            "sample_count": len(image_manifest),
            "selection_seed": int(config["selection"]["seed"]),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "images": image_manifest,
        },
        "protocol": {
            "version": "cpu_campaign_v2",
            "benchmark_version": "cpu_threads_v2",
            "backend_setup_predictions": 1,
            "batch_size": 1,
            "cpu_threads": cpu_threads,
            "warmup_runs_per_session": int(protocol["warmup_runs"]),
            "repetitions_per_image": int(protocol["repetitions_per_image"]),
            "sessions": session_count,
            "stabilization_sessions_excluded": list(
                range(1, stabilization_sessions + 1)
            ),
            "timing_scope": protocol["timing_scope"],
            "disk_io_included": protocol["disk_io_included"],
            "resolution_order_seed": seed,
        },
        "schedule": schedule,
        "output_dir": str(output_dir),
        "results": [],
        "summary": None,
        "limitations": [
            "A campaign is not registered as evidence until every scheduled run completes.",
            f"The first {stabilization_sessions} sessions are excluded by a predeclared stabilization rule.",
            "Pooled image timings are repeated measures, not independent observations.",
        ],
    }


def _package_version(name: str) -> Any:
    try:
        from importlib.metadata import version
    except ImportError:
        from importlib_metadata import version
    try:
        return version(name)
    except Exception:
        return None


def validate_thread_evidence(result: Dict[str, Any], requested: int) -> None:
    """Reject legacy/missing evidence, not just explicitly wrong thread counts."""
    protocol = result.get("protocol", {})
    environment = result.get("environment", {})
    if protocol.get("version") != "cpu_threads_v2":
        raise CampaignError("Missing cpu_threads_v2 thread protocol evidence.")
    for field, actual in (
        ("protocol.requested_cpu_threads", protocol.get("requested_cpu_threads")),
        ("environment.requested_cpu_threads", environment.get("requested_cpu_threads")),
        ("environment.torch_num_threads", environment.get("torch_num_threads")),
    ):
        if type(actual) is not int or actual != requested:
            raise CampaignError(f"Thread mismatch: {field}: requested {requested}, observed {actual}.")
    observations = environment.get("thread_observations", {})
    for stage in ("before_warmup", "after_warmup", "before_timing", "after_timing"):
        actual = observations.get(stage)
        if type(actual) is not int or actual != requested:
            raise CampaignError(f"Thread mismatch or missing observation at {stage}.")
    samples = result.get("raw_samples", [])
    if not samples:
        raise CampaignError("Missing per-sample thread observations.")
    for sample in samples:
        for field in ("torch_num_threads_before", "torch_num_threads_after"):
            actual = sample.get(field)
            if type(actual) is not int or actual != requested:
                raise CampaignError(f"Thread mismatch or missing per-sample observation: {field}.")


def _execute_campaign(
    plan: Dict[str, Any],
    *,
    base_config: Path,
    output_dir: Path,
    root: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    plan["status"] = "running"
    plan_path = output_dir / "campaign.json"
    _write_json(plan_path, plan)
    image_manifest_path = output_dir / "image_manifest.json"
    _write_json(image_manifest_path, plan["dataset"])

    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    excluded = plan["protocol"]["stabilization_sessions_excluded"]
    by_resolution: Dict[int, List[Dict[str, Any]]] = {
        size: [] for size in plan["schedule"][0]["resolution_order"]
    }
    for session in plan["schedule"]:
        session_number = session["session"]
        for resolution in session["resolution_order"]:
            session_dir = output_dir / f"session_{session_number:02d}"
            session_dir.mkdir(parents=True, exist_ok=True)
            result_path = session_dir / f"imgsz_{resolution}.json"
            session_config = deepcopy(config)
            session_config["benchmark_id"] = (
                f"{plan['campaign_id']}_s{session_number}_imgsz{resolution}"
            )
            session_config["candidate_id"] = (
                f"yolo26n_kitti_pilot_imgsz{resolution}_cpu"
            )
            session_config["model"]["checkpoint"] = os.path.relpath((root / config["model"]["checkpoint"]).resolve(), root)
            session_config["dataset"]["images_dir"] = os.path.relpath((root / config["dataset"]["images_dir"]).resolve(), root)
            session_config["protocol"]["image_size"] = resolution
            session_config["protocol"]["cpu_threads"] = plan["protocol"]["cpu_threads"]
            session_config["campaign_id"] = plan["campaign_id"]
            session_config["session_id"] = session_number
            session_config["output"]["result_file"] = str(result_path)
            config_path = session_dir / f"benchmark_imgsz_{resolution}.yaml"
            with config_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(session_config, stream, sort_keys=False)
            command = [
                sys.executable,
                "-m",
                "src.benchmarking.cpu_benchmark",
                str(config_path),
                "--hardware-id",
                plan["hardware_id"],
                "--threads",
                str(plan["protocol"]["cpu_threads"]),
                "--campaign-id",
                plan["campaign_id"],
                "--session-id",
                str(session_number),
            ]
            if sha256_file(root / session_config["model"]["checkpoint"]) != plan["checkpoint"]["sha256"]:
                raise CampaignError("Checkpoint changed since preflight.")
            for image in plan["dataset"]["images"]:
                if sha256_file(root / image["path"]) != image["sha256"]:
                    raise CampaignError("Image changed since preflight: " + image["path"])
            for source, expected_hash in plan["source_sha256"].items():
                if sha256_file(root / source) != expected_hash:
                    raise CampaignError("Source changed since preflight: " + source)
            subprocess.run(command, cwd=root, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_thread_evidence(result, plan["protocol"]["cpu_threads"])
            expected_images = [str((root / image["path"]).relative_to(root / session_config["dataset"]["images_dir"])) for image in plan["dataset"]["images"]]
            if result["selected_image_files"] != expected_images:
                raise CampaignError("Measured image list differs from campaign manifest.")
            expected_count = len(expected_images) * plan["protocol"]["repetitions_per_image"]
            samples = result["raw_samples"]
            if len(samples) != expected_count or any(sample["image"] not in expected_images for sample in samples):
                raise CampaignError("Raw sample count or image identity mismatch.")
            session_result = {
                "session": session_number,
                "resolution": resolution,
                "requested_cpu_threads": plan["protocol"]["cpu_threads"],
                "observed_cpu_threads": result["environment"]["torch_num_threads"],
                "raw_samples": result["raw_samples"],
                "median_latency_ms": result["summary"]["median_latency_ms"],
                "p95_latency_ms": result["summary"]["p95_latency_ms"],
                "result_file": str(result_path.relative_to(root)),
            }
            by_resolution[resolution].append(session_result)
            plan["results"].append(session_result)
            plan_path.write_text(
                json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    plan["summary"] = {
        str(resolution): summarize_sessions(sessions, excluded)
        for resolution, sessions in by_resolution.items()
    }
    plan["status"] = "completed"
    plan_path.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return plan


def execute_campaign(plan, *, base_config, output_dir, root):
    try:
        return _execute_campaign(plan, base_config=base_config, output_dir=output_dir, root=root)
    except (Exception, KeyboardInterrupt) as error:
        if plan.get("status") == "running":
            plan["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            plan["error"] = type(error).__name__ + ": " + str(error)
            (output_dir / "campaign.json").write_text(
                json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark_cpu.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resolutions", nargs="+", type=int, default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--exclude-stabilization-sessions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true", help="Run measurements; default is preflight only.")
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("campaign_%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/benchmarks/campaigns") / run_id
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        if output_dir.exists():
            raise CampaignError(f"Campaign output already exists: {output_dir}")
        plan = make_plan(
            config_path,
            root=root,
            run_id=run_id,
            hardware_id=args.hardware_id,
            output_dir=output_dir,
            resolutions=args.resolutions,
            session_count=args.sessions,
            stabilization_sessions=args.exclude_stabilization_sessions,
            seed=args.seed,
            cpu_threads=args.threads,
        )
        if args.execute:
            plan = execute_campaign(
                plan,
                base_config=config_path,
                output_dir=output_dir,
                root=root,
            )
            print(f"Campaign status: {plan['status']}")
            print(f"Campaign report: {output_dir / 'campaign.json'}")
        else:
            print(json.dumps({
                "campaign_id": plan["campaign_id"],
                "status": plan["status"],
                "resolution_count": len(args.resolutions),
                "session_count": args.sessions,
                "sampled_images": plan["dataset"]["sample_count"],
                "checkpoint_sha256": plan["checkpoint"]["sha256"],
                "dataset_manifest_sha256": plan["dataset"]["manifest_sha256"],
                "schedule": plan["schedule"],
                "output_dir": str(output_dir),
                "code_dirty": plan["code_dirty"],
            }, indent=2, ensure_ascii=False, allow_nan=False))
            print("PREFLIGHT ONLY: no files written and no measurements run.")
        return 0
    except (CampaignError, OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Campaign error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
