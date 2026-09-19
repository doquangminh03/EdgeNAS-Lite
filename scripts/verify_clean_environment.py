"""Verify dependency installation and tests in a separate virtual environment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


def run_step(name, command, root, folder, env, timeout):
    log = folder / (name + ".log")
    started = time.monotonic()
    print(f"Running: {name} | log: {log}", flush=True)
    result = {"name": name, "log": log.name, "status": "failed"}
    with log.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.run(
                command, cwd=root, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=timeout,
            )
            result["exit_code"] = process.returncode
            result["status"] = "passed" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            result["status"] = "timed_out"
        except OSError as error:
            result["error"] = str(error)
    result["seconds"] = round(time.monotonic() - started, 2)
    print(f"{name}: {result['status']}", flush=True)
    if result["status"] != "passed":
        print("\n".join(log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-25:]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_id
    ):
        parser.error("run-id may contain only letters, digits, underscores, or hyphens")

    root = Path(__file__).resolve().parents[1]
    requirements = root / "requirements.txt"
    if not requirements.is_file() or not (root / "tests").is_dir():
        parser.error("requirements.txt and tests/ must exist in the project root")
    folder = root / "results/reproducibility" / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = requirements.read_bytes()
    (folder / "requirements_snapshot.txt").write_bytes(snapshot)
    report_path = folder / "report.json"
    state = {
        "schema_version": "1.0", "run_id": args.run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clean dependencies; current working tree; same machine",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requirements_sha256": hashlib.sha256(snapshot).hexdigest(),
        "steps": [],
        "limitations": [
            "Not a fresh clone or a different hardware/OS test.",
            "No live Gemini workflow or new model measurement is requested.",
            "Dataset contents and checkpoint contents are not validated.",
        ],
    }

    def save():
        temporary = folder / "report.json.tmp"
        temporary.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(report_path)

    save()
    try:
        environment = Path(
            tempfile.mkdtemp(prefix="edgenas_clean_")
        ) / "venv"
        state["environment_path"] = str(environment)
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        env = os.environ.copy()
        for key in (
            "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
            "GEMINI_API_KEY", "GOOGLE_API_KEY",
        ):
            env.pop(key, None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")

        steps = [
            (
                "create_venv",
                [sys.executable, "-m", "venv", str(environment)],
                120,
            ),
            (
                "install",
                [
                    str(python), "-m", "pip", "--isolated", "install",
                    "--disable-pip-version-check", "-r",
                    str(folder / "requirements_snapshot.txt"),
                ],
                1800,
            ),
            (
                "pip_check",
                [str(python), "-m", "pip", "check"],
                120,
            ),
            (
                "imports",
                [
                    str(python), "-c",
                    "import torch, ultralytics, yaml; print('Model imports OK')",
                ],
                180,
            ),
            (
                "tests",
                [
                    str(python), "-m", "unittest",
                    "discover", "-s", "tests", "-v",
                ],
                600,
            ),
            (
                "workflow_help",
                [
                    str(python), "-m",
                    "src.llm_agent.workflow", "--help",
                ],
                120,
            ),
        ]
        save()
        for name, command, timeout in steps:
            state["active_step"] = name
            save()
            result = run_step(
                name, command, root, folder, env, timeout
            )
            state["steps"].append(result)
            save()
            if result["status"] != "passed":
                state["status"] = "failed"
                break
        else:
            state["status"] = "passed"

        state["requirements_unchanged"] = (
            requirements.read_bytes() == snapshot
        )
        if not state["requirements_unchanged"]:
            state["status"] = "needs_review"

    except KeyboardInterrupt:
        state["status"] = "interrupted"
    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()

    print("Result:", state["status"])
    print("Report:", report_path)
    print("The temporary environment is retained for inspection.")
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
