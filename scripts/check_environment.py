"""Check local prerequisites without API calls or model execution."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("demo", "evaluate"), default="demo",
        help="demo checks recorded evidence; evaluate also checks local assets.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = []

    def check(label, ok, detail):
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Project: {root}")
    print(f"Mode: {args.mode}")

    check(
        "Working directory", Path.cwd().resolve() == root,
        "Run commands from the project root.",
    )

    requirements = root / "requirements.txt"
    check("Requirements file", requirements.is_file(), str(requirements))
    if requirements.is_file():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" not in line:
                check("Requirement format", False, line)
                continue
            name, expected = line.split("==", 1)
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                check(name, False, "not installed")
                continue
            check(name, actual == expected,
                  f"installed={actual}; expected={expected}")

    try:
        import yaml
    except ImportError:
        check("YAML loader", False, "PyYAML is required.")
        return 1

    index_path = root / "knowledge/index.yaml"
    try:
        index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
        entries = index["candidate_records"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("candidate_records must be a non-empty list")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        check("Knowledge index", False, str(error))
        return 1

    checkpoints = set()
    for entry in entries:
        try:
            record_path = root / entry["record_path"]
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["candidate_id"] != entry["candidate_id"]:
                raise ValueError("candidate_id does not match the index")
            checkpoint = record["model"]["checkpoint"]
            if not isinstance(checkpoint, str) or not checkpoint:
                raise ValueError("checkpoint path is missing")
            checkpoints.add(checkpoint)
            check(entry["candidate_id"], True, entry["record_path"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            check("Candidate record", False, str(error))

    if args.mode == "evaluate":
        for checkpoint in sorted(checkpoints):
            path = root / checkpoint
            check("Checkpoint", path.is_file(), str(path))

        data_config = root / "kitti.yaml"
        if not data_config.is_file():

            spec = importlib.util.find_spec("ultralytics")
            matches = []
            if spec and spec.submodule_search_locations:
                for location in spec.submodule_search_locations:
                    matches.extend(
                        Path(location).rglob("kitti.yaml")
                    )
            matches = sorted(set(matches))
            if len(matches) == 1:
                data_config = matches[0]
            elif len(matches) > 1:
                print("Multiple packaged KITTI configurations found:")
                for match in matches:
                    print(f"  {match}")

        check("Dataset configuration", data_config.is_file(), str(data_config))
        print("Dataset image paths and checkpoint contents are not validated.")

    print("API credentials and connectivity are not checked.")
    print("Recorded latency applies to the hardware that produced the evidence.")
    print(f"Result: {len(failures)} failed check(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
