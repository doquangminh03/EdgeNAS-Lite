import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import yaml

from src.proposal.output import save_proposal
from src.proposal.rule_based import propose_from_knowledge


def hardware_identifier(value: str) -> str:
    identifier = value.strip()

    if not identifier or identifier == "<MISSING>":
        raise argparse.ArgumentTypeError(
            "Hardware ID must be a non-empty identifier."
        )

    return identifier


def resolve_project_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = root / path

    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a measured candidate from the Knowledge DB "
            "using compatibility checks and deployment constraints."
        ),
        allow_abbrev=False,
    )

    parser.add_argument(
        "requirement",
        help="Path to the requirement YAML or JSON file.",
    )
    parser.add_argument(
        "--hardware-id",
        required=True,
        type=hardware_identifier,
        help="Target hardware identifier used for compatibility checks.",
    )
    parser.add_argument(
        "--model-family",
        default=None,
        help="Optional model-family filter, for example YOLO26.",
    )
    parser.add_argument(
        "--index",
        default="knowledge/index.yaml",
        help="Knowledge DB index path.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Base directory for relative paths; defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path. An existing proposal file will be replaced.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        root = Path(args.project_root).expanduser().resolve()
        requirement_path = resolve_project_path(args.requirement, root)
        index_path = resolve_project_path(args.index, root)
        output_path = resolve_project_path(args.output, root)

        # Prevent accidental replacement of configuration inputs.
        if output_path in {requirement_path, index_path}:
            raise ValueError(
                "Output must differ from the requirement and index files."
            )

        proposal = propose_from_knowledge(
            requirement_path,
            index_file=index_path,
            project_root=root,
            model_family=args.model_family,
            expected_hardware_id=args.hardware_id,
        )

        # Candidate records are evidence, so output must not replace them.
        source_paths = {
            resolve_project_path(item["record_path"], root)
            for item in proposal["candidates"]
        }

        if output_path in source_paths:
            raise ValueError(
                "Output must not replace a retrieved candidate record."
            )

        saved_path = save_proposal(proposal, output_path)

    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print("Request:", proposal["request_id"])
    print("Hardware:", proposal["expected_hardware_id"])
    print("Status:", proposal["proposal_status"])
    print("Retrieved:", proposal["candidate_count"])
    print("Compatible:", len(proposal["compatible_candidate_ids"]))
    print("Excluded:", len(proposal["excluded_candidate_ids"]))
    print("Feasible:", len(proposal["feasible_candidate_ids"]))

    selection = proposal["selection"]
    selected_id = (
        selection["selected_candidate_id"]
        if selection is not None
        else None
    )

    print("Selected:", selected_id or "None")
    print("Saved:", saved_path)

    if proposal["proposal_status"] == "selected":
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())