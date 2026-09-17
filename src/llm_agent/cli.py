"""Interpret a natural-language requirement and propose a known candidate."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from src.llm_agent.gemini_provider import (
    GeminiProvider,
    GeminiProviderError,
)
from src.llm_agent.interpreter import interpret_requirement
from src.proposal.rule_based import propose_from_knowledge


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Interpret a natural-language requirement with Gemini "
            "and select a candidate from the knowledge database."
        )
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Natural-language requirement.",
    )
    parser.add_argument(
        "--request-id",
        required=True,
        help="Request identifier: letters, digits, underscores or hyphens.",
    )
    parser.add_argument(
        "--hardware-id",
        required=True,
        help="Target hardware identity used for compatibility checks.",
    )
    parser.add_argument(
        "--model-family",
        default="YOLO26",
        help="Candidate model family. Default: YOLO26.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash-lite",
        help="Gemini model identifier.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory.",
    )
    parser.add_argument(
        "--index",
        default="knowledge/index.yaml",
        help="Knowledge index, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "New JSON report path, relative to the project root "
            "or absolute. Existing files are not overwritten."
        ),
    )
    return parser


def resolve_path(root, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        root = Path(args.project_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("Project root must be an existing directory.")

        hardware_id = args.hardware_id.strip()
        model_family = args.model_family.strip()

        if not hardware_id:
            raise ValueError("hardware-id must not be empty.")
        if not model_family:
            raise ValueError("model-family must not be empty.")

        index_path = resolve_path(root, args.index)
        if not index_path.is_file():
            raise ValueError("Knowledge index file does not exist.")

        output_path = resolve_path(root, args.output)
        if output_path.suffix.lower() != ".json":
            raise ValueError("Output file must have a .json extension.")
        if output_path.exists():
            raise ValueError(
                "Output already exists. Choose a new output filename."
            )

        provider = GeminiProvider(model=args.model)

        print("Interpreting requirement with Gemini...", flush=True)

        interpretation = interpret_requirement(
            args.text,
            request_id=args.request_id,
            provider=provider,
        )

        print("Interpretation:", interpretation["status"])
        proposal = None

        if interpretation["status"] == "ready":
            with TemporaryDirectory(
                prefix="edgenas_requirement_"
            ) as temp_dir:
                requirement_path = Path(temp_dir) / "requirement.yaml"

                # JSON is valid YAML for the existing requirement loader.
                requirement_path.write_text(
                    json.dumps(
                        interpretation["requirement"],
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    ) + "\n",
                    encoding="utf-8",
                )

                proposal = propose_from_knowledge(
                    requirement_path,
                    project_root=root,
                    index_file=index_path,
                    model_family=model_family,
                    expected_hardware_id=hardware_id,
                )

            print("Proposal:", proposal["proposal_status"])

            source = proposal["selected_source"]
            print(
                "Selected:",
                source["candidate_id"] if source else "None",
            )
        else:
            for question in interpretation["questions"]:
                print("Question:", question)
            for reason in interpretation["reasons"]:
                print("Reason:", reason)

        report = {
            "schema_version": "1.0",
            "request_id": args.request_id,
            "user_text": args.text.strip(),
            "provider": {
                "name": "gemini",
                "model": args.model,
            },
            "expected_hardware_id": hardware_id,
            "model_family": model_family,
            "interpretation": interpretation,
            "proposal": proposal,
        }

        serialized = json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation also prevents accidental overwrites
        # if a file appears after the initial existence check.
        with output_path.open("x", encoding="utf-8") as output:
            output.write(serialized)

        print("Saved:", output_path)

        if proposal and proposal["proposal_status"] == "selected":
            return 0

        return 1

    except (GeminiProviderError, ValueError, OSError) as error:
        print("Error:", error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())