"""Natural-language selection with optional budgeted experiments."""
import argparse
import json
from pathlib import Path
import re
import sys


def run_workflow(args, *, interpret_cli=None, batch_runner=None, ranker=None):
    from src.llm_agent.experiment_controller import save_report

    root = Path.cwd().resolve()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Invalid run ID.")
    if type(args.budget) is not int or not 0 <= args.budget <= 7:
        raise ValueError("Budget must be between 0 and 7.")
    if (
        not args.text.strip()
        or not args.hardware_id
        or any(c.isspace() for c in args.hardware_id)
    ):
        raise ValueError(
            "Text and a hardware ID without whitespace are required."
        )
    if args.budget and not args.template_candidate_id:
        raise ValueError(
            "--template-candidate-id is required when budget > 0."
        )
    if not (root / "knowledge/index.yaml").is_file():
        raise ValueError(
            "Run from the project root containing knowledge/index.yaml."
        )

    folder = root / "results/workflows" / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / "workflow.json"

    state = {
        "schema_version": "1.0",
        "run_id": args.run_id,
        "status": "interpreting",
        "user_text": args.text,
        "hardware_id": args.hardware_id,
        "experiment_budget": args.budget,
        "model": args.model,
    }
    save_report(output, state)

    try:
        if interpret_cli is None:
            from src.llm_agent.cli import main as interpret_cli

        interpretation_path = folder / "interpretation.json"
        exit_code = interpret_cli([
            "--text", args.text,
            "--request-id", args.run_id,
            "--hardware-id", args.hardware_id,
            "--model", args.model,
            "--project-root", str(root),
            "--output", str(interpretation_path),
        ])

        if exit_code not in (0, 1) or not interpretation_path.is_file():
            raise RuntimeError(
                "Requirement interpretation failed; "
                "inspect the preceding CLI error."
            )

        interpreted = json.loads(
            interpretation_path.read_text(encoding="utf-8")
        )
        state["interpretation"] = interpreted["interpretation"]
        state["initial_proposal"] = interpreted["proposal"]

        status = state["interpretation"]["status"]
        if status != "ready":
            state["status"] = status
            save_report(output, state)
            return state

        requirement = folder / "requirement.yaml"
        with requirement.open("x", encoding="utf-8") as stream:
            json.dump(
                state["interpretation"]["requirement"],
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")

        state["requirement_file"] = str(requirement.relative_to(root))
        state["final_proposal"] = interpreted["proposal"]
        save_report(output, state)

        if args.budget:
            if batch_runner is None:
                from src.llm_agent.budget_controller import (
                    run_budget as batch_runner,
                )

            state["status"] = "searching"
            state["batch_id"] = "workflow_" + args.run_id
            save_report(output, state)

            batch = batch_runner(
                root=root,
                batch_id=state["batch_id"],
                budget=args.budget,
                requirement_file=str(requirement.relative_to(root)),
                hardware_id=args.hardware_id,
                template_candidate_id=args.template_candidate_id,
                model=args.model,
            )
            state["batch"] = batch
            save_report(output, state)

            if batch["status"] not in ("budget_finished", "exhausted"):
                raise RuntimeError(
                    "Batch did not finish successfully; inspect its report."
                )

            if ranker is None:
                from src.proposal.rule_based import (
                    propose_from_knowledge as ranker,
                )

            state["final_proposal"] = ranker(
                requirement,
                project_root=root,
                index_file=root / "knowledge/index.yaml",
                model_family="YOLO26",
                expected_hardware_id=args.hardware_id,
            )

        state["status"] = "completed"
        state["selection_status"] = state["final_proposal"]["proposal_status"]
        save_report(output, state)
        return state

    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        save_report(output, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Maximum experiment slots; 0 selects known candidates only.",
    )
    parser.add_argument("--template-candidate-id")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    args = parser.parse_args(argv)

    try:
        state = run_workflow(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print("Workflow:", state["status"])
    source = (state.get("final_proposal") or {}).get("selected_source")
    print("Selected:", source["candidate_id"] if source else "None")
    print("Report:", f"results/workflows/{args.run_id}/workflow.json")
    return 0 if state.get("selection_status") == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
