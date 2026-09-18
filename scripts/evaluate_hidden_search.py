"""Live LLM search over progressively revealed archived measurements."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import time


def evaluate_order(order, initial, oracle, ranking):
    seen = set(initial)
    rows = []
    for step in range(len(order) + 1):
        ids = {oracle[size]["candidate_id"] for size in seen}
        best = next(
            (row for row in ranking if row["candidate_id"] in ids),
            None,
        )
        rows.append({
            "step": step,
            "selected_candidate_id": best["candidate_id"] if best else None,
            "best_score": best["score"] if best else None,
        })
        if step < len(order):
            seen.add(order[step])
    return rows


def run_trial(provider, propose, pool, requirement, oracle,
              initial, budget, ranking, save):
    seen = list(initial)
    trial = {
        "status": "running",
        "initial_sizes": list(initial),
        "order": [],
        "calls": [],
    }
    save(deepcopy(trial))
    try:
        for _ in range(budget):
            available = sorted(set(oracle) - set(seen))
            if not available:
                break

            # Hidden metrics and final ranking never enter this context.
            context = {
                "available_image_sizes": available,
                "measured_evidence": [
                    deepcopy(oracle[size]) for size in seen
                ],
            }
            call = {
                "status": "requesting",
                "visible_sizes": list(seen),
                "available_sizes": available,
            }
            trial["calls"].append(call)
            save(deepcopy(trial))

            start = time.perf_counter()
            try:
                proposal = propose(
                    provider, deepcopy(requirement),
                    deepcopy(pool), context,
                )
            finally:
                call["elapsed_seconds"] = round(
                    time.perf_counter() - start, 6
                )

            size = proposal["image_size"]
            if type(size) is not int or size not in available:
                raise ValueError("Candidate is unavailable or already seen.")

            call.update(status="completed", proposal=proposal)
            trial["order"].append(size)
            seen.append(size)
            trial["trajectory"] = evaluate_order(
                trial["order"], initial, oracle, ranking
            )
            save(deepcopy(trial))

        trial["status"] = "completed"
        save(deepcopy(trial))
        return trial
    except Exception as error:
        trial.update(status="failed", error=str(error))
        save(deepcopy(trial))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from src.llm_agent.candidate_proposer import (
        load_search_space, request_proposal,
    )
    from src.llm_agent.experiment_controller import save_report
    from src.proposal.compatibility import check_candidate_compatibility

    root = Path.cwd()
    source = (
        root / "results/llm_experiments"
        / "low_latency_batch_001_002.json"
    )
    report = json.loads(source.read_text(encoding="utf-8"))
    if report["status"] != "completed":
        raise ValueError("Source experiment must be completed.")

    requirement = report["requirement"]
    selection = report["final_selection"]["selection"]
    ranking = selection["ranking"]
    pool = load_search_space(
        root / "configs/search_space_llm_pool.yaml",
        check_paths=False,
    )

    # Only fixed settings are passed to the existing proposer.
    public_pool = {
        "fixed_configuration": deepcopy(pool["fixed_configuration"])
    }
    oracle = {}

    for row in ranking + selection["rejected_candidates"]:
        cid = row["candidate_id"]
        record_path = root / "results/candidates" / (cid + ".json")
        record = json.loads(record_path.read_text(encoding="utf-8"))

        compatibility = check_candidate_compatibility(
            record,
            expected_hardware_id=report["hardware_id"],
        )
        if compatibility["status"] != "compatible":
            raise ValueError("Incompatible archived record: " + cid)

        size = record["accuracy"]["image_size"]
        if size != record["benchmark"]["image_size"] or size in oracle:
            raise ValueError("Duplicate or mismatched image size.")

        measured = {
            "map50_95": record["accuracy"]["map50_95"],
            "median_latency_ms": record["benchmark"]["median_latency_ms"],
            "model_size_mb": record["model"]["model_size_mb"],
        }
        if measured != row["metrics"]:
            raise ValueError("Metrics changed since final ranking: " + cid)

        oracle[size] = dict(
            candidate_id=cid, image_size=size, **measured
        )

    expected_sizes = pool["variable_dimensions"]["image_size"]["values"]
    if set(oracle) != set(expected_sizes):
        raise ValueError("Pool and archived candidate set differ.")

    initial = [640]
    remaining = sorted(set(oracle) - set(initial))
    if (
        640 not in oracle
        or not 1 <= args.budget <= len(remaining)
        or not 1 <= args.repeats <= 10
    ):
        raise ValueError("Invalid initial candidate, budget or repeat count.")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Output exists; choose a new output filename.")

    random_orders = list(
        itertools.permutations(remaining, args.budget)
    )
    result = {
        "evaluation_version": "hidden_search_v1",
        "status": "planned",
        "initial_sizes": initial,
        "budget": args.budget,
        "repeats": args.repeats,
        "maximum_provider_calls": args.budget * args.repeats,
        "requirement": requirement,
        "hardware_id": report["hardware_id"],
        "source_report": str(source.relative_to(root)),
        "oracle_snapshot": oracle,
        "selection_snapshot": selection,
        "ascending_order": remaining[:args.budget],
        "ascending_trajectory": evaluate_order(
            remaining[:args.budget], initial, oracle, ranking
        ),
        "random_order_count": len(random_orders),
        "random_trajectories": [
            evaluate_order(order, initial, oracle, ranking)
            for order in random_orders
        ],
        "trials": [],
        "limitations": [
            "Live LLM choices reveal archived measurements, not new benchmarks.",
            "Initial 640 was chosen after inspecting prior results; exploratory evaluation.",
            "Historical measurements have differing software environments.",
            "Repeated LLM calls may be identical and are not necessarily independent.",
            "Elapsed time covers proposal calls, not model evaluation.",
            "Tokens, internal provider retries and monetary cost are not measured.",
        ],
    }

    if args.dry_run:
        print(json.dumps({
            key: result[key]
            for key in (
                "initial_sizes", "budget", "repeats",
                "maximum_provider_calls", "random_order_count",
            )
        }, indent=2))
        print("DRY RUN: no API calls or files written.")
        return

    from src.llm_agent.gemini_provider import GeminiProvider

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    try:
        result["status"] = "running"
        for index in range(args.repeats):
            provider = GeminiProvider(
                **({"model": args.model} if args.model else {})
            )
            result["model"] = getattr(provider, "model", args.model)
            result["trials"].append({})

            def save(trial):
                result["trials"][index] = trial
                save_report(output, result)

            trial = run_trial(
                provider, request_proposal, public_pool, requirement,
                oracle, initial, args.budget, ranking, save,
            )
            print(
                "Trial", index + 1, "order:", trial["order"],
                flush=True,
            )

        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        save_report(output, result)

    print("Saved:", output)


if __name__ == "__main__":
    main()
