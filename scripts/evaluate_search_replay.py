"""Replay measured search orders; this is not a fresh controlled experiment."""
import itertools
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = [
    "low_latency_cycle_001",
    "low_latency_batch_001_001",
    "low_latency_batch_001_002",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    reports = [
        read_json(ROOT / "results/llm_experiments" / f"{run_id}.json")
        for run_id in RUN_IDS
    ]

    requirement = reports[0]["requirement"]
    hardware_id = reports[0]["hardware_id"]

    for report in reports:
        if report["status"] != "completed":
            raise ValueError("All three experiments must be completed.")
        if report["requirement"] != requirement:
            raise ValueError("Requirements differ between experiments.")
        if report["hardware_id"] != hardware_id:
            raise ValueError("Hardware IDs differ between experiments.")

    initial_evidence = reports[0]["context"]["measured_evidence"]
    initial_ids = {item["candidate_id"] for item in initial_evidence}
    sizes = {
        item["candidate_id"]: item["image_size"]
        for item in initial_evidence
    }

    actual_order = []
    seen = set(initial_ids)

    for report in reports:
        context_ids = {
            item["candidate_id"]
            for item in report["context"]["measured_evidence"]
        }
        if context_ids != seen:
            raise ValueError(
                "Evidence changed outside the expected sequence; "
                "inspect the reports before comparing."
            )

        candidate_id = report["candidate_id"]
        if candidate_id in seen:
            raise ValueError("An experiment repeats an existing candidate.")

        sizes[candidate_id] = report["proposal"]["image_size"]
        actual_order.append(candidate_id)
        seen.add(candidate_id)

    selection = reports[-1]["final_selection"]["selection"]
    ranking = selection["ranking"]
    feasible = {item["candidate_id"]: item for item in ranking}
    rejected = {
        item["candidate_id"]: item
        for item in selection["rejected_candidates"]
    }

    if set(feasible) & set(rejected):
        raise ValueError("A candidate is both feasible and rejected.")
    if set(feasible) | set(rejected) != seen:
        raise ValueError("Final selection does not cover the replay candidates.")

    def best(available):
        # Preserve the actual selector's order, including tie-breaking.
        return next(
            (item for item in ranking if item["candidate_id"] in available),
            None,
        )

    initial_best = best(initial_ids)
    if initial_best is None:
        raise ValueError("This evaluator requires a feasible initial baseline.")

    def replay(order):
        available = set(initial_ids)
        trajectory = []

        for step in range(len(order) + 1):
            chosen = best(available)
            trajectory.append({
                "new_experiments": step,
                "selected_candidate_id": chosen["candidate_id"],
                "selected_image_size": sizes[chosen["candidate_id"]],
                "best_score": chosen["score"],
                "gain_over_initial": round(
                    chosen["score"] - initial_best["score"], 6
                ),
            })
            if step < len(order):
                available.add(order[step])

        return trajectory

    ascending = sorted(actual_order, key=lambda cid: sizes[cid])
    permutations = list(itertools.permutations(actual_order))
    random_replays = [replay(order) for order in permutations]

    random_summary = []
    for step in range(len(actual_order) + 1):
        values = [
            trajectory[step]["best_score"]
            for trajectory in random_replays
        ]
        random_summary.append({
            "new_experiments": step,
            "mean_best_score": round(mean(values), 6),
            "minimum_best_score": min(values),
            "maximum_best_score": max(values),
        })

    actual = replay(actual_order)
    sequential = replay(ascending)

    result = {
        "evaluation_version": "search_replay_v1",
        "evaluation_type": "retrospective_measured_evidence_replay",
        "source_run_ids": RUN_IDS,
        "requirement": requirement,
        "hardware_id": hardware_id,
        "initial_image_sizes": sorted(sizes[cid] for cid in initial_ids),
        "experiment_budget": len(actual_order),
        "llm_observed_order": [sizes[cid] for cid in actual_order],
        "ascending_order": [sizes[cid] for cid in ascending],
        "llm_trajectory": actual,
        "ascending_trajectory": sequential,
        "random_permutation_count": len(permutations),
        "random_without_replacement": random_summary,
        "final_selected_candidate_id": actual[-1]["selected_candidate_id"],
        "final_gain_over_initial": actual[-1]["gain_over_initial"],
        "limitations": [
            "Retrospective replay, not fresh randomized experiments.",
            "Random results enumerate all orders of these remaining candidates.",
            "Only one observed LLM sequence is available.",
            "No API cost, token usage or wall-clock comparison is made.",
            "Historical measurements were not all collected under the same OS.",
            "Scores and feasibility reuse the saved final selection policy.",
            "Findings apply only to this requirement and bounded resolution pool.",
        ],
    }

    output_dir = ROOT / "results/search_evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "search_replay_v1.json"

    # Refuse to silently replace a previous evaluation.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("Initial sizes:", result["initial_image_sizes"])
    print("Observed LLM order:", result["llm_observed_order"])
    print("Ascending order:", result["ascending_order"])
    print("Random orders evaluated:", len(permutations))
    print()
    print("Budget | LLM best | Ascending best | Random mean [min, max]")
    for step, random_item in enumerate(random_summary):
        print(
            f"{step:>6} | "
            f"{actual[step]['best_score']:.6f} | "
            f"{sequential[step]['best_score']:.6f} | "
            f"{random_item['mean_best_score']:.6f} "
            f"[{random_item['minimum_best_score']:.6f}, "
            f"{random_item['maximum_best_score']:.6f}]"
        )

    print("\nFinal selection:", result["final_selected_candidate_id"])
    print("Gain over initial:", result["final_gain_over_initial"])
    print("Saved:", output.relative_to(ROOT))


if __name__ == "__main__":
    main()
