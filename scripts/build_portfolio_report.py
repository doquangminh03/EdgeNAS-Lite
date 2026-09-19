"""Build an English portfolio report from verified selection evidence."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verification",
        default="results/reproducibility/fresh_clone_002/report.json",
    )
    parser.add_argument("--output", default="results/portfolio/demo_v1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    evidence_path = root / args.verification
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    if evidence.get("status") != "passed":
        raise ValueError("Fresh-clone verification must have passed.")

    selection = evidence["selection"]
    selected_id = selection["selected_candidate_id"]

    index = yaml.safe_load(
        (root / "knowledge/index.yaml").read_text(encoding="utf-8")
    )
    paths = {
        item["candidate_id"]: item["record_path"]
        for item in index["candidate_records"]
    }

    rows = []
    groups = (
        (selection["ranking"], True),
        (selection["rejected_candidates"], False),
    )
    for items, feasible in groups:
        for item in items:
            candidate_id = item["candidate_id"]
            record_path = paths[candidate_id]
            record = json.loads(
                (root / record_path).read_text(encoding="utf-8")
            )
            if record["candidate_id"] != candidate_id:
                raise ValueError("Candidate identity mismatch: " + candidate_id)

            image_size = record["accuracy"]["image_size"]
            if image_size != record["benchmark"]["image_size"]:
                raise ValueError("Image-size mismatch: " + candidate_id)

            metrics = item["metrics"]
            current = {
                "map50_95": record["accuracy"]["map50_95"],
                "median_latency_ms": record["benchmark"]["median_latency_ms"],
                "model_size_mb": record["model"]["model_size_mb"],
            }
            for name, value in current.items():
                if not math.isfinite(float(value)) or not math.isclose(
                    value, metrics[name], rel_tol=0, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"Record differs from verified evidence: "
                        f"{candidate_id}, {name}"
                    )

            rows.append({
                "candidate_id": candidate_id,
                "image_size": image_size,
                **metrics,
                "feasible": feasible,
                "selected": candidate_id == selected_id,
                "rank": item.get("rank", ""),
                "score": item.get("score", ""),
                "failed_constraints": ", ".join(
                    item.get("failed_constraints", [])
                ),
                "record_path": record_path,
            })

    rows.sort(key=lambda row: row["image_size"])
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate evidence.")
    if len(rows) != selection["evaluated_candidate_count"]:
        raise ValueError("Candidate count does not match verification.")
    if sum(r["feasible"] for r in rows) != selection["feasible_candidate_count"]:
        raise ValueError("Feasible count does not match verification.")

    chosen = [r for r in rows if r["selected"]]
    if len(chosen) != 1 or not chosen[0]["feasible"]:
        raise ValueError("Expected exactly one feasible selected candidate.")
    chosen = chosen[0]

    output = root / args.output
    if output.exists():
        raise ValueError("Output already exists; choose another --output.")
    output.mkdir(parents=True)

    with (output / "candidates.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Thresholds for the verified low_latency_balanced_demo milestone.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for feasible, color, marker, label in (
        (True, "#237A57", "o", "Feasible"),
        (False, "#C14B45", "X", "Rejected"),
    ):
        subset = [r for r in rows if r["feasible"] == feasible]
        ax.scatter(
            [r["median_latency_ms"] for r in subset],
            [r["map50_95"] * 100 for r in subset],
            color=color, marker=marker, s=85, label=label,
        )

    for row in rows:
        ax.annotate(
            str(row["image_size"]),
            (row["median_latency_ms"], row["map50_95"] * 100),
            xytext=(6, 8), textcoords="offset points", fontsize=10,
        )

    ax.scatter(
        chosen["median_latency_ms"], chosen["map50_95"] * 100,
        s=250, facecolors="none", edgecolors="#172B4D",
        linewidths=2, label="Selected by balanced score",
    )
    ax.axvline(12, color="#777777", linestyle="--", label="Latency limit: 12 ms")
    ax.axhline(20, color="#AAAAAA", linestyle=":", label="Minimum mAP: 20%")
    ax.set(
        xlabel="Median CPU latency (ms)",
        ylabel="KITTI validation mAP50–95 (%)",
        title="EdgeNAS-Lite: measured deployment configurations",
    )
    ax.grid(alpha=0.15)
    ax.margins(x=0.12, y=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"accuracy_latency.{suffix}", dpi=180)
    plt.close(fig)

    table = [
        "| Input size | mAP50–95 | Median latency | Model size | Result | Score |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        result = (
            "Selected" if row["selected"]
            else "Feasible" if row["feasible"]
            else "Rejected: " + row["failed_constraints"]
        )
        score = f'{row["score"]:.6f}' if row["score"] != "" else "—"
        table.append(
            f'| {row["image_size"]} | {row["map50_95"]:.6f} | '
            f'{row["median_latency_ms"]:.3f} ms | '
            f'{row["model_size_mb"]:.3f} MB | {result} | {score} |'
        )

    report = f"""# EdgeNAS-Lite — Pilot Results

## Objective and scope

EdgeNAS-Lite combines Gemini-based requirement interpretation and
bounded candidate proposals with deterministic validation, measured
evaluation, and constraint-aware selection.

This pilot searches input resolutions for one trained YOLO26n checkpoint
on KITTI using CPU inference. It is deployment configuration search;
it does not search neural network architectures or retrain each candidate.

## Demonstrated workflow

1. Interpret a natural-language request with Gemini.
2. Validate the structured requirement.
3. Retrieve compatible measured evidence.
4. Propose an unmeasured resolution from the bounded pool.
5. Validate and benchmark the proposed configuration.
6. Update the knowledge database and rank feasible candidates.
7. Stop when the experiment budget is consumed or the pool is exhausted.

Live LLM runs and model measurements were performed in earlier experiments.
The fresh-clone verification used here checks archived-evidence selection.

## Request and selection

The demo requires mAP50–95 >= 0.20, median latency <= 12 ms,
and model size <= 6 MB, with balanced optimization.

- Evaluated candidates: {selection["evaluated_candidate_count"]}
- Feasible candidates: {selection["feasible_candidate_count"]}
- Selected candidate: `{selected_id}`
- Selected mAP50–95: {chosen["map50_95"]:.6f}
- Selected median latency: {chosen["median_latency_ms"]:.3f} ms
- Selected model size: {chosen["model_size_mb"]:.3f} MB

## Measured results

{chr(10).join(table)}

![Measured accuracy and latency](accuracy_latency.png)

Point labels indicate input resolution. The plot shows recorded measurements;
it does not include confidence intervals.

## Why the balanced policy selects 416

The implemented balanced score averages normalized accuracy, latency,
and model-size headroom after rejecting constraint violations.
Under this policy, the latency headroom of resolution 416 outweighs
the accuracy gains of the other feasible resolutions.

This is the best candidate under the specified scoring rule and measured
pool. It is not a claim that 416 is universally optimal.

## Reproducibility evidence

- Fresh-clone status: `{evidence["status"]}`
- Verified commit: `{evidence["commit"]}`
- Scope: {evidence["scope"]}
- Requirements SHA-256: `{evidence["requirements_sha256"]}`
- Source report: `{args.verification}`

The verification passed dependency consistency, unit tests,
workflow CLI loading, and archived-evidence selection.
It did not call Gemini or run new YOLO measurements.

## Limitations

- One checkpoint, one dataset, one CPU target, and seven resolutions.
- Hardware identity is operator-supplied.
- Historical measurements were collected across different macOS versions;
  the results do not establish a controlled resolution-only effect.
- Fresh-clone verification used the same machine and an already verified
  environment. Cross-machine execution remains unverified.
- Earlier replay evaluation showed no score gain over its initial pool.
  Current evidence does not establish an LLM search advantage over baselines.
- Full architecture search, broader hardware support, and stronger
  prospective search evaluation remain future work.

## Files

- `candidates.csv`: measured metrics and selection outcomes.
- `accuracy_latency.png`: report figure.
- `accuracy_latency.svg`: vector figure.
- `evidence.json`: selection and source references used by this report.
"""

    (output / "REPORT.md").write_text(report, encoding="utf-8")
    (output / "evidence.json").write_text(
        json.dumps({
            "verification_report": args.verification,
            "verified_commit": evidence["commit"],
            "selection": selection,
            "candidates": rows,
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Report:", output / "REPORT.md")
    print("Figure:", output / "accuracy_latency.png")
    print("CSV:", output / "candidates.csv")


if __name__ == "__main__":
    main()
