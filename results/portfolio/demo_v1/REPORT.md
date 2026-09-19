# EdgeNAS-Lite — Pilot Results

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

- Evaluated candidates: 7
- Feasible candidates: 4
- Selected candidate: `yolo26n_kitti_pilot_imgsz416_cpu`
- Selected mAP50–95: 0.207384
- Selected median latency: 8.754 ms
- Selected model size: 5.102 MB

## Measured results

| Input size | mAP50–95 | Median latency | Model size | Result | Score |
| --- | ---: | ---: | ---: | --- | ---: |
| 416 | 0.207384 | 8.754 ms | 5.102 MB | Selected | 0.143132 |
| 448 | 0.221918 | 10.628 ms | 5.102 MB | Feasible | 0.097132 |
| 480 | 0.236280 | 10.934 ms | 5.102 MB | Feasible | 0.094617 |
| 512 | 0.242931 | 11.279 ms | 5.102 MB | Feasible | 0.087805 |
| 576 | 0.261257 | 13.082 ms | 5.102 MB | Rejected: maximum_median_latency_ms | — |
| 608 | 0.272838 | 13.616 ms | 5.102 MB | Rejected: maximum_median_latency_ms | — |
| 640 | 0.273000 | 16.429 ms | 5.102 MB | Rejected: maximum_median_latency_ms | — |

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

- Fresh-clone status: `passed`
- Verified commit: `b6adb98d505f859533e79d33dfd0768beefbb863`
- Scope: remote branch; fresh checkout; previously verified environment; same machine
- Requirements SHA-256: `e9d815766bc0a69323eb87b394edac8ac1f66217c172bd5b405f3064d2e3b264`
- Source report: `results/reproducibility/fresh_clone_002/report.json`

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
