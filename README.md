# EdgeNAS-Lite

LLM-guided, hardware-aware YOLO deployment-configuration search with measured evidence and deterministic selection.

> Updated 18 September 2026. The prototype includes Gemini requirement interpretation, bounded candidate proposals, automated experiment execution, budgeted batches, and a natural-language workflow. Seven resolutions have been measured. The balanced demo still selects 416; the experiments do not establish an LLM advantage over baseline search.

## Scope

EdgeNAS-Lite translates deployment requirements into constraints, retrieves compatible measurements, optionally evaluates new configurations, and selects a feasible candidate. Gemini interprets requests and proposes experiments; Python validates the outputs, measures candidates, and determines the final ranking.

The current search changes input resolution for one trained YOLO26n checkpoint. It is deployment-configuration search, not architecture-level NAS. Model-scale search, retraining search, export/quantization search, and a dashboard remain future work.

## Implemented components

| Component | Responsibility |
|---|---|
| Requirement Parser | Validate structured YAML/JSON requirements |
| Search Space Parser | Validate the fixed configuration and expand resolution grids |
| Candidate Runner | Full validation, five CPU benchmark sessions, and candidate records |
| Constraint Checker and Candidate Selector | Enforce hard constraints and rank feasible candidates |
| Search Controller | Orchestrate the deterministic structured-requirement pipeline |
| Knowledge Database | Validate references and core metrics; query measured records |
| Compatibility-gated Proposal | Exclude incompatible evidence before selection |
| Gemini Interpreter and CLI | Convert natural language to validated requirements or clarification/unsupported results |
| Candidate Proposer | Suggest one eligible, unmeasured resolution from the bounded pool |
| Experiment Controller | Propose, execute, check compatibility, update the index, and rerank |
| Budget Controller | Run sequential experiment slots and stop at the budget or pool exhaustion |
| Natural-language Workflow | Connect interpretation, initial selection, optional experiments, and final selection |
| Environment Checker | Check pinned package versions, candidate references, and optional local assets |

Principal LLM entry points are under `src/llm_agent/`: `cli.py`, `candidate_proposer.py`, `experiment_controller.py`, `budget_controller.py`, and `workflow.py`.

The older Search Controller and the newer LLM Experiment Controller are separate entry points. The two historical Search Controller runs reused records; the newer Experiment Controller has live execution evidence.

## Measured results

The pilot used 10 epochs, 25% of KITTI training data, image size 640, batch size 4, MPS, and seed 42. All seven deployment configurations use `runs/detect/kitti_yolo26n_pilot/weights/best.pt`, without retraining. The checkpoint has 2,376,396 parameters and occupies 5.102 MB.

Full validation covers 1,496 images and 6,989 selected-class instances: car, pedestrian, and cyclist (KITTI IDs 0, 3, and 5).

Demo constraints: mAP50–95 >= 0.20, median latency <= 12 ms, model size <= 6 MB; optimization goal `balanced`.

| Resolution | mAP50–95 | Median CPU latency (ms) | Model size (MB) | Balanced score | Result |
|---:|---:|---:|---:|---:|---|
| 416 | 0.207384 | 8.754 | 5.102 | 0.143132 | Selected |
| 448 | 0.221918 | 10.628 | 5.102 | 0.097132 | Rank 2 |
| 480 | 0.236280 | 10.934 | 5.102 | 0.094617 | Rank 3 |
| 512 | 0.242931 | 11.279 | 5.102 | 0.087805 | Rank 4 |
| 576 | 0.261257 | 13.082 | 5.102 | — | Rejected: latency |
| 608 | 0.272838 | 13.616 | 5.102 | — | Rejected: latency |
| 640 | 0.273000 | 16.429 | 5.102 | — | Rejected: latency |

Final selection evaluated seven candidates and found four feasible. The selected ID is `yolo26n_kitti_pilot_imgsz416_cpu`; the 640 record retains the ID `yolo26n_kitti_pilot`.

### Benchmark and ranking policy

CPU benchmarks use batch size 1, 100 validation images selected with seed 42, 10 warm-ups per session, and three repetitions per image. Five sessions are saved; sessions 1–2 are excluded as stabilization runs, while sessions 3–5 supply 900 pooled samples. Timing covers preprocessing, inference, and postprocessing; disk I/O is excluded. P95 is recorded but is not the demo's hard latency constraint.

Feasible candidates receive equal-weight normalized headroom scores:

- Accuracy: `(map50_95 - minimum_map50_95) / (1 - minimum_map50_95)`.
- Latency: `(maximum_median_latency_ms - median_latency_ms) / maximum_median_latency_ms`.
- Model size: `(maximum_model_size_mb - model_size_mb) / maximum_model_size_mb`.

Components are clamped to [0, 1] and rounded to six decimals; their mean is rounded to six decimals. For minimum accuracy 1.0, accuracy headroom is set to 1.0. Ties favor higher accuracy, lower latency, smaller model size, then ascending candidate ID. This fixed policy explains why 416 wins despite lower accuracy. Model size is constant across this pool.

**Comparison limit:** historical 416/512/640 benchmarks record macOS 26.3.1; 448 records 26.6.2. The collection spans software environments. Rankings are reproducible from stored metrics, but latency differences cannot be attributed solely to resolution.

## Completed experiment milestones

| Milestone | Confirmed outcome |
|---|---|
| Manual 448 cycle | Gemini proposal, validation, benchmark, compatibility check, indexing, and reranking |
| `low_latency_cycle_001` | Automated experiment completed for 480 |
| `low_latency_batch_001` | Budget-two batch completed for 576 and 608 |
| `english_demo_001` | Natural-language workflow with budget 0 completed and selected 416 |
| `english_demo_002` | Budget 1 reached `exhausted` / `no_available_candidate`; workflow completed and selected 416 |
| Local environment check | `--mode evaluate` reported zero failed checks; `pip check` reported no broken requirements |

All seven pool resolutions are now measured. A positive budget permits experiments; it does not force a new measurement when no eligible candidate remains. The exhausted workflow did not demonstrate a new model execution through that wrapper; actual execution was demonstrated by the earlier experiment and batch runs.

### Search-strategy evaluation

`scripts/evaluate_search_replay.py` replayed the observed order `[480, 576, 608]` against ascending and all six permutations of those three choices. Initial evidence already included 416, 448, 512, and 640. Every strategy retained score 0.143132 at budgets 0–3: gain over the initial best was zero.

`scripts/evaluate_hidden_search.py` conducted three live Gemini trials with budget three, starting with only the 640 measurement visible and revealing archived measurements after each choice:

| Trial | Proposed order | Found 416 within budget |
|---:|---|---|
| 1 | 480, 416, 448 | Yes, step 2 |
| 2 | 416, 448, 480 | Yes, step 1 |
| 3 | 512, 480, 576 | No |

All three trials found a feasible candidate on the first choice. Ascending order would choose 416 first; uniform random selection of three distinct candidates from the six remaining sizes includes 416 with probability 50%. These are exploratory results from a small, retrospectively selected pool, not evidence of LLM superiority. Hidden-search evaluation uses archived metrics rather than new YOLO measurements.

## Installation and local checks

The verified local environment uses Python 3.9.6, PyTorch 2.8.0, Ultralytics 8.4.123, and macOS ARM64. `requirements.txt` pins the recorded environment; installation in a fresh environment and on other platforms remains unverified.

```bash
git clone https://github.com/doquangminh03/EdgeNAS-Lite.git
cd EdgeNAS-Lite
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py --mode demo
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Cross-platform installation has not yet been validated.

For local validation/benchmark prerequisites:

```bash
python scripts/check_environment.py --mode evaluate
```

The checker looks for `kitti.yaml` in the project root, then in the installed Ultralytics package. It checks file existence, indexed candidate IDs, and pinned package versions. It does **not** validate dataset images/labels, checkpoint contents, API credentials, or network access. It makes no API calls and runs no models.

Recorded-evidence selection does not load KITTI images or checkpoint weights. New measurements need both, plus a valid template record. Dataset files, local environments, and full model run directories are not included with a normal clone.

## Run the natural-language workflow

Set `GEMINI_API_KEY` in the process environment using your own credentials. Automatic `.env` loading is not established. The implemented default model is `gemini-3.5-flash-lite`; `--model` overrides it. Live access depends on your provider account and model availability.

Run from the project root and choose an unused run ID:

```bash
python -m src.llm_agent.workflow \
  --text "I need object detection on KITTI using CPU. Minimum mAP50-95 is 0.20, maximum median latency is 12 ms, and maximum model file size is 6 MB. Use balanced optimization." \
  --run-id english_demo_003 \
  --hardware-id local_mac_cpu_01 \
  --budget 0
```

Budget 0 still uses Gemini to interpret the request; it selects from recorded evidence without new YOLO evaluation. For budget 1–7, add `--template-candidate-id yolo26n_kitti_pilot` and change the budget. With the current complete pool, the experiment branch should stop as exhausted.

`local_mac_cpu_01` identifies the machine that produced the archived benchmarks. On another machine, using this ID means exploring evidence for that original target, not measuring the new machine. New target hardware needs its own identity and measurements.

Reports are stored under `results/workflows/<run_id>/`: `interpretation.json`, `workflow.json`, and, for a ready interpretation, `requirement.yaml`. Inspect the saved requirement against the original text. Clarification and unsupported requests stop before experiments. Existing run directories are protected; the workflow itself has no resume option.

Exit codes: 0 for successful selection, 1 for a completed result without selection or requiring clarification, and 2 for errors. Workflow `completed` alone does not imply a feasible selection; inspect `selection_status` and `final_proposal`.

### Structured selection without a live API call

```bash
python -m src.proposal.cli \
  configs/requests/low_latency_balanced_demo.yaml \
  --hardware-id local_mac_cpu_01 \
  --model-family YOLO26 \
  --output results/proposals/low_latency_current.json
```

The current index yields seven compatible records, four feasible candidates, and selection 416. This structured CLI replaces an existing output at the chosen path; use a new filename to preserve prior snapshots.

### Execution and recovery boundaries

Runner and benchmark accept an explicit `--hardware-id` and propagate it into measurements. The ID is operator-supplied, not automatic device detection. `--template-candidate-id` supplies matching model/training metadata; new candidate accuracy and latency are measured separately.

Experiment Controller validates proposals and measured records before index registration and reranking. Budget Controller reserves slots before execution; failed attempts consume slots. Its limited `--resume` reconciles completed child reports and can continue unstarted slots with matching settings. Failed or uncertain runs require review and are not automatically rerun. Locks are not a complete transaction system; forced termination can leave stale locks or partial artifacts.

Experiment budgets limit slots, not tokens, monetary cost, or all possible HTTP attempts. Use each module's `--help` for its own arguments; the workflow, experiment controller, and batch controller have different options.

## Evidence and tests

| Location | Contents |
|---|---|
| `knowledge/index.yaml` | References to seven measured candidate records |
| `results/candidates/`, `results/benchmarks/` | Metrics, benchmark sessions, and raw samples |
| `results/evaluations/`, `results/selections/`, `results/search_runs/` | Historical deterministic pipeline results |
| `results/proposals/` | Proposal rationale and selection snapshots |
| `results/llm_experiments/low_latency_cycle_001.json` | Completed 480 experiment |
| `results/llm_experiments/low_latency_batch_001_002.json` | Final batch experiment and seven-candidate selection |
| `results/llm_batches/low_latency_batch_001.json` | Completed two-slot batch |
| `results/search_evaluations/search_replay_v1.json` | Retrospective replay comparison |
| `results/search_evaluations/hidden_search_v1.json` | Hidden-evidence search evaluation |
| `results/workflows/english_demo_002/workflow.json` | Completed exhausted-pool workflow |
| `results/llm_evaluations/` | Live interpretation evaluations |
| `artifacts/kitti_pilot/` | Retained pilot plots and metrics |

The last explicitly counted full-suite result supplied was **193 tests OK**, after hardware-metadata changes. Earlier milestones were 166 and 178. Later modules and checks were added, but a new full-suite total has not been confirmed. Do not use 193 as a verified count for the entire latest tree.

Live interpretation evaluation: six semantic cases passed; three additional clarification cases passed automatic checks. Full manual review of all clarification questions remains unconfirmed. These API cases are separate from software tests.

```bash
python -m unittest discover -s tests -v
```

This documentation update does not rerun tests, training, benchmarks, or API evaluations, and does not confirm that all local artifacts have been committed or pushed.

## Remaining work

1. Validate installation and tests in a clean environment; then verify a fresh-clone demo and document asset/key setup.
2. Record a new full-suite result for the current source tree and verify repository artifact completeness.
3. Strengthen checkpoint/dataset identity, software and thread metadata, and reuse-policy consistency; benchmark under matched conditions for controlled comparisons.
4. Expand semantic and search-strategy evaluation using predefined tasks, fair initial evidence, equal budgets, and more trials.
5. Improve recovery and add a presentation dashboard if needed for the portfolio.
6. Extend model scale, export format, quantization, or hardware only with new measured evidence. Architecture search is longer-term work.

See [the detailed roadmap](EdgeNAS-Lite_Progress_Roadmap.md).

## Limitations and attribution

The prototype covers one pilot checkpoint, KITTI, CPU deployment, and seven input resolutions. The `kitti_cpu_pilot_v1` policy checks selected protocol metadata and the supplied hardware ID; it does not establish full software, dataset, checkpoint-content, or thermal equivalence. Some historical hardware metadata was completed after measurement. Recorded FPS derived from median latency is not full application throughput.

Balanced selection is a fixed policy, not global optimality. Neither the unchanged best score nor the three hidden-search trials establishes that Gemini searches better than simple baselines. There is no interactive clarification continuation, general failure recovery, verified cross-platform setup, or dashboard yet.

Ultralytics YOLO, pretrained weights, KITTI, and provider services retain their respective credits and licenses. Keep credentials, datasets, environments, local backups, and full run directories outside commits; review generated artifacts for machine-specific paths.
