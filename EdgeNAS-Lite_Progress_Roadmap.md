# EdgeNAS-Lite — Progress and Roadmap

> Updated 18 September 2026 from supplied reports and terminal output. Completion refers to the current prototype scope. This update does not run tests or establish commit/push status.

## Current milestone

The core LLM-guided deployment-configuration pipeline is implemented and has live experiment evidence. Gemini interprets requirements and proposes bounded resolutions; deterministic code performs validation, measurement, compatibility checks, indexing, and selection.

The seven-resolution pool is fully measured. Four candidates satisfy the low-latency demo, and 416 remains selected with balanced score 0.143132. Further experiments in this exhausted pool are not needed merely to demonstrate that the loop works.

## Completed work

### Foundation and measurement

- [x] Project environment, YOLO26 smoke inference, and KITTI setup.
- [x] One-epoch smoke training and ten-epoch pilot training on 25% of training data.
- [x] Full validation on 1,496 images and 6,989 selected-class instances.
- [x] Standardized CPU measurement with saved raw samples.
- [x] Five sessions per new candidate, pooling sessions 3–5 into 900 samples.
- [x] Measured resolutions 416, 448, 480, 512, 576, 608, and 640.
- [x] Explicit hardware-ID propagation through Runner and benchmark.
- [x] Template-candidate validation for single-candidate plans.

### Deterministic modules

- [x] Structured Requirement Parser and Search Space Parser.
- [x] Candidate Runner, Constraint Checker, and Candidate Selector.
- [x] Deterministic ranking and tie-breaking for supported optimization goals.
- [x] Search Controller with two historical reuse-based demo runs.
- [x] File-based Knowledge DB with index validation and query filters.
- [x] Compatibility-gated rule-based proposal and CLI.
- [x] Separate reusable measurements and requirement-specific reports.

### LLM and orchestration

- [x] Real Gemini provider and validated natural-language interpretation.
- [x] `ready`, `needs_clarification`, and `unsupported` outcomes.
- [x] Natural-language selection CLI with persisted interpretation/proposal.
- [x] Bounded Candidate Proposer and manual live 448 cycle.
- [x] Experiment Controller: propose, execute, validate, register, and rerank.
- [x] Live automated 480 experiment: `low_latency_cycle_001`.
- [x] Sequential budget controller with saved slots and limited resume.
- [x] Live two-slot batch for 576 and 608: `low_latency_batch_001`.
- [x] Natural-language workflow connecting interpretation and optional experiments.
- [x] English budget-zero demo selecting 416: `english_demo_001`.
- [x] English budget-one demo stopping at pool exhaustion: `english_demo_002`.

Limited resume preserves completed work and may continue unstarted slots. Failed or ambiguous attempts require review; automatic retry and full crash recovery are not complete. The positive-budget English demo confirms exhaustion handling, not a newly executed candidate through that wrapper.

### Evaluation and setup checks

- [x] Six live semantic cases passed.
- [x] Three clarification cases passed automatic checks.
- [x] Replay comparison with ascending order and six permutations.
- [x] Three live hidden-evidence trials, budget three per trial.
- [x] Local prerequisite checker for pinned versions, candidate references, checkpoint existence, and KITTI configuration discovery.
- [x] Local evaluate-mode check: zero failed checks.
- [x] `pip check`: no broken requirements found.
- [ ] Complete manual review of all clarification questions.
- [ ] Confirm the full-suite count after all later modules were added.

The last explicitly counted full-suite result was 193 tests OK after hardware-metadata changes. The earlier 178-test milestone is historical. Local prerequisite checks do not prove fresh-install reproducibility or dataset integrity.

## Measured outcome

Requirement: mAP50–95 >= 0.20, median CPU latency <= 12 ms, model <= 6 MB, balanced ranking.

| Resolution | mAP50–95 | Median latency (ms) | Result |
|---:|---:|---:|---|
| 416 | 0.207384 | 8.754 | Selected |
| 448 | 0.221918 | 10.628 | Feasible, rank 2 |
| 480 | 0.236280 | 10.934 | Feasible, rank 3 |
| 512 | 0.242931 | 11.279 | Feasible, rank 4 |
| 576 | 0.261257 | 13.082 | Rejected: latency |
| 608 | 0.272838 | 13.616 | Rejected: latency |
| 640 | 0.273000 | 16.429 | Rejected: latency |

All use the same 5.102 MB checkpoint. Hardware identity is operator-supplied. Historical and later records span different macOS versions; matched-environment measurements are needed before attributing latency differences solely to input size.

Replay gain was zero because the initial evidence already contained the eventual best candidate. Hidden-search orders were `[480, 416, 448]`, `[416, 448, 480]`, and `[512, 480, 576]`; two of three trials found 416 within budget. Ascending order finds it first in this pool. These exploratory results do not establish an LLM search advantage.

## Next: finish reproducibility and release evidence

1. **Clean environment:** install the pinned dependencies in a separate environment, leaving the working environment intact. Run dependency checks and the software suite; record Python/platform/package versions and failures.
2. **Fresh checkout:** verify the committed code, configs, and indexed records suffice for the recorded-evidence demo. Keep this distinct from new measurement, which requires KITTI and checkpoint assets.
3. **Live demo:** configure a private Gemini API key and use an unused run ID. Check the saved structured requirement and selected metrics. A recorded-evidence demo on another machine still refers to the original benchmark target.
4. **Release review:** record the latest test total, review staged artifacts, update setup instructions, and verify commit/push completion.

Completion criteria: a clean checkout and documented environment can run tests and the English recorded-evidence workflow; required assets and hardware provenance are explicit. These criteria have not yet been confirmed.

## Follow-up engineering

- [ ] Check dataset image/label availability rather than only YAML existence.
- [ ] Record checkpoint hashes, dataset/image identities, software versions, and CPU thread settings.
- [ ] Preserve hardware/template provenance consistently for older and new records.
- [ ] Align reuse checks across the deterministic and LLM workflows.
- [ ] Define invalidation/remeasurement rules and collect matched-environment benchmarks.
- [ ] Add focused regression coverage for workflow failure, clarification, exhaustion, and successful new execution as needed.
- [ ] Improve recovery from partial writes, interrupted experiments, and stale locks.
- [ ] Separate experiment-slot budgets from API/token/cost limits.
- [ ] Add interactive clarification continuation if needed.

## Follow-up evaluation

- [ ] Expand unit, ambiguity, conflict, and boundary-value semantic cases.
- [ ] Record prompt/model versions and retain expected interpretation outputs.
- [ ] Predefine search tasks and initial evidence before comparing strategies.
- [ ] Compare LLM, ascending/grid, and random approaches under equal budgets over more trials.
- [ ] Report feasibility discovery, best score, regret, calls, and runtime/cost where available.
- [ ] Evaluate alternative objective weights or Pareto selection separately from the current fixed policy.

Do not retroactively change historical proposals or benchmark records to improve results. Keep new evaluations as separate artifacts.

## Optional expansion after the reproducible prototype

- [ ] Dashboard for requirements, experiment status, evidence, and rankings.
- [ ] Model-scale, export-format, or quantization search with new measurements.
- [ ] Additional physical hardware and dataset-specific compatibility policies.
- [ ] Controlled depth/width/block search for architecture-level NAS.

These are extensions, not prerequisites for documenting the current configuration-search prototype accurately.

## Reporting boundaries

- Implemented orchestration does not imply fully autonomous research or general fault recovery.
- Seven resolutions represent one trained checkpoint, not seven independently trained models.
- A valid schema does not guarantee correct interpretation of every natural-language request.
- Local environment checks do not establish cross-platform installation or validate dataset contents.
- Best under the fixed balanced policy does not mean globally optimal or highest accuracy.
- No confirmed evidence currently supports LLM superiority over the evaluated baselines.
- Commit/push status must be verified from Git output, not inferred from local execution.

See [README](README.md) for commands, artifact paths, metric definitions, and the current workflow.
