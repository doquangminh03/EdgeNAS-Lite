# EdgeNAS-Lite — Progress and Roadmap

Updated **25 October 2026** (Asia/Taipei). This roadmap summarizes verified
repository and workflow state after portfolio remediation; implementation,
archived evidence, new measurements and unexecuted evaluations are distinct.

## Current release state

- [x] [PR #1](https://github.com/doquangminh03/EdgeNAS-Lite/pull/1) merged into
  `main`: `8fe37931bd5194e499af371d07f88aee5ddfeab1`.
- [x] Local `main` fast-forwarded to that merge while preserving all 30
  pre-existing untracked files; `fix/portfolio-readiness` retained locally and
  on GitHub. The subsequent balanced replay wrote a new local output.
- [x] [GitHub Actions core and full](https://github.com/doquangminh03/EdgeNAS-Lite/actions/runs/40107248046)
  passed on PR head `00923dedf47c6deb36c9c585e31b5eb091d41a6e`.
- [x] Latest implementation validation: **235 local software tests passed**,
  real local-checkpoint integration passed, new 1-thread campaign independently
  verified. Historical counts 193/211/219/229 are earlier milestones.
- [x] Local USB Git maintenance: 13 AppleDouble `.idx` sidecars were backed up,
  checksum-verified and quarantined outside `objects/pack`; 13 real pack/index
  pairs passed verification and all 39 real `.pack/.idx/.rev` files were
  unchanged. Local `git status` no longer reports the non-monotonic-index error.
  This is local maintenance, not an application feature or repository artifact.

## Supported workflow

1. **Offline replay:** structured requirement → compatible archived records →
   hard-constraint evaluation → deterministic selection → a new output file.
   Use `requirements-demo.txt`; no dataset, weights or Gemini key are required.
2. **Controlled remeasurement:** full dependencies + matching local checkpoint
   and dataset → campaign preflight → new versioned campaign/run ID → saved
   raw samples, hashes and requested/observed thread evidence.
3. **Evidence promotion:** obtain corresponding accuracy and complete enough
   checkpoint/dataset/environment provenance before registering replacements
   or changing the selection index. This step is **not completed**.
4. **Optional live interpretation/search evaluation:** obtain an explicit API
   spending budget, run the prepared evaluator and retain responses, versions
   and costs when actually available. This step is **NOT_RUN** for the expanded
   evaluation; no paid API work is implied by available credentials.

For commands, see [README](README.md), [setup](SETUP_AND_USAGE.md) and
[reproduction levels](docs/REPRODUCIBILITY.md). Use unique outputs, keep local
owner files separate from reviewed artifacts, and preserve historical records.

## Completed engineering

### Deterministic selection and evidence handling

- [x] Requirement/search-space parsers, Candidate Runner, Constraint Checker,
  Candidate Selector and file-based Knowledge DB.
- [x] Finite-value validation at the input boundary, including YAML/JSON NaN,
  infinity, bool and numeric-overflow regressions.
- [x] Shared pilot compatibility gate for Search Controller reuse and proposal
  paths; explicit hardware identity for structured execution, in addition to
  checkpoint/resolution checks against the search space.
- [x] Deterministic ranking/tie-breaks and legacy balanced policy preserved.
  Scoring assumptions, fixed-weight sensitivity and Pareto analysis documented.
- [x] Optional strict comparison mode rejects incomplete or mismatched supplied
  provenance; archived demo policy remains available with its stated limits.
- [x] Separate reusable measurements, requirement-specific reports and archived
  selection outputs; invalid/no-feasible/hardware-mismatch cases covered.

### LLM orchestration

- [x] Gemini interpretation/provider and bounded candidate proposal integrated
  with Python-side validation, execution, registration and reranking.
- [x] `ready`, `needs_clarification` and `unsupported` interpretation outcomes.
- [x] Workflow, Experiment Controller and sequential experiment-slot budget,
  including pool exhaustion and limited resume behavior.
- [x] Historical live 448/480/576/608 experiments and English workflow examples
  retained as evidence, not repeated during remediation.

Limited resume preserves completed work and may continue unstarted slots.
Failed/ambiguous attempts, complete crash recovery and an interactive
clarification continuation are not fully solved.

### Reproducibility, benchmark and evaluation tooling

- [x] Fresh local lightweight replay environment and automated offline scenario
  checks. CI core verified the archived demo from a checkout without ML assets.
- [x] Full CI dependency installation, `pip check` and software suite passed.
  This does not establish cross-platform measurement reproducibility.
- [x] Local dataset manifest: 5,985 train / 1,496 validation images, labels checked,
  no overlap by image ID/hash. Exact historical split generation is not recovered.
- [x] Checkpoint/image manifests, software versions, execution-source hashes,
  raw sessions and independent statistics verification for the new campaign.
- [x] Ultralytics thread reset fixed: one untimed backend setup prediction,
  enforce requested threads, verify configured warmup and each timed prediction
  boundary, fail on missing/mismatched observations.
- [x] `cpu_campaign_v2` / `cpu_threads_v2` campaign completed with all 35 runs
  and 10,500 raw samples; requested/observed = 1 at the recorded timing boundaries.
- [x] 36 additional pre-labelled language cases with offline/live evaluator,
  schema/status/field checks and clarification rubric. Mocked regression is
  separate from live model-quality evidence.
- [x] Predefined equal-budget archived search replay for four requirements and
  five random seeds, plus optional live reveal tooling and scoring sensitivity.
- [x] Three archived clarification responses reviewed by the coding agent with
  rubric/reasons; independent human review remains pending.

## Evidence sets and measured outcomes

### Archived selection demo — index unchanged

For mAP50–95 >= 0.20, median latency <= 12 ms, model file <= 6 MB and balanced
ranking: **7 evaluated, 4 feasible, 416 selected**, balanced score 0.143132.

| Resolution | Archived mAP50–95 | Archived median ms | Archived result |
|---:|---:|---:|---|
| 416 | 0.207384 | 8.754 | Selected |
| 448 | 0.221918 | 10.628 | Feasible, rank 2 |
| 480 | 0.236280 | 10.934 | Feasible, rank 3 |
| 512 | 0.242931 | 11.279 | Feasible, rank 4 |
| 576 | 0.261257 | 13.082 | Rejected: latency |
| 608 | 0.272838 | 13.616 | Rejected: latency |
| 640 | 0.273000 | 16.429 | Rejected: latency |

These records describe one 5.102 MB checkpoint, not seven trained models.
Historical benchmark software environments differ. The balanced demo was
successfully replayed on merged `main`; it still uses these archived metrics.
[Archived portfolio report](results/portfolio/demo_v1/REPORT.md).

### Verified 1-thread campaign — latency only

`threads1_fixed_20261024` uses 100 hashed images, batch 1, five sessions, one
untimed setup prediction before 10 configured warmups, and three repetitions.
Sessions 1–2 are excluded by the declared stabilization rule; sessions 3–5
provide 900 retained samples per resolution (6,300 total).

| Resolution | New pooled median ms | New pooled P95 ms |
|---:|---:|---:|
| 416 | 9.5850 | 10.2863 |
| 448 | 12.3300 | 12.7111 |
| 480 | 13.0655 | 14.9185 |
| 512 | 13.6230 | 15.8328 |
| 576 | 17.5055 | 19.3564 |
| 608 | 18.3610 | 19.9458 |
| 640 | 22.0320 | 22.7890 |

[Raw verification](results/remediation/thread_fix_20261024/campaign_verification.json)
and [thread-control report](docs/THREAD_CONTROL_FIX.md) record sample identities,
thread boundaries, hashes and limitations. This is not a new accuracy/latency
selection result. No new accuracy validation or index promotion has occurred.
Repeated timings are not independent samples and the host was also used for
documentation. Thread observations are at prediction boundaries, not continuous
OS-thread monitoring.

The older `continuation_measured_v2_20261024` requested 1 thread but recorded 7;
it is preserved and **does not satisfy the 1-thread protocol**. Do not combine
its results, or the new campaign's latency, with archived accuracy to imply a
new validated deployment choice.

## Remaining work, in order

1. **Accuracy and provenance:** collect matching full-validation accuracy,
   checkpoint/data identities and comparable metadata before promoting the new
   campaign. Do not assume 416 or four feasible candidates remain the outcome
   under replacement measurements.
2. **Historical data/training lineage:** recover the conversion/split/subset
   procedure or define a new versioned process. Document retrain commands and
   manifests without claiming bit-for-bit reproduction of the pilot.
3. **Live language evaluation:** obtain budget approval and run the 36 prepared
   cases; preserve model/prompt IDs, responses and manual-review decisions.
   Keep current live status NOT_RUN until actually executed.
4. **Search evaluation:** run the prepared LLM/ascending/random comparison under
   the same initial evidence and budget, keeping hidden measurements out of the
   prompt. Archived reveal is not online experiment runtime; token/cost fields
   remain unknown unless measured. No LLM superiority has been demonstrated.
5. **Independent human review and licensing:** review the archived clarification
   questions and choose a repository-owned-code license. Third-party software,
   datasets and weights retain their own terms; owner review is still needed.
6. **Operational hardening, when needed:** improve interrupted-run recovery,
   retries/locks and API monetary-budget controls; add interactive clarification
   continuation only if required. These are unfinished extensions, not current
   guarantees.

An archived replay and core/full CI are now verified. A new accuracy/provenance
release, expanded live model evaluation, exact retraining reproduction and
additional physical targets are not completed claims. Historical handoff/audit
files record their state at creation; this roadmap is the current summary.

Optional later scope remains a dashboard, additional model/export/quantization
configurations, more hardware and architecture-level NAS. The implemented
project remains deployment-configuration selection, not architecture search.
