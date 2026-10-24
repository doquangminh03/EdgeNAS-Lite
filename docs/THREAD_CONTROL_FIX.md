# CPU thread control follow-up — 2026-10-24

This change follows the thread mismatch found in
[REMEDIATION_AUDIT.md](REMEDIATION_AUDIT.md). The old
`continuation_measured_v2_20261024` campaign remains unchanged and does **not**
establish the requested 1-thread protocol: its 35 raw records report 7 threads.
No accuracy measurements or index entries are replaced.

## Implementation

`initialize_backend` runs one untimed `model.predict` call through the public
Ultralytics API. That initializes the lazy predictor/backend and its internal
warmup before the requested torch intra-op thread count is enforced. The setup
prediction is separate from the configured warmup count and raw sample count.

The benchmark then checks `torch.get_num_threads()` before and after every
configured warmup and every timed prediction, and when building the result.
It fails immediately on a requested/observed mismatch rather than resetting
threads again and hiding drift. The observations surround the predict call;
thread checks are outside the timer. These are boundary observations, not
continuous sampling of all OS threads or torch inter-op thread counts.

New raw records retain schema 1.0 with additive metadata:

- `protocol.version = cpu_threads_v2`, one backend setup prediction,
  `requested_cpu_threads`, and declared thread-check scope.
- `environment.requested_cpu_threads`, actual `torch_num_threads`, and stage
  `thread_observations`: after backend setup, before/after configured warmup,
  before/after timing.
- Each raw sample has `torch_num_threads_before` and `torch_num_threads_after`.

A positive integer can be supplied by `protocol.cpu_threads` or `--threads`.
Conflicting values fail before execution. Without either, the benchmark leaves
Ultralytics' default unchanged and records requested null; such a run cannot
pass the campaign's explicit thread contract.

Campaign protocol is now `cpu_campaign_v2`. It independently verifies the raw
benchmark version, requested and observed environment fields, warmup/timing stage
observations, and both thread fields in every sample. Missing evidence, bool
values masquerading as integers, or mismatched values fail the campaign before
accepting that session. Failed runs retain status failed and cannot acquire a
completed summary. Source SHA-256 for benchmark and campaign scripts is captured
at planning and checked before every subprocess, alongside checkpoint/image
hashes. This addresses the missing execution-time source fingerprint for these
two entry points; it is not a hash of the whole installed ML environment.

## Validation

- Full software suite: **235 pass, 0 fail/error/skip**, exit 0:
  [tests_verified.log](../results/remediation/thread_fix_20261024/tests_verified.log).
  The preceding audit count 229 remains a historical result; six tests were
  added in this follow-up (five benchmark tests and one campaign evidence test).
- Regression simulates backend setup resetting threads to 7, verifies enforcement
  before warmup/timing and verifies later reset failures at the actual expected
  stage. Setter failure, pre-timing drift, invalid settings and CLI/config
  conflicts are covered. A campaign integration fixture returning a successful
  subprocess result with 7 observed threads must still be rejected and saved
  as failed; legacy/missing per-sample evidence is rejected too.
- Local checkpoint integration: **PASS**, one setup prediction, 2 configured
  warmups, 3 images × 2 repetitions = 6 timed predictions. Actual observations:
  **backend setup 7 → before warmup 1 → after warmup 1 → before/after timing 1**;
  all 6 samples record 1 at both boundaries. See
  [integration_verification.json](../results/remediation/thread_fix_20261024/integration_verification.json),
  [raw result](../results/remediation/thread_fix_20261024/integration_result.json)
  and [log](../results/remediation/thread_fix_20261024/integration.log).

Integration command, executed with local assets and no API:

```bash
MPLCONFIGDIR=/private/tmp/edgenas-thread-fix-mpl .venv/bin/python -m src.benchmarking.cpu_benchmark results/remediation/thread_fix_20261024/integration.yaml --threads 1 --hardware-id continuation_mac_cpu_20261024
```

## New campaign

After the passing suite and integration, preflight found the matching local
checkpoint and 100-image manifest. A new campaign was started with:

```bash
MPLCONFIGDIR=/private/tmp/edgenas-thread-fix-mpl .venv/bin/python scripts/benchmark_campaign.py --hardware-id continuation_mac_cpu_20261024 --run-id threads1_fixed_20261024 --threads 1 --execute
```

The shared writable Matplotlib cache avoids rebuilding a font cache in every
subprocess; cache initialization remains outside timing. The campaign uses the
same seven resolutions, seeded order, 5 sessions, 10 configured warmups and 3
repetitions per 100 images. Sessions 1–2 remain predeclared stabilization
exclusions. It has a distinct run ID and protocol version; it does not change
or relabel historical evidence. Completion and independent raw verification are recorded below.

Index promotion remains blocked on adequate accuracy/checkpoint/dataset
provenance. No combined accuracy/latency selection claim is made by this fix.
No stage, commit, push, merge or paid API calls were performed. Git permission
and AppleDouble findings from the audit are unchanged.

## Completed campaign and independent verification

[Campaign manifest](../results/benchmarks/campaigns/threads1_fixed_20261024/campaign.json):
**completed**, 35/35 runs, 10,500 raw samples, 6,300 retained samples. Every run
records backend setup threads 7, then requested/observed 1 for warmup and timing;
all 10,500 timed samples record 1 at both boundaries. One environment signature
is present across runs. Each resolution retains 900 samples from sessions 3–5.

[Independent verification](../results/remediation/thread_fix_20261024/campaign_verification.json)
recomputed median/P95 from the raw files and compared their sample identities,
stage/thread evidence, checkpoint/image hashes, source hashes, schedule, raw
copies embedded in campaign JSON and session statistics. It also checked all
143 prior evidence files against the snapshot taken before the new campaign,
and all 127 tracked historical files against HEAD: **no content changes**.
The verifier does not execute a model or promote candidate records.

| Resolution | Pooled median ms | Pooled P95 ms | SD of retained session medians ms |
|---:|---:|---:|---:|
| 416 | 9.5850 | 10.2863 | 0.1186 |
| 448 | 12.3300 | 12.7111 | 0.0682 |
| 480 | 13.0655 | 14.9185 | 0.0519 |
| 512 | 13.6230 | 15.8328 | 0.0759 |
| 576 | 17.5055 | 19.3564 | 0.4073 |
| 608 | 18.3610 | 19.9458 | 0.0670 |
| 640 | 22.0320 | 22.7890 | 0.0377 |

Session SD uses the three rounded session medians saved by the runner, as in the
existing campaign summary. Repeated image timings are not independent samples;
no confidence interval is claimed. Documentation edits took place on the host
while the campaign ran; it was not an isolated benchmarking host. No automatic
performance comparison with the old 7-thread campaign or new combined
accuracy/latency selection result is claimed.

Commands verified, all exit 0:

```bash
MPLCONFIGDIR=/private/tmp/edgenas-thread-fix-mpl .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=/private/tmp/edgenas-thread-fix-mpl .venv/bin/python -m src.benchmarking.cpu_benchmark results/remediation/thread_fix_20261024/integration.yaml --threads 1 --hardware-id continuation_mac_cpu_20261024
MPLCONFIGDIR=/private/tmp/edgenas-thread-fix-mpl .venv/bin/python scripts/benchmark_campaign.py --hardware-id continuation_mac_cpu_20261024 --run-id threads1_fixed_20261024 --threads 1 --execute
.venv/bin/python results/remediation/thread_fix_20261024/verify_campaign.py
```

Use a **new run ID and output paths** for future execution; existing campaign
outputs are protected. The command block records what was run, not a suggestion
to overwrite it. Preflight output and all verification logs are in
`results/remediation/thread_fix_20261024/`.

## Current snapshot and commit proposal

The preceding source snapshots and audit artifacts remain unchanged. Current
source/config/test/document hashes are in
[source_sha256.json](../results/remediation/thread_fix_20261024/source_sha256.json).
The campaign additionally fingerprints the two executable scripts at start and
checks them before each run; those hashes match the final code used here.

Updated lists, **not staged**:

- [commit_include.txt](../results/remediation/thread_fix_20261024/commit_include.txt)
- [commit_exclude.txt](../results/remediation/thread_fix_20261024/commit_exclude.txt)
- [commit_candidates.json](../results/remediation/thread_fix_20261024/commit_candidates.json)

The updated proposal adds `tests/test_cpu_benchmark.py`, this follow-up document,
the selected fix/integration/test verification artifacts and all 72 files of the
new complete campaign (manifest, image manifest, 35 raw results, 35 configs).
It preserves the earlier selected evidence and keeps the old invalid campaign
with its explicit thread correction. Intermediate failed fixture log `tests.log`
is retained locally rather than proposed for commit; the final passing log is
`tests_verified.log`. Untracked owner files, raw full-host diagnostics, caches,
AppleDouble, data, weights, secrets and Git internals remain excluded. No Git
metadata or permissions were changed.


The lists above record the thread-fix handoff. The final commit-preparation
review, including subsequent documentation-only clarification and exact updated
include/exclude lists, is [FINAL_COMMIT_REVIEW.md](FINAL_COMMIT_REVIEW.md).
