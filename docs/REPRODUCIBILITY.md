# Reproduction levels

## Replay (no weights, dataset or Gemini)

Use Python 3.9 and `pip install -r requirements-demo.txt`, then run the offline
selection command in SETUP_AND_USAGE. This uses archived measurements associated
with `local_mac_cpu_01`; running on another host does not measure that host.

Run all offline scenarios in a new output directory:

```bash
python -m scripts.verify_offline_demo --output-dir results/remediation/my_unique_demo
python -m scripts.analyze_selection --output results/remediation/my_unique_analysis.json
python -m scripts.evaluate_language --output results/remediation/my_unique_language.json
```

Search replay has four predefined requirements, initial resolution 640, budget 3, and random seeds 11/29/47/83/101. `analyze_selection --live --model MODEL --max-calls 60` prepares a same-budget LLM comparison, saving progressively revealed evidence calls; it requires separate API budget authorization and has not been run here. Random seeds do not control provider sampling. Saved orders can be supplied with `--orders`, but lack equivalent provenance unless recorded with this protocol.

The language evaluator leaves absent saved responses NOT_RUN. Provide
`--responses DIRECTORY` with `<case_id>.json` envelopes containing `response_text`,
`model_id`, `timestamp_utc`, and prompt/config identifiers to evaluate saved model
responses. Do not substitute expected labels for model responses. A live run
requires a separately authorized API budget, `--live --model MODEL --max-calls 36`;
this project has not run the expanded suite live. A call-count limit is not a
monetary spending limit. Old 6 semantic and 3 clarification responses remain
regression evidence; the added agent review is separate from human review.

## Remeasure (matching local assets)

Install `requirements.txt`. Supply the pilot checkpoint at
`runs/detect/kitti_yolo26n_pilot/weights/best.pt` and YOLO-format KITTI under
`datasets/kitti/{images,labels}/{train,val}`. No verified public pilot-checkpoint
download is available in this repository. The untrained `yolo26n.pt` is not a
substitute. The local checkpoint observed during continuation has SHA-256
`301f3942e2171b57b077e55002f10946753fa829fbe348d3c61657858dfff749`;
this observation does not retroactively identify historical weights.

```bash
python -m scripts.dataset_manifest --dataset-root datasets/kitti --output results/remediation/my_unique_manifest.json
python scripts/benchmark_campaign.py --hardware-id YOUR_ACTUAL_TARGET_ID --run-id my_unique_campaign --threads 1
python scripts/benchmark_campaign.py --hardware-id YOUR_ACTUAL_TARGET_ID --run-id my_unique_campaign --threads 1 --execute
```

**Thread control update (2026-10-24):** new benchmarks initialize the lazy
backend through one untimed setup prediction, then enforce the requested torch
intra-op threads before configured warmup and timing. They check both boundaries
of every prediction and fail on mismatch. Raw metadata stores requested, observed
and per-sample boundary counts; `cpu_campaign_v2` rejects missing/mismatched
thread evidence. The old continuation campaign requested 1 but recorded 7 and
remains invalid as 1-thread evidence. See [thread fix and new measurement](THREAD_CONTROL_FIX.md).

Preflight writes nothing and loads no model. Execute creates a new directory and
refuses existing output. All seven resolutions use the same 100-image manifest,
batch 1, seed 42 for image selection and per-repetition shuffle, 10 warmups,
3 repetitions, and 5 sessions. Session resolution order is shuffled using seed
20261024 + session number. Sessions 1–2 are excluded by declaration. Raw timings
include the `model.predict` call (preprocess, inference, postprocess and Python
call overhead), excluding image loading. No separate component latency is claimed.
P95 uses linear interpolation. Session median population SD describes observed
variability; no confidence interval or independence claim is made for 900 repeated
image timings. Protocol `cpu_campaign_v2` uses raw benchmark `cpu_threads_v2` and adds one untimed backend setup prediction before the 10 configured warmups. Historical
candidate records retain `cpu_v1` and are not rewritten.

Failed/interrupted campaigns retain a non-completed report and never register
candidates. Every session is required; do not remove slow sessions. Campaign
completion alone does not establish accuracy equivalence. Validate accuracy
using the same hashed checkpoint, full validation image/label manifest and
class mapping before creating replacement candidate records. The campaign does
not automatically promote its output into `knowledge/index.yaml`.

`--strict-comparison` on the proposal CLI requires matching, complete top-level
`comparison_provenance`: checkpoint and benchmark/accuracy manifest SHA-256,
OS (`system`), architecture, Python/torch/ultralytics/numpy versions, torch thread
count, protocol version, timing scope, batch size, classes and split. It rejects
archived records that lack this envelope. This is an equality check of supplied
metadata, not proof that the metadata is truthful. Default proposal behavior
retains `kitti_cpu_pilot_v1` for archived demonstrations. Search Controller now
requires `--hardware-id` for execution and uses that same pilot compatibility
check before writing selection reports, in addition to checkpoint/resolution
checks. A dry run is planning only.

## Retrain (a new experiment)

The checked-in `configs/kitti_pilot.yaml` records 10 epochs, fraction 0.25,
image size 640, batch 4, seed 42, deterministic mode, classes 0/3/5. Local dataset
YAML maps these IDs to car/pedestrian/cyclist. The exact historical raw-label
conversion, split generation and selected training subset were not recovered.
Do not describe retraining as bit-for-bit pilot reproduction.

The local dataset YAML contains an Ultralytics ZIP source and the class mapping.
Use an authorized KITTI source and applicable terms before acquiring data. No
new download, raw-label converter or train/validation split algorithm has been
validated here. For a new versioned experiment, record the source archive hash,
conversion implementation, split seed and image IDs, and run `dataset_manifest`
before training. It rejects image-ID/hash overlap, missing labels and invalid
normalized labels. Empty label files represent background images. It does not
establish source rights or historical lineage.

Once an authorized YOLO-format dataset is prepared, write a local dataset YAML
with an absolute `path`, `train: images/train`, `val: images/val`, and the exact
class mapping from `datasets/kitti/kitti.yaml`. Then run with a unique run name:

```bash
yolo detect train cfg=configs/kitti_pilot.yaml data=/absolute/path/to/local_kitti.yaml project=runs/retrained name=my_unique_retrain
```

This command is supplied for continuation, not claimed as executed. Verify the
resulting checkpoint hash, selected class IDs and full-validation metrics before
benchmarking it. Dataset and weights remain outside Git.
