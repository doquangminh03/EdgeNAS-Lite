# Installation and usage

Run all commands from the EdgeNAS-Lite project root. Shell examples use macOS/Linux syntax.

## 1. Install

The verified environment is Python 3.9.6 on macOS ARM64, with PyTorch 2.8.0 and Ultralytics 8.4.123. Pinned dependencies and tests passed in a clean environment on the original machine. Other platforms have not been validated; these pins are not a cross-platform compatibility guarantee.

```bash
git clone https://github.com/doquangminh03/EdgeNAS-Lite.git
cd EdgeNAS-Lite
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py --mode demo
```

Use a Python interpreter matching the verified version when reproducing that environment. On Windows PowerShell, environment activation is `.venv\Scripts\Activate.ps1`; adapt shell commands accordingly.

## 2. Select from recorded evidence without an API key

This is the quickest way to try the project. It needs the indexed candidate records, but does not load model weights or dataset images and makes no Gemini request.

```bash
python -m src.proposal.cli \
  configs/requests/low_latency_balanced_demo.yaml \
  --hardware-id local_mac_cpu_01 \
  --model-family YOLO26 \
  --output "results/proposals/local_selection_$(date +%Y%m%d_%H%M%S).json"
```

Expected result with the current index: **7 evaluated, 4 feasible, selected `yolo26n_kitti_pilot_imgsz416_cpu`**, with mAP50–95 0.207384, median latency 8.754 ms, and model size 5.102 MB.

The hardware ID identifies the original measurement target. Using it on another computer selects from evidence for that original target; it does not measure or predict the new computer's latency.

The structured CLI can replace an existing output file. Use a unique filename to preserve previous results.

## 3. Use a natural-language request

Set your own `GEMINI_API_KEY` in the terminal environment. The following prompt keeps the key out of terminal input history and display:

```bash
printf 'Gemini API key: '
read -r -s GEMINI_API_KEY
printf '\n'
export GEMINI_API_KEY
```

Automatic `.env` loading is not established. The implemented default model is `gemini-3.5-flash-lite`; use `--model MODEL_ID` to override it with a model supported by your provider account. This workflow makes a live API call and requires network access.

```bash
python -m src.llm_agent.workflow \
  --text "I need object detection on KITTI using CPU. Minimum mAP50-95 is 0.20, maximum median latency is 12 ms, and maximum model file size is 6 MB. Use balanced optimization." \
  --run-id "balanced_$(date +%Y%m%d_%H%M%S)" \
  --hardware-id local_mac_cpu_01 \
  --budget 0
```

Budget 0 uses Gemini for interpretation and selects from stored measurements without new YOLO evaluation. Review the saved requirement to check that it matches your request.

To require higher accuracy:

```bash
python -m src.llm_agent.workflow \
  --text "I need object detection on KITTI using CPU. Minimum mAP50-95 is 0.24, maximum median latency is 12 ms, and maximum model file size is 6 MB. Prioritize accuracy." \
  --run-id "accuracy_$(date +%Y%m%d_%H%M%S)" \
  --hardware-id local_mac_cpu_01 \
  --budget 0
```

If interpreted as specified, the current measurements yield resolution **512**: mAP50–95 0.242931 and median latency 11.279 ms. Unsupported or incomplete requests may require clarification; impossible constraints may produce no feasible candidate.

## 4. Find and read the output

Each natural-language run writes to `results/workflows/<run_id>/`:

| File | Contents |
|---|---|
| `interpretation.json` | Interpretation result and initial proposal |
| `requirement.yaml` | Validated requirement, when interpretation is ready |
| `workflow.json` | Workflow status and final proposal, plus batch details when applicable |

In `workflow.json`, inspect `final_proposal.selection` for `selected_candidate_id`, `selected_metrics`, `ranking`, and `rejected_candidates`. A completed workflow does not by itself mean a feasible candidate was selected.

Use a new run ID for every workflow run. Existing run directories are protected, and the workflow has no resume option. If clarification is requested, provide a more complete request in a new run.

## 5. Optional experiment execution

New measurements require local KITTI images and labels, the trained checkpoint at `runs/detect/kitti_yolo26n_pilot/weights/best.pt`, and a compatible template candidate record. Dataset files and full model run directories are not included in a normal clone. Obtain or reproduce the matching assets before attempting evaluation; an unrelated checkpoint does not reproduce this pilot.

```bash
python scripts/check_environment.py --mode evaluate
python -m src.llm_agent.workflow --help
python -m src.candidate_runner.runner --help
```

The environment checker looks for `kitti.yaml` in the project root or installed Ultralytics package. It checks existence and selected metadata, but does not validate dataset images/labels, checkpoint contents, API credentials, or connectivity.

To enable experiments, change a workflow command to `--budget 1` (supported range 1–7) and add `--template-candidate-id yolo26n_kitti_pilot`. The current seven-resolution pool is fully measured, so the experiment branch can stop with `exhausted` / `no_available_candidate` without running YOLO. A positive budget permits experiments; it does not force remeasurement.

New hardware, datasets, or checkpoints require corresponding measurements and compatible metadata. Inspect each controller's `--help` before using its execution or recovery options. Failed or interrupted measurements may leave partial artifacts; workflow execution does not automatically resume them.

## 6. Tests and reports

Run the software test suite:

```bash
python -m unittest discover -s tests -v
```

Historical evidence records 211 tests passing in a clean environment. Fresh-clone verification used that environment on the same machine; it did not call Gemini or run new YOLO measurements.

- [Clean-environment verification](results/reproducibility/clean_env_001/report.json)
- [Fresh-clone verification](results/reproducibility/fresh_clone_002/report.json)
- [Portfolio report](results/portfolio/demo_v1/REPORT.md)
- [Candidate index](knowledge/index.yaml)

To generate another portfolio report from the saved verification and current records:

```bash
python scripts/build_portfolio_report.py \
  --output "results/portfolio/report_$(date +%Y%m%d_%H%M%S)"
```

The generator requires matching evidence and a new output directory. It produces Markdown, CSV, PNG, SVG, and JSON without API calls or model execution. It targets the original balanced pilot thresholds; it is not a report generator for arbitrary workflow requests.

## Troubleshooting

| Issue | Action |
|---|---|
| Module not found | Run from the cloned project root with the intended virtual environment active; check that the source file exists in the checkout. |
| Dependency installation fails | Compare Python version and platform with the verified environment; inspect the first installation error. |
| Gemini authentication or model error | Check the exported key, provider access, and `--model` setting. |
| Run directory already exists | Choose a new `--run-id`. |
| No feasible candidate | Inspect failed constraints and decide whether to revise requirements or collect additional measurements. |
| Missing dataset or checkpoint | Supply the matching local assets before running evaluation; recorded-evidence selection can run without them. |

Keep API keys, virtual environments, datasets, and local backups out of Git. Review generated logs before sharing them.
