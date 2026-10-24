# Báo cáo tiếp tục remediation EdgeNAS-Lite

Ngày thực hiện: 2026-10-24, múi giờ người dùng Asia/Taipei.

## Follow-up CPU threads

Bản sửa tiếp nối sau audit có **235 test pass** (229 là kết quả audit trước đó)
và integration thật xác nhận requested/observed = 1 sau backend setup. Campaign
mới `threads1_fixed_20261024` đã hoàn tất 35 lượt/10.500 raw samples, requested/observed = 1, dùng thread gate và protocol v2; bằng chứng và
trạng thái cuối được ghi tại [THREAD_CONTROL_FIX.md](THREAD_CONTROL_FIX.md).
Campaign cũ vẫn không đạt protocol 1-thread, giữ nguyên tất cả raw và số liệu.
Không promote index, chưa có accuracy/provenance mới đủ cho selection mới.

## Điểm xuất phát và bảo toàn workspace

- HEAD gốc: `d6653dfb6f2b8255149d9cf249d6af221bb57bdf`, branch `main`.
- Không tìm thấy AGENTS.md trong repository hoặc các thư mục cha đã kiểm tra;
  không tìm thấy báo cáo bàn giao riêng. README, setup và roadmap được đọc trước
  khi tiếp tục. File remediation là đặc tả, không được coi là bằng chứng đã làm.
- Thay đổi nội dung có sẵn: `src/requirement_parser/parser.py`,
  `tests/test_requirement_parser.py`, `src/benchmarking/cpu_benchmark.py`.
  Script `scripts/benchmark_campaign.py` và test campaign đã có nhưng chưa track.
  Chúng được tiếp tục sửa và kiểm chứng, không thay bằng implementation mới.
- Phần lớn thay đổi ban đầu khác quyền 100644/100755 trên ổ USB. Dùng
  `git -c core.filemode=false diff` để phân biệt nội dung, không đổi Git config
  hoặc quyền file hàng loạt. Git báo `non-monotonic index` với AppleDouble
  `.git/objects/pack/._*.idx`; không sửa/xóa object hoặc metadata Git.
- File chưa track có sẵn như `metadata_inspection.json`, backup candidate,
  baseline `remediation_baseline_20261024T132107Z.json` và batch workflow cũ
  được giữ nguyên. Không reset/clean, commit, push, merge hoặc gọi Gemini.
- `.git` được cấu hình **read-only trong sandbox của phiên làm việc**; đây không phải kết luận volume hoặc POSIX permissions bị khóa. Audit xác nhận USB exFAT mount đọc/ghi, owner có quyền ghi hiển thị, không thấy cờ immutable/ACL deny và 13 pack index thật đều verify thành công. Không tạo branch/commit hoặc sửa Git metadata; xem [audit và đề xuất khắc phục](REMEDIATION_AUDIT.md).
  Code cuối vẫn là working tree dirty trên HEAD trên, không phải commit đã phát hành.

## Môi trường và lỗi được phân loại

Môi trường thật được ghi trong
[environment.json](../results/remediation/continuation_20261024/environment.json).
Python `.venv` là 3.9.6, ARM64, torch 2.8.0, Ultralytics 8.4.123, numpy 2.0.2.
Python mặc định ngoài `.venv` là 3.14.8; không dùng nó để tuyên bố đã kiểm chứng
full ML environment.

Lệnh test đầu tiên không khởi động được vì Python đọc file AppleDouble
`._*.pth` như `.pth`. Thử bỏ qua site cho thấy thêm file `._*.mplstyle` làm lỗi
Matplotlib và subprocess vẫn gặp lỗi site. Đây là lỗi môi trường trên ổ USB.
Đã đổi tên các file phụ đó sang đuôi `.appledouble-disabled`, có thể hoàn tác;
không xóa hoặc sửa package code. Danh sách đầy đủ:
[environment_repair.json](../results/remediation/continuation_20261024/environment_repair.json).
`.gitignore` bỏ qua AppleDouble mới; quy tắc ignore không sửa dữ liệu đã tồn tại.

**Baseline sau sửa môi trường có 219 test pass**, trước khi thêm 10 regression tests
(`tests_environment_fixed.log`). **Suite cuối có 229 test pass**
(`tests_final_verified.log`); audit chạy lại cũng có **229 pass, 0 fail/error/skip**
([tests_rerun.log](../results/remediation/audit_20261024/tests_rerun.log)).
219 là mốc trung gian, không phải số test của bản bàn giao cuối. Các log khởi đầu
và trung gian được giữ trong thư mục bằng chứng. `pip check` exit 0, không broken
requirements; điều này không chứng minh fresh full install. Có cảnh báo LibreSSL
và Matplotlib cache không ghi được ở home; suite và demo vẫn chạy. Campaign
khởi động lại Matplotlib ở mỗi process, có overhead ngoài phạm vi timing.

## Trạng thái A–G

| Mục | Trạng thái | Kết quả và giới hạn |
|---|---|---|
| A — khảo sát/baseline | DONE | Bảo toàn dirty work, kiểm chứng baseline 7/4/416 và suite hiện hành. |
| B — validation/orchestration | DONE | Giữ finite validation; test YAML/JSON mọi constraint, overflow, CLI. Search Controller dùng chung pilot gate, yêu cầu hardware và kiểm tra cả tập record trước khi ghi evaluation/selection. |
| C — campaign/provenance | PARTIAL | Campaign cũ có mismatch 1/7; follow-up v2 mới đã xác minh 35 lượt/10.500 mẫu requested/observed = 1. Chưa accuracy/provenance mới đủ để promote index. |
| D — tái tạo | PARTIAL | Fresh lightweight replay được chạy; có manifest kiểm tra label/overlap và hướng dẫn retrain mới. Chưa khôi phục conversion/split/subset lịch sử, chưa retrain hoặc fresh-install full ML. |
| E — đánh giá LLM/search | PARTIAL | 36 held-out có nhãn và rubric; offline/live evaluator và equal-budget search reveal đã chuẩn bị. Không có live run mới; NOT_RUN giữ nguyên. Có agent review 3 clarification cũ; human review vẫn pending. |
| F — balanced/sensitivity | DONE | Giữ default/tie-break, giải thích scale và file-size; sensitivity 4 policies, Pareto và replay từ dữ liệu thật. Không thêm weights vào requirement schema. |
| G — CI/tài liệu/license | PARTIAL | CI core/full và demo có cấu hình, local checks có bằng chứng; README/setup/roadmap cập nhật. Chưa GitHub run; chủ repo chưa chọn license. |

## Thay đổi và hành vi

- Parser từ chối NaN, ±inf, bool, overflow trước orchestration; lỗi chỉ rõ field.
  Test thêm qua YAML/JSON thực, kể cả `1e309`, không chỉ gọi helper.
- Search Controller trước đây chỉ kiểm tra checkpoint, resolution và một nhãn
  standardized nên có thể reuse record sai hardware/protocol/split. Nay dùng
  `check_candidate_compatibility`, truyền hardware xuống runner, fail trước khi
  ghi báo cáo lựa chọn. Test public `run_search` xác nhận lỗi này. Candidate
  generation vẫn có thể để lại output trước khi một candidate khác thất bại;
  không tuyên bố transaction/crash recovery toàn hệ thống.
- Proposal thêm `--strict-comparison`: so sánh envelope provenance đầy đủ và
  bằng nhau. Record archived thiếu envelope bị loại; default vẫn là demo policy
  lịch sử, không chứng minh môi trường tương đương. Envelope là metadata được
  cung cấp, không phải cơ chế chứng thực. Chưa tự động đăng ký record mới.
- Campaign tiếp tục script cũ: sửa xung đột tên config giữa resolution, canonical
  root `/var`/`/private/var`, loại AppleDouble khỏi ảnh, kiểm tra manifest/raw count,
  giữ failed/interrupted status. Checkpoint dùng đường tương đối để tránh
  Ultralytics loại dấu nháy trong tên volume ở đường tuyệt đối.
- `dataset_manifest` kiểm tra file label và overlap theo hash/ID, giữ mapping
  0/3/5. Không đoán lại quy trình conversion/split lịch sử.
- Language evaluator tách schema/status/field/manual review. Search replay dùng
  4 requirement, 5 seed, initial 640 và budget 3; random không hoàn lại. Đường
  live chỉ đưa metrics đã reveal vào proposer, có lưu diễn tiến call và model/
  prompt fingerprint. Không đo token/cost nếu provider adapter không cung cấp.
- Sensitivity sử dụng cùng weights cho mọi candidate trong policy. Bốn policies
  được định trước, không thay công thức để ép candidate thắng. Đây là policy
  preference trên archived metrics, không là thí nghiệm inference mới.

## Kiểm chứng và bằng chứng trước thread fix

Thư mục chính: `results/remediation/continuation_20261024/`. Log được tạo ở run
mới; không ghi đè benchmark/eval lịch sử.

| Kiểm tra | Exit | Bằng chứng |
|---|---:|---|
| `.venv/bin/python -m unittest discover -s tests -v` | 0 | `tests_final_verified.log`: **229 pass, 0 fail/error/skip**; không gọi API/YOLO thực trong unit tests. |
| `.venv/bin/python -m pip check` | 0 | `pip_check.log`, không broken requirements. |
| Fresh Python 3.9 venv + `requirements-demo.txt` | 0 | `/private/tmp/edgenas-demo-continuation-20261024`, chỉ PyYAML 6.0.3; lệnh cài sandbox ban đầu lỗi DNS, lần được cấp quyền tải gói thành công. |
| Fresh demo parser suite | 0 | `fresh_demo_tests.log`: 12 test pass, 0 fail/skip. |
| Fresh demo structured selection | 0 | `fresh_demo_transcript.log`, `fresh_demo_selection.json`: 7 evaluated, 4 feasible, chọn 416. |
| `python -m scripts.verify_offline_demo --output-dir .../offline_scenarios` | 0 | `offline_scenarios/verification.json`; CLI no-feasible/hardware/strict trả exit 1 có báo cáo đúng trạng thái; exit 1 ở các ca này là kỳ vọng. |
| Fresh tools/campaign tests chỉ PyYAML | 0 | `fresh_tools_final.log`: 7 pass, `fresh_campaign_tests.log`: 4 pass; cộng parser 12 là 23 test core, không ML dependencies. |
| Campaign execute + raw verification | 0 | 35 lượt, 10.500 raw samples, 6.300 retained samples; **exit 0 không xác nhận thread protocol**. [Audit raw](../results/remediation/audit_20261024/campaign_raw_audit.json) xác nhận số học đúng nhưng 1 requested / 7 recorded threads. |
| Dataset manifest | 0 | `dataset_manifest.json`: 5985 train / 1496 val; không overlap hash/ID, labels hợp lệ theo checker. |
| Campaign preflight | 0 | `campaign_preflight_verified.log`; chỉ lập kế hoạch, không có số đo. |
| Language offline evaluator | 0 | `language_eval.json`: 36 NOT_RUN, không gán pass live. |
| Sensitivity/Pareto/search offline | 0 | `sensitivity_replay_v2.json`: archived replay; LLM NOT_RUN, latency/token/cost null. |
| `git -c core.filemode=false diff --check` | 0 | Không lỗi whitespace; các cảnh báo object AppleDouble là lỗi môi trường riêng. |

Demo transcript rút gọn, từ lệnh thật trong fresh lightweight environment:

```text
Status: selected
Retrieved: 7
Compatible: 7
Excluded: 0
Feasible: 4
Selected: yolo26n_kitti_pilot_imgsz416_cpu
```

Accuracy scenario chọn 512; constraint không khả thi trả
`no_feasible_candidate`; sai hardware và strict archived trả
`no_compatible_candidates`. Test thiếu protocol loại cả 7 candidate trước
constraint evaluation. Không nới lỏng hard constraints.

[Agent clarification review](../results/remediation/continuation_20261024/clarification_agent_review.json)
đọc nội dung ba câu hỏi thật, ghi rubric và lý do riêng từng ca. Không sửa pending
trong report lịch sử, không coi review của coding agent là human evaluation độc lập.

## Benchmark mới và ranh giới kết luận

Campaign đầu `continuation_measured_20261024` có status failed: Ultralytics loại
apostrophe khỏi đường checkpoint tuyệt đối, rồi thử kiểm tra assets trên mạng.
Không cần network để đo checkpoint local; đã sửa đường tương đối. Không có số đo
thành công nào được suy ra từ lần lỗi. Lần gọi tiếp theo gặp syntax error do sửa
parenthesis và đã được sửa trước retry, log được giữ riêng.

Campaign `continuation_measured_v2_20261024` đo bằng checkpoint local và 100 ảnh
được hash. Toàn bộ output nằm ngoài archived candidate records. [Campaign đã hoàn tất](../results/benchmarks/campaigns/continuation_measured_v2_20261024/campaign.json)
và [được đối chiếu lại từ raw](../results/remediation/continuation_20261024/campaign_verification.json):
35 lượt, 300 samples/lượt, mỗi image/repetition xuất hiện đúng một lần, 5 session
mỗi resolution. Environment signature giữa 35 lượt giống nhau. Máy đo báo
macOS 26.6.2 ARM64, Python 3.9.6; **plan yêu cầu 1 CPU thread nhưng cả 35 raw record
đều ghi `environment.torch_num_threads = 7`**. Báo cáo trước ghi 1 thread là sai.
Raw ghi giá trị ở bước build result, không log thread theo từng sample. Audit tái
hiện `select_device('cpu')` đổi 1 thành 7: benchmark đặt thread trước khi predictor
khởi tạo, sau đó Ultralytics reset thread. Campaign kiểm tra hash/samples nhưng
chưa kiểm tra thread mismatch nên vẫn đánh dấu `completed`. Không dùng campaign
này làm bằng chứng đạt protocol 1-thread. Giữ nguyên raw/plan/report cũ, dùng audit
bổ sung để đính chính. Không mặc định 416 hoặc policy cũ được tái lập trên môi trường mới.

| Resolution | Pooled median ms | Pooled P95 ms | SD median giữa session ms |
|---:|---:|---:|---:|
| 416 | 8.7275 | 9.0181 | 0.0330 |
| 448 | 10.6200 | 10.9792 | 0.0397 |
| 480 | 11.3255 | 14.3694 | 0.5148 |
| 512 | 11.4830 | 12.7081 | 0.4995 |
| 576 | 13.5750 | 15.6242 | 0.6887 |
| 608 | 13.9785 | 14.4471 | 0.0787 |
| 640 | 16.9510 | 19.5728 | 0.2731 |

Thống kê pooled dùng session 3–5, 900 samples/resolution; SD là population SD của ba median session đã lưu (làm tròn bởi runner). Không báo confidence interval cho các lần đo ảnh lặp.


Không có benchmark accuracy mới, không có live Gemini mới, không có before/after
model-quality claim. Session timings chạy trên máy đang dùng để sửa code và test,
không là host benchmark cô lập; kết quả chỉ mô tả campaign đó. Hash checkpoint
vừa quan sát không tự chứng minh checkpoint trùng nội dung weights lịch sử.

## Blocker và lệnh tiếp tục

- **Live language/search NOT_RUN:** cần model được tài khoản hỗ trợ và ngân sách
  tiền được chủ repo cho phép rõ ràng. Có key hoặc experiment budget không đủ.
  Sau cấp phép: `python -m scripts.evaluate_language --live --model MODEL --max-calls 36 --output NEW.json`
  và `python -m scripts.analyze_selection --live --model MODEL --max-calls 60 --output NEW.json`.
  Lệnh thứ hai vẫn là archived-measurement reveal, không là online inference.
- **Human clarification review:** người review độc lập đọc 3 response cũ theo
  rubric đã nêu; giữ kết luận riêng với agent review.
- **Accuracy/data lineage:** cần nguồn/manifest conversion và split lịch sử,
  hoặc tuyên bố dataset version mới rồi validation/retrain riêng. Xem
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md); không có weights download được xác minh.
- **Thread protocol chưa đạt:** cần enforce và verify thread sau predictor/backend initialization, trước warmup/timing và trong từng run; campaign phải fail khi requested/observed mismatch. Lượt audit gốc chưa sửa; follow-up đã sửa/test và chạy campaign bằng run ID mới như liên kết đầu báo cáo. Không relabel campaign cũ thành 1-thread.
- **Promote benchmark:** cần kiểm chứng full-validation accuracy manifest, classes,
  checkpoint và environment envelope trước khi tạo candidate/index mới. Không
  tự ghép accuracy cũ với latency mới thành strict evidence.
- **CI remote:** cần chủ repo cho phép push và một GitHub Actions run thật; chưa
  thực hiện theo yêu cầu phiên này. Workflow YAML không đồng nghĩa CI xanh.
- **License:** chủ repo phải chọn license cho code thuộc quyền sở hữu, kiểm tra
  third-party/data/weights terms trước phân phối. Xem [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Sensitivity và manifest local

Từ archived metrics, equal weights chọn 416, accuracy-heavy (8/1/1) chọn 512,
latency-heavy và size-heavy chọn 416. Cả 7 cấu hình thuộc Pareto frontier của
pool accuracy/latency hiện có. Đây không là phép đo mới hoặc bằng chứng policy
nào đúng hơn. Search replay có 20 tổ hợp requirement/seed; không có LLM run mới.

Manifest local có SHA-256 `c57e19f061cd2ea143771d4ad6acbeaa8f1f1d64b6dc65e42fd3f84f33d88058`.
Kiểm tra overlap/label thành công chỉ xác nhận tập file hiện tại; không khôi phục
lịch sử tạo split và không xác minh accuracy mới. Expanded live evaluator/search
chưa được kiểm chứng với API thật, GitHub Actions chưa chạy trên server, và
fresh full ML install chưa được thực hiện.

## Mức sẵn sàng portfolio

Demo engineering offline đã có bằng chứng chạy trong môi trường nhẹ mới.
Validation và orchestration có regression, default archived selection được giữ.
Công cụ benchmark/eval/replay có thể tiếp tục mà không cần đổi kiến trúc.
Tái tạo nghiên cứu lịch sử chưa hoàn chỉnh; không có bằng chứng LLM search
vượt ascending/random, không claim architecture NAS hoặc cross-platform support.


## Định danh code cuối

[final_source_sha256.json](../results/remediation/continuation_20261024/final_source_sha256.json)
ghi hash của 80 source/config/test/tài liệu tại lần bàn giao trước. Audit đã so sánh
**80/80 hash khớp trước khi chỉnh tài liệu trong lượt audit này**; xem
[original_source_snapshot_check.json](../results/remediation/audit_20261024/original_source_snapshot_check.json).
Không ghi đè snapshot cũ. Tài liệu đã đính chính sẽ khác snapshot đó theo chủ đích;
[source snapshot sau audit](../results/remediation/audit_20261024/source_sha256_after_audit.json)
định danh trạng thái audit trước thread fix. [Snapshot sau thread fix](../results/remediation/thread_fix_20261024/source_sha256.json) định danh code hiện tại. Campaign cũ chưa có source fingerprint tại thời điểm bắt đầu,
nên snapshot cuối không chứng minh byte-for-byte source đã chạy campaign.
229 test pass là bằng chứng software regression, không là accuracy của Gemini.
Các runtime test dùng fixtures/mocks được tách khỏi campaign YOLO đo thật.
