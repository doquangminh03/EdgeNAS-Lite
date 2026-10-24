# Audit báo cáo remediation — 2026-10-24

Phần audit gốc dưới đây chỉ đọc source và bằng chứng, chạy lại test offline và chỉnh tài liệu.
Không sửa application code, benchmark/eval lịch sử, Git metadata hoặc quyền file;
không commit, stage, push, merge, đo lại campaign hoặc gọi API trả phí.

## Follow-up: đã sửa thread control

Lượt tiếp nối đã sửa benchmark/campaign, thêm regression và chạy **235 test pass**.
Integration thật với checkpoint local xác nhận backend setup ghi 7, sau enforce
thì warmup và mọi boundary timing ghi 1. Campaign mới dùng run ID
`threads1_fixed_20261024` và protocol `cpu_campaign_v2` đã hoàn tất 35 lượt/10.500 raw samples, mọi boundary timing ghi 1; xem
[THREAD_CONTROL_FIX.md](THREAD_CONTROL_FIX.md) cho trạng thái và bằng chứng mới.
Kết quả audit 229 test và sai lệch campaign cũ dưới đây vẫn là bằng chứng lịch sử.
Không sửa raw cũ hoặc promote index.

## Kết quả đối chiếu tại lần audit gốc

| Hạng mục | Kết quả | Bằng chứng |
|---|---|---|
| Suite hiện tại | **229 pass, 0 fail/error/skip**, exit 0 | [tests_rerun.log](../results/remediation/audit_20261024/tests_rerun.log) |
| Mốc 219/229 | 219 là baseline sau sửa môi trường; thêm 10 regression thành 229 ở bản bàn giao cuối. Báo cáo đã sửa cách diễn đạt. | `tests_environment_fixed.log`: 219; `tests_final_verified.log`: 229; rerun: 229 |
| Hash trước chỉnh tài liệu | **80/80 khớp**, không thiếu file | [original_source_snapshot_check.json](../results/remediation/audit_20261024/original_source_snapshot_check.json) |
| File lịch sử tracked | **127/127 khớp HEAD** trong `results/`, `knowledge/`, `artifacts/`, `configs/` | [tracked_content_vs_HEAD.json](../results/remediation/audit_20261024/tracked_content_vs_HEAD.json) |
| Toàn bộ tracked blobs | So sánh byte bằng Git blob ID với HEAD `d6653df...`: 195 file, 13 thay đổi nội dung đúng phạm vi implementation/docs; không dựa vào mtime/index cache hoặc mode | Cùng file audit tracked ở trên |
| Raw campaign | **35 lượt × 300 = 10.500 mẫu**, 100 ảnh mỗi lượt, mỗi cặp image/repetition xuất hiện đúng một lần; 6.300 mẫu giữ lại, 900/resolution | [campaign_raw_audit.json](../results/remediation/audit_20261024/campaign_raw_audit.json) |
| Median/P95/SD | Tính lại từ raw khớp campaign và verification cũ; SD dùng median session đã làm tròn của runner | Cùng file audit raw |
| Hash campaign/assets | Campaign JSON khớp hash cũ; manifest canonical đúng hash; checkpoint và 100 file ảnh hiện tại khớp hash đã lưu | Cùng file audit raw |
| Thread protocol | **FAIL về đối chiếu khai báo:** plan yêu cầu 1; cả 35 raw record ghi 7 | Audit raw và [thread_reset_reproduction.json](../results/remediation/audit_20261024/thread_reset_reproduction.json) |
| Git pack thật | **13/13 `git verify-pack` exit 0** | [git_pack_verification.json](../results/remediation/audit_20261024/git_pack_verification.json) |

Snapshot cũ `final_source_sha256.json` được giữ nguyên. Sau khi sửa báo cáo và
hướng dẫn, các hash tài liệu tương ứng thay đổi có chủ đích; dùng
[source_sha256_after_audit.json](../results/remediation/audit_20261024/source_sha256_after_audit.json)
cho trạng thái sau audit. Không suy diễn snapshot cuối là source fingerprint tại
thời điểm campaign bắt đầu: campaign chỉ có git SHA và dirty flag.

So sánh lịch sử chỉ áp dụng cho các file có baseline trong HEAD. Không có baseline
đáng tin cậy để kết luận nội dung trước lần remediation của các file owner chưa
track như `metadata_inspection.json`, backup candidate hoặc batch workflow cũ.
Audit này giữ chúng tại chỗ, không đưa vào proposed commit.

## Sai lệch thread và giới hạn campaign

Trong `src/benchmarking/cpu_benchmark.py`, `torch.set_num_threads(--threads)` chạy
trước khi gọi YOLO/warmup. Predictor của Ultralytics gọi `select_device()` ở bước
setup backend; implementation đã cài tại
`ultralytics/utils/torch_utils.py` đặt lại `torch.set_num_threads(NUM_THREADS)`
cho CPU. Trên máy này `NUM_THREADS = 7`. Gọi trực tiếp helper đó tái hiện **1 → 7**,
không load weights, không inference hoặc network.

Metadata raw được ghi lúc build result, không có thread measurement cho từng
sample. Bằng chứng chắc chắn là 35 giá trị được ghi đều bằng 7 và cơ chế reset
đã được tái hiện; không gọi đây là phép đo 1-thread đã kiểm chứng. Script campaign
hiện chưa đối chiếu requested/observed threads nên status `completed` chỉ phản
ánh các subprocess hoàn tất. Verification cũ đúng về số học nhưng thiếu kiểm tra
này; báo cáo audit bổ sung giới hạn đó, không sửa raw/manifest cũ thành số khác.

Tại thời điểm audit gốc chưa sửa code. Các bước sửa đề xuất khi đó là khởi tạo
predictor/backend trước, enforce thread trước warmup/timing, lưu requested cùng
observed và fail khi lệch; thêm regression mô phỏng backend reset. Sau sửa cần
campaign với run ID mới. Không relabel hoặc tái sử dụng bộ hiện tại làm evidence
đạt protocol 1-thread. Follow-up đã thực hiện phần thread control như liên kết trên; accuracy mới, index promotion và live LLM vẫn chưa chạy.

## `.git` chỉ đọc: nguyên nhân và cách khắc phục

Có hai vấn đề riêng:

1. **Quyền sandbox của phiên:** cấu hình filesystem cấp quyền read cho `.git`.
   Mount USB thực tế là exFAT với `local,nodev,nosuid,noowners,noatime,fskit`,
   không có `read-only`; `ls -ldeO@` cho thấy mode `.git` là `drwx------`, owner
   là user đang chạy, không cờ immutable và không ACL deny hiển thị. Volume còn
   khoảng 37 GiB. Bằng chứng không chỉ ra cần `chmod -R` hoặc `chown -R`.
   Chưa thử ghi vào `.git` để giữ nguyên trạng. Thay quyền POSIX không cấp thêm
   quyền cho sandbox; muốn agent ghi Git cần phiên có quyền Git phù hợp hoặc
   approval có phạm vi cho lệnh Git cụ thể. Audit này không yêu cầu/thực hiện.
2. **AppleDouble trên exFAT:** sidecar `._pack-*.idx` có magic `00051607`, khác
   real Git index `ff744f63`. Git quét sidecar như index và báo `non-monotonic`.
   Toàn bộ 13 real index/pack đã verify thành công. Đây chưa phải bằng chứng
   object database thật bị hỏng. Không xóa sidecar hoặc chạy sửa Git tự động.

`diskutil info` không truy cập được DiskManagement framework trong môi trường
thực thi; nhận định filesystem ở đây dựa vào `mount`, `df`, `ls` và verify-pack,
không dựa vào kết quả `diskutil`. Chưa xác nhận thao tác ghi từ shell ngoài
sandbox hoặc audit toàn bộ reflog/unreachable objects.

Phương án ưu tiên bảo toàn dữ liệu, **chỉ đề xuất, chưa chạy**:

1. Tạo backup độc lập trên APFS của toàn bộ workspace, bao gồm `.git`, refs,
   reflogs, index, tracked/untracked/ignored files cần giữ; kiểm tra checksum.
   Bản clone không thay thế backup này vì không giữ toàn bộ uncommitted state
   hoặc dangling objects. Không format USB.
2. Khi có thư mục APFS đích và quyền phù hợp, tạo clone mới bằng Git với
   `--no-local --no-hardlinks` từ repo hiện tại, giữ repo gốc nguyên trạng.
   Git transfer tái tạo object database ở đích, tránh sao chép file phụ AppleDouble
   như một phần repository đang dùng. Verify clone/HEAD và packs trước khi dùng.
3. Xuất tracked patch bằng `git -c core.filemode=false diff --binary HEAD`;
   lệnh này bao gồm staged + unstaged so với HEAD. Ghi riêng inventory/hash của
   các file chưa track; copy theo danh sách rõ ràng. Không dùng `git clean`,
   `reset --hard`, `chmod -R`, `chown -R`, `dot_clean` hoặc xóa `.git`/metadata.
4. Áp dụng patch và copy file đã chọn sang clone mới, so lại source/evidence
   hashes, diff, test. Giữ backup và repo gốc cho tới khi xác nhận. Kiểm tra và
   thiết lập remote đúng ở clone sau khi review; clone local có thể trỏ origin
   về đường dẫn USB. Chưa commit/push trong quy trình audit này.

Giải pháp ngắn hạn là tiếp tục review ở repo hiện tại với
`git -c core.filemode=false ...`; đây chỉ bỏ nhiễu mode trong từng lệnh, không
đổi config hoặc khắc phục AppleDouble/quyền sandbox. Không cần sửa metadata
Git chỉ để đọc source hoặc xác minh raw evidence.

## Phạm vi proposed commit

Danh sách bên dưới là đề xuất tại audit gốc. Danh sách cập nhật sau thread fix nằm trong [THREAD_CONTROL_FIX.md](THREAD_CONTROL_FIX.md). Tất cả vẫn **chưa stage**:

- [commit_include.txt](../results/remediation/audit_20261024/commit_include.txt)
- [commit_exclude.txt](../results/remediation/audit_20261024/commit_exclude.txt)
- [commit_candidates.json](../results/remediation/audit_20261024/commit_candidates.json), có lý do từng nhóm.

Nên đưa vào commit:

- 13 tracked file đã sửa: `.gitignore`, README, setup, roadmap; benchmark CPU;
  proposal CLI/compatibility/rule-based; requirement parser; Search Controller;
  tests parser/controller/integration. Đây gồm cả phần agent trước đã được review,
  không chỉ những dòng thay đổi ở lượt audit này.
- CI, `requirements-demo.txt`, 36 language labels, 5 script remediation,
  2 test module mới và 4 tài liệu trong `docs/`.
- Bộ campaign thành công đầy đủ: campaign JSON, image manifest, 35 raw JSON và
  35 session configs; kèm audit đính chính thread. Có thể commit cả 3 file của
  campaign lỗi cùng log lỗi để giải thích lần restart; không coi đó là số đo.
- Bằng chứng chọn lọc: log cuối 229, baseline 219, 23 core tests trong fresh
  environment, environment/repair metadata, demo scenarios, manifest hashes,
  sensitivity v2, language NOT_RUN, agent clarification review và audit mới.

Loại khỏi proposed commit, vẫn giữ local:

- `EdgeNAS_Lite_Codex_Remediation.md`, `metadata_inspection.json`,
  `results/candidates/yolo26n_kitti_pilot_imgsz448_cpu.json.before_hardware_id.bak`,
  `results/llm_batches/workflow_english_demo_002.json`,
  `results/proposals/remediation_baseline_20261024T132107Z.json`:
  đều là file owner chưa track có sẵn, cần review riêng thay vì gom tự động.
- Log test trung gian/trùng, sensitivity v1 và output preflight cũ; danh sách
  exclude liệt kê chính xác. Giữ log thành công cuối và các lỗi startup/campaign
  cần cho provenance. File diagnostics filesystem đầy đủ chứa mount/user paths
  ngoài phạm vi dự án được giữ local; đưa bản assessment có chọn lọc vào commit.
- `.git/**`, AppleDouble `._*`, `.DS_Store`, `.venv`, cache, `.env`, dataset,
  run weights và `.pt`/`.onnx`. Các file lịch sử đã track và không đổi nội dung
  không cần stage lại chỉ vì mode USB khác.

Không dùng `git add .`. Danh sách include chỉ là đề xuất để review; chưa có
stage/commit, chưa GitHub CI, chưa API live. Ghi chú chưa sửa thread thuộc audit gốc đã được thay thế bởi follow-up ở đầu tài liệu.
