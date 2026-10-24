# Chuẩn bị commit cuối — 2026-10-24

Không thêm tính năng hoặc đổi application code ở lượt này. Rà soát diff so với
HEAD `d6653dfb6f2b8255149d9cf249d6af221bb57bdf`, các file mới trong danh sách
include, các snapshot và log đã có. Chỉ bổ sung nhãn archived ngay trước bảng
kết quả README cùng tài liệu/bằng chứng chuẩn bị commit.

## Ranh giới bằng chứng đã kiểm tra

| Bộ dữ liệu/bằng chứng | Ý nghĩa khi đọc tài liệu |
|---|---|
| `knowledge/index.yaml`, archived `cpu_v1` | Demo giữ nguyên 7 evaluated / 4 feasible / chọn 416, latency 8.754 ms. Không phải selection từ campaign 1-thread mới. |
| `continuation_measured_v2_20261024` | 35 lượt có số học đã đối chiếu, raw ghi 7 threads dù plan yêu cầu 1. Không đạt protocol 1-thread. Giữ nguyên để truy vết. |
| `threads1_fixed_20261024`, `cpu_campaign_v2` / `cpu_threads_v2` | 35 lượt, 10.500 raw samples, requested/observed = 1 ở các boundary đã ghi. Pooled median 416 là 9.585 ms; đây là latency-only evidence, không ghép với accuracy cũ để tạo selection mới. |

README đánh dấu archived ngay trước bảng kết quả. Setup, reproducibility,
remediation report, audit và thread-fix report phân biệt các bộ trên. Chưa
accuracy/provenance mới đủ để promote index. Snapshot nguồn và số học campaign
đã được kiểm chứng ở lượt trước; rà soát hiện tại khớp 82/82 hash trước khi chỉnh
tài liệu. Không đo lại model hoặc mở lại công việc thread fix.

## Diff, secrets và file ngoài phạm vi

Diff nội dung hiện có bao gồm 14 tracked files cùng các file remediation mới.
Các thay đổi mode trên USB không được gom vào commit: dùng
`git -c core.filemode=false ...` khi review. Không sửa Git config hoặc quyền.

Danh sách thread-fix ban đầu có 244 include / 23 exclude, không trùng, không
thiếu, không overlap; 14 tracked content changes đều thuộc include, mọi file
untracked không bị ignore đều được phân loại. Danh sách cuối bên dưới bổ sung
chính các file chuẩn bị commit ở lượt này; danh sách cũ được giữ nguyên.

Quét nội dung text trong 425 đường dẫn (union của tracked files và 244 include
ban đầu) không tìm thấy pattern credential thông dụng: Google/GitHub/OpenAI/AWS
access IDs, private-key headers, credential URLs, bearer tokens và literal
secret assignments. Bốn file ảnh nhị phân lịch sử không được OCR; chúng không
nằm trong thay đổi nội dung hoặc include. Quét pattern không bảo đảm phát hiện
mọi loại secret. Không đọc/in giá trị biến môi trường hoặc nội dung `.env`.
Các file chuẩn bị commit mới cũng được quét trước bàn giao.

Không include `.git`, `.env`, AppleDouble, venv/cache, dataset, run weights,
`.pt`/`.onnx`, backup `.bak`, hoặc file owner chưa track ngoài nhiệm vụ.
Các path tuyệt đối trong campaign configs/logs là provenance của máy đo và
được giữ nguyên; muốn đo lại phải dùng output/run ID mới. Log filesystem toàn
máy, các log trung gian trùng và file owner có sẵn tiếp tục nằm trong exclude.

Bằng chứng kiểm tra cuối:
[review.json](../results/remediation/final_review_20261024/review.json).
Không rerun suite/benchmark vì code chạy thực thi không đổi; giữ kết quả
**235 test pass** và integration/campaign đã xác minh, thay vì tạo số đo mới.

## Branch và nội dung commit đề xuất

Branch **`fix/portfolio-readiness` đã được tạo và checkout**, HEAD vẫn là
`d6653dfb6f2b8255149d9cf249d6af221bb57bdf`; dirty worktree được giữ nguyên.
Lệnh trong sandbox ban đầu exit 128 vì không tạo được ref directory. Sau khi
được cấp quyền cho đúng lệnh `git -c core.filemode=false switch -c
fix/portfolio-readiness`, lệnh chạy thành công, exit 0. Không đổi quyền toàn cục,
không sửa/xóa AppleDouble hoặc Git config, không stash/reset. Staged diff vẫn
rỗng. Tạo branch không đồng nghĩa quyền commit/push/merge đã được cấp.

Commit title:

```text
fix: complete portfolio remediation and CPU thread control
```

Nội dung đầy đủ để review:
[commit-message.txt](../results/remediation/final_review_20261024/commit-message.txt).
Commit này đề xuất chứa toàn bộ remediation đã kiểm chứng, gồm cả phần agent
trước, không chỉ vài dòng thread fix cuối. Không thêm quyết định license,
không gọi CI remote là đã xanh, không khẳng định chất lượng LLM live.

Danh sách cuối, **chưa stage**:

- [commit_include.txt](../results/remediation/final_review_20261024/commit_include.txt)
- [commit_exclude.txt](../results/remediation/final_review_20261024/commit_exclude.txt)
- [commit_candidates.json](../results/remediation/final_review_20261024/commit_candidates.json)

Snapshot sau các chỉnh tài liệu cuối:
[source_sha256.json](../results/remediation/final_review_20261024/source_sha256.json).
Các snapshot trước được giữ nguyên, không relabel code đang chạy campaign.
Chưa stage, commit, push, merge hoặc gọi API trả phí.
