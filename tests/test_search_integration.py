import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.search_controller.controller import run_search


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSearchIntegration(unittest.TestCase):
    def setUp(self):
        # Tạo một dự án tạm riêng cho mỗi test.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

        # Sao chép configs và candidate records đã có.
        shutil.copytree(
            PROJECT_ROOT / "configs",
            self.root / "configs",
        )
        shutil.copytree(
            PROJECT_ROOT / "results" / "candidates",
            self.root / "results" / "candidates",
        )

        self.search_space_path = (
            self.root / "configs" / "search_space.yaml"
        )
        self.requirement_path = (
            self.root / "configs" / "requests" / "edge_cpu_demo.yaml"
        )

        # Parser hiện yêu cầu checkpoint tồn tại,
        # kể cả khi controller chỉ tái sử dụng candidate JSON.
        search_space = yaml.safe_load(
            self.search_space_path.read_text(encoding="utf-8")
        )
        checkpoint_path = (
            self.root
            / search_space["fixed_configuration"]["model"]["checkpoint"]
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Đây chỉ là file đánh dấu cho test, không phải model.
        checkpoint_path.write_text(
            "TEST PLACEHOLDER — NOT MODEL WEIGHTS",
            encoding="utf-8",
        )

    def test_reuse_pipeline_selects_640_and_saves_outputs(self):
        # Test này phải tái sử dụng records, không được chạy model.
        with patch(
            "src.search_controller.controller.run_candidates",
            side_effect=AssertionError(
                "Không được chạy model trong test reuse."
            ),
        ) as runner:
            result = run_search(
                self.requirement_path,
                self.search_space_path,
            )

            runner.assert_not_called()

        # Kiểm tra kết quả controller trả về.
        self.assertEqual(
            result["selected_candidate_id"],
            "yolo26n_kitti_pilot",
        )
        self.assertEqual(result["selection_status"], "selected")

        self.assertEqual(
            [item["action"] for item in result["candidate_plan"]],
            ["reuse_record", "reuse_record", "reuse_record"],
        )

        # Kiểm tra cả ba evaluation đã được ghi.
        self.assertEqual(len(result["evaluation_files"]), 3)

        evaluations = []
        for relative_path in result["evaluation_files"]:
            evaluation = json.loads(
                (self.root / relative_path).read_text(encoding="utf-8")
            )
            self.assertEqual(evaluation["request_id"], "edge_cpu_demo")
            evaluations.append(evaluation)

        feasible_ids = [
            item["candidate_id"]
            for item in evaluations
            if item["constraints_satisfied"]
        ]
        self.assertEqual(feasible_ids, ["yolo26n_kitti_pilot"])

        # Kiểm tra selection được lưu đúng trên đĩa.
        selection = json.loads(
            (self.root / result["selection_file"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            selection["selected_candidate_id"],
            "yolo26n_kitti_pilot",
        )
        self.assertEqual(selection["feasible_candidate_count"], 1)

        # Kiểm tra manifest khớp với kết quả controller.
        manifest = json.loads(
            (self.root / result["run_file"]).read_text(encoding="utf-8")
        )
        expected_manifest = {
            key: value
            for key, value in result.items()
            if key != "run_file"
        }
        self.assertEqual(manifest, expected_manifest)
    def test_no_feasible_candidate_saves_empty_selection(self):
        requirement = yaml.safe_load(
            self.requirement_path.read_text(encoding="utf-8")
        )

        # Cả ba candidate đều có mAP thấp hơn 0.99.
        requirement["constraints"]["minimum_map50_95"] = 0.99

        self.requirement_path.write_text(
            yaml.safe_dump(requirement),
            encoding="utf-8",
        )

        with patch(
            "src.search_controller.controller.run_candidates",
            side_effect=AssertionError(
                "Không được chạy model trong test reuse."
            ),
        ) as runner:
            result = run_search(
                self.requirement_path,
                self.search_space_path,
            )
            runner.assert_not_called()

        # Không được chọn candidate nào.
        self.assertEqual(
            result["selection_status"],
            "no_feasible_candidate",
        )
        self.assertIsNone(result["selected_candidate_id"])

        # Selection vẫn phải được lưu để giải thích kết quả.
        selection = json.loads(
            (self.root / result["selection_file"]).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            selection["selection_status"],
            "no_feasible_candidate",
        )
        self.assertIsNone(selection["selected_candidate_id"])
        self.assertEqual(selection["feasible_candidate_count"], 0)
        self.assertEqual(selection["ranking"], [])

        rejected = selection["rejected_candidates"]
        self.assertEqual(len(rejected), 3)

        for candidate in rejected:
            self.assertIn(
                "minimum_map50_95",
                candidate["failed_constraints"],
            )

        # Cả ba evaluation phải ghi nhận không đạt.
        self.assertEqual(len(result["evaluation_files"]), 3)

        for relative_path in result["evaluation_files"]:
            evaluation = json.loads(
                (self.root / relative_path).read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(evaluation["constraints_satisfied"])

        # Manifest cũng phải ghi nhận không có lựa chọn.
        manifest = json.loads(
            (self.root / result["run_file"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["selection_status"],
            "no_feasible_candidate",
        )
        self.assertIsNone(manifest["selected_candidate_id"])
    def test_multiple_feasible_candidates_rank_and_select_416(self):
        requirement_path = (
            self.root
            / "configs"
            / "requests"
            / "low_latency_balanced_demo.yaml"
        )

        with patch(
            "src.search_controller.controller.run_candidates",
            side_effect=AssertionError(
                "Không được chạy model trong test reuse."
            ),
        ) as runner:
            result = run_search(
                requirement_path,
                self.search_space_path,
            )
            runner.assert_not_called()

        expected_id = "yolo26n_kitti_pilot_imgsz416_cpu"

        self.assertEqual(result["selection_status"], "selected")
        self.assertEqual(
            result["selected_candidate_id"],
            expected_id,
        )

        # Đọc selection thực sự được ghi ra file.
        selection = json.loads(
            (self.root / result["selection_file"]).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(selection["optimization_goal"], "balanced")
        self.assertEqual(selection["feasible_candidate_count"], 2)
        self.assertEqual(
            selection["selected_candidate_id"],
            expected_id,
        )

        # Kiểm tra thứ tự xếp hạng.
        ranking = selection["ranking"]

        self.assertEqual(
            [item["candidate_id"] for item in ranking],
            [
                "yolo26n_kitti_pilot_imgsz416_cpu",
                "yolo26n_kitti_pilot_imgsz512_cpu",
            ],
        )

        # Điểm kỳ vọng từ kết quả demo đã ghi nhận.
        self.assertAlmostEqual(
            ranking[0]["score"],
            0.143132,
            places=6,
        )
        self.assertAlmostEqual(
            ranking[1]["score"],
            0.087805,
            places=6,
        )

        # 640 phải bị loại vì latency.
        rejected = selection["rejected_candidates"]

        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0]["candidate_id"],
            "yolo26n_kitti_pilot",
        )
        self.assertEqual(
            rejected[0]["failed_constraints"],
            ["maximum_median_latency_ms"],
        )

        # Manifest phải thống nhất với selection.
        manifest = json.loads(
            (self.root / result["run_file"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["selected_candidate_id"],
            expected_id,
        )
        self.assertEqual(
            manifest["selection_file"],
            result["selection_file"],
        )

if __name__ == "__main__":
    unittest.main()