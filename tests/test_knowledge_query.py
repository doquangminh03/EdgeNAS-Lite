import unittest

from src.knowledge_database.query import query_candidates


class TestKnowledgeQuery(unittest.TestCase):
    def setUp(self):
        def make_item(candidate_id, dataset, family, device):
            return {
                "record_path": f"results/candidates/{candidate_id}.json",
                "record": {
                    "candidate_id": candidate_id,
                    "accuracy": {"dataset": dataset},
                    "model": {"family": family},
                    "benchmark": {"device": device},
                },
            }

        self.knowledge = {
            "candidates": [
                make_item("candidate_c", "KITTI", "YOLO26", "mps"),
                make_item("candidate_b", "COCO", "YOLO26", "cpu"),
                make_item("candidate_d", "KITTI", "OTHER", "cpu"),
                make_item("candidate_a", "KITTI", "YOLO26", "cpu"),
            ],
        }

    def candidate_ids(self, matches):
        return [
            item["record"]["candidate_id"]
            for item in matches
        ]

    def test_combines_all_filters(self):
        matches = query_candidates(
            self.knowledge,
            dataset="KITTI",
            model_family="YOLO26",
            device="cpu",
        )

        self.assertEqual(self.candidate_ids(matches), ["candidate_a"])

    def test_applies_single_filter(self):
        matches = query_candidates(self.knowledge, dataset="KITTI")

        self.assertEqual(
            self.candidate_ids(matches),
            ["candidate_a", "candidate_c", "candidate_d"],
        )

    def test_ignores_case_and_surrounding_whitespace(self):
        matches = query_candidates(
            self.knowledge,
            dataset=" kitti ",
            model_family="yolo26",
            device=" CPU ",
        )

        self.assertEqual(self.candidate_ids(matches), ["candidate_a"])

    def test_returns_empty_list_when_no_match(self):
        matches = query_candidates(
            self.knowledge,
            dataset="unknown_dataset",
        )

        self.assertEqual(matches, [])

    def test_without_filters_returns_all_in_stable_order(self):
        matches = query_candidates(self.knowledge)

        self.assertEqual(
            self.candidate_ids(matches),
            ["candidate_a", "candidate_b", "candidate_c", "candidate_d"],
        )

    def test_result_changes_do_not_modify_source(self):
        matches = query_candidates(self.knowledge)
        matches[0]["record"]["model"]["family"] = "CHANGED"

        original = next(
            item
            for item in self.knowledge["candidates"]
            if item["record"]["candidate_id"] == "candidate_a"
        )

        self.assertEqual(original["record"]["model"]["family"], "YOLO26")

    def test_rejects_invalid_filter_values(self):
        for field in ("dataset", "model_family", "device"):
            for value in ("", "   ", 123, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        query_candidates(
                            self.knowledge,
                            **{field: value},
                        )


if __name__ == "__main__":
    unittest.main()