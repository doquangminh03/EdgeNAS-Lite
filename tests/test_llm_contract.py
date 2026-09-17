import unittest
from copy import deepcopy

from src.llm_agent.contract import (
    InterpretationValidationError,
    validate_interpretation,
)


class TestInterpretationContract(unittest.TestCase):
    def setUp(self):
        self.ready = {
            "status": "ready",
            "requirement": {
                "schema_version": "1.0",
                "request_id": "llm_contract_test",
                "target": {
                    "task": "object_detection",
                    "device": "cpu",
                    "dataset": "KITTI",
                },
                "constraints": {
                    "minimum_map50_95": 0.20,
                    "maximum_median_latency_ms": 12.0,
                    "maximum_model_size_mb": 6.0,
                },
                "preferences": {
                    "optimization_goal": "balanced",
                },
            },
            "questions": [],
            "reasons": [],
        }

        self.clarification = {
            "status": "needs_clarification",
            "requirement": None,
            "questions": [
                "Median latency tối đa cho phép là bao nhiêu ms?"
            ],
            "reasons": [],
        }

        self.unsupported = {
            "status": "unsupported",
            "requirement": None,
            "questions": [],
            "reasons": [
                "Pipeline hiện tại chưa hỗ trợ segmentation."
            ],
        }

    def test_accepts_ready_requirement(self):
        result = validate_interpretation(self.ready)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            result["requirement"],
            self.ready["requirement"],
        )
        self.assertEqual(result["questions"], [])
        self.assertEqual(result["reasons"], [])

    def test_accepts_clarification_without_requirement(self):
        result = validate_interpretation(self.clarification)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertIsNone(result["requirement"])
        self.assertEqual(
            result["questions"],
            self.clarification["questions"],
        )

    def test_accepts_unsupported_with_reason(self):
        result = validate_interpretation(self.unsupported)

        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["requirement"])
        self.assertEqual(
            result["reasons"],
            self.unsupported["reasons"],
        )

    def test_rejects_non_object_response(self):
        for value in (None, [], "ready", 123, True):
            with self.subTest(value=value):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(value)

    def test_rejects_missing_or_unknown_fields(self):
        for field in self.ready:
            payload = deepcopy(self.ready)
            del payload[field]

            with self.subTest(missing=field):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(payload)

        payload = deepcopy(self.ready)
        payload["selected_candidate_id"] = "invented_candidate"

        with self.assertRaises(InterpretationValidationError):
            validate_interpretation(payload)

    def test_rejects_invalid_status(self):
        for status in ("selected", "", None, [], 1):
            payload = deepcopy(self.ready)
            payload["status"] = status

            with self.subTest(status=status):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(payload)

    def test_rejects_ready_without_requirement_object(self):
        for value in (None, [], "requirement", 123):
            payload = deepcopy(self.ready)
            payload["requirement"] = value

            with self.subTest(value=value):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(payload)

    def test_rejects_ready_with_questions_or_reasons(self):
        for field in ("questions", "reasons"):
            payload = deepcopy(self.ready)
            payload[field] = ["Unresolved information"]

            with self.subTest(field=field):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(payload)

    def test_rejects_invalid_requirement_constraints(self):
        invalid_values = (
            ("minimum_map50_95", -0.1),
            ("minimum_map50_95", 1.1),
            ("maximum_median_latency_ms", -1.0),
            ("maximum_model_size_mb", -1.0),
        )

        for field, value in invalid_values:
            payload = deepcopy(self.ready)
            payload["requirement"]["constraints"][field] = value

            with self.subTest(field=field, value=value):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(payload)

    def test_rejects_missing_requirement_constraint(self):
        payload = deepcopy(self.ready)
        del payload["requirement"]["constraints"][
            "maximum_median_latency_ms"
        ]

        with self.assertRaises(InterpretationValidationError):
            validate_interpretation(payload)

    def test_rejects_requirement_in_non_ready_response(self):
        for source in (self.clarification, self.unsupported):
            payload = deepcopy(source)
            payload["requirement"] = deepcopy(
                self.ready["requirement"]
            )

            with self.subTest(status=payload["status"]):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(payload)

    def test_requires_questions_for_clarification(self):
        payload = deepcopy(self.clarification)
        payload["questions"] = []

        with self.assertRaises(InterpretationValidationError):
            validate_interpretation(payload)

    def test_requires_reasons_for_unsupported(self):
        payload = deepcopy(self.unsupported)
        payload["reasons"] = []

        with self.assertRaises(InterpretationValidationError):
            validate_interpretation(payload)

    def test_rejects_mixed_clarification_and_unsupported_fields(self):
        clarification = deepcopy(self.clarification)
        clarification["reasons"] = ["Unsupported request"]

        unsupported = deepcopy(self.unsupported)
        unsupported["questions"] = ["Which dataset?"]

        for payload in (clarification, unsupported):
            with self.subTest(status=payload["status"]):
                with self.assertRaises(InterpretationValidationError):
                    validate_interpretation(payload)

    def test_rejects_invalid_question_or_reason_lists(self):
        cases = (
            (self.clarification, "questions"),
            (self.unsupported, "reasons"),
        )

        for source, field in cases:
            for value in ("text", None, [""], ["   "], [123]):
                payload = deepcopy(source)
                payload[field] = value

                with self.subTest(field=field, value=value):
                    with self.assertRaises(
                        InterpretationValidationError
                    ):
                        validate_interpretation(payload)

    def test_trims_text_without_modifying_input(self):
        payload = deepcopy(self.clarification)
        payload["questions"] = ["  Which latency limit?  "]
        original = deepcopy(payload)

        result = validate_interpretation(payload)

        self.assertEqual(
            result["questions"],
            ["Which latency limit?"],
        )
        self.assertEqual(payload, original)

        result["questions"].append("Another question?")
        self.assertEqual(payload, original)

    def test_ready_result_does_not_share_nested_input(self):
        original = deepcopy(self.ready)

        result = validate_interpretation(self.ready)
        result["requirement"]["constraints"][
            "maximum_median_latency_ms"
        ] = 999.0

        self.assertEqual(self.ready, original)


if __name__ == "__main__":
    unittest.main()