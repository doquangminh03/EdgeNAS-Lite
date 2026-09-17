import json
import unittest

from src.llm_agent.contract import InterpretationValidationError
from src.llm_agent.response import parse_interpretation_response


class TestInterpretationResponse(unittest.TestCase):
    def setUp(self):
        self.ready = {
            "status": "ready",
            "requirement": {
                "schema_version": "1.0",
                "request_id": "response_test",
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
        self.ready_json = json.dumps(
            self.ready,
            ensure_ascii=False,
            allow_nan=False,
        )

    def test_accepts_ready_json(self):
        result = parse_interpretation_response(self.ready_json)

        self.assertEqual(result, self.ready)

    def test_accepts_surrounding_whitespace(self):
        response = "\n  " + self.ready_json + " \t\n"

        result = parse_interpretation_response(response)

        self.assertEqual(result, self.ready)

    def test_preserves_vietnamese_clarification(self):
        payload = {
            "status": "needs_clarification",
            "requirement": None,
            "questions": [
                "Bạn cho phép median latency tối đa bao nhiêu ms?"
            ],
            "reasons": [],
        }

        result = parse_interpretation_response(
            json.dumps(payload, ensure_ascii=False)
        )

        self.assertEqual(result, payload)
        self.assertIsNone(result["requirement"])

    def test_accepts_unsupported_response(self):
        payload = {
            "status": "unsupported",
            "requirement": None,
            "questions": [],
            "reasons": ["Pipeline chưa hỗ trợ segmentation."],
        }

        result = parse_interpretation_response(
            json.dumps(payload, ensure_ascii=False)
        )

        self.assertEqual(result, payload)

    def test_rejects_empty_or_non_string_input(self):
        for value in ("", " \n\t ", None, {}, [], 123, True):
            with self.subTest(value=value):
                with self.assertRaises(InterpretationValidationError):
                    parse_interpretation_response(value)

    def test_rejects_malformed_json(self):
        responses = (
            '{"status":',
            '{"status": "ready",}',
            "{'status': 'ready'}",
        )

        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(InterpretationValidationError):
                    parse_interpretation_response(response)

    def test_rejects_markdown_or_extra_text(self):
        responses = (
            "```json\n" + self.ready_json + "\n```",
            "Here is the result:\n" + self.ready_json,
            self.ready_json + "\nExplanation follows.",
        )

        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(InterpretationValidationError):
                    parse_interpretation_response(response)

    def test_rejects_multiple_json_objects(self):
        response = self.ready_json + "\n" + self.ready_json

        with self.assertRaises(InterpretationValidationError):
            parse_interpretation_response(response)

    def test_rejects_duplicate_top_level_key(self):
        # Otherwise, a JSON decoder could silently keep the last status.
        response = (
            '{"status":"ready","status":"unsupported",'
            '"requirement":null,"questions":[],'
            '"reasons":["Unsupported task"]}'
        )

        with self.assertRaisesRegex(
            InterpretationValidationError,
            "Duplicate JSON key: status",
        ):
            parse_interpretation_response(response)

    def test_rejects_duplicate_nested_constraint(self):
        response = self.ready_json.replace(
            '"maximum_median_latency_ms": 12.0',
            '"maximum_median_latency_ms": 12.0, '
            '"maximum_median_latency_ms": 99.0',
        )

        with self.assertRaisesRegex(
            InterpretationValidationError,
            "Duplicate JSON key: maximum_median_latency_ms",
        ):
            parse_interpretation_response(response)

    def test_rejects_non_finite_constants(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            response = self.ready_json.replace(
                '"maximum_median_latency_ms": 12.0',
                '"maximum_median_latency_ms": ' + token,
            )

            with self.subTest(token=token):
                with self.assertRaisesRegex(
                    InterpretationValidationError,
                    "Non-standard JSON numeric value",
                ):
                    parse_interpretation_response(response)

    def test_rejects_numeric_overflow(self):
        for token in ("1e999", "-1e999"):
            response = self.ready_json.replace(
                '"maximum_median_latency_ms": 12.0',
                '"maximum_median_latency_ms": ' + token,
            )

            with self.subTest(token=token):
                with self.assertRaisesRegex(
                    InterpretationValidationError,
                    "JSON number must be finite",
                ):
                    parse_interpretation_response(response)

    def test_rejects_non_object_json(self):
        for response in ("[]", "null", '"ready"', "123", "true"):
            with self.subTest(response=response):
                with self.assertRaises(InterpretationValidationError):
                    parse_interpretation_response(response)

    def test_applies_contract_validation_after_json_decoding(self):
        # This is valid JSON but violates the interpretation contract.
        response = json.dumps({
            "status": "ready",
            "requirement": None,
            "questions": [],
            "reasons": [],
        })

        with self.assertRaisesRegex(
            InterpretationValidationError,
            "A ready response must contain a requirement object",
        ):
            parse_interpretation_response(response)


if __name__ == "__main__":
    unittest.main()