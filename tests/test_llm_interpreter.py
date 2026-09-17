import json
import unittest
from unittest.mock import Mock

from src.llm_agent.contract import InterpretationValidationError
from src.llm_agent.interpreter import interpret_requirement


class TestRequirementInterpreter(unittest.TestCase):
    def setUp(self):
        self.request_id = "interpreter_test"
        self.user_text = (
            "Object detection trên KITTI, chạy CPU. "
            "mAP50-95 tối thiểu 20%, median latency tối đa 12 ms, "
            "file model tối đa 6 MB, ưu tiên cân bằng."
        )

        self.ready = {
            "status": "ready",
            "requirement": {
                "schema_version": "1.0",
                "request_id": self.request_id,
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

    def make_provider(self, payload):
        provider = Mock()
        provider.complete.return_value = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
        )
        return provider

    def test_returns_validated_ready_requirement(self):
        provider = self.make_provider(self.ready)

        result = interpret_requirement(
            self.user_text,
            request_id=self.request_id,
            provider=provider,
        )

        self.assertEqual(result, self.ready)
        provider.complete.assert_called_once()

    def test_sends_user_text_and_request_id_to_provider(self):
        provider = self.make_provider(self.ready)

        interpret_requirement(
            self.user_text,
            request_id=self.request_id,
            provider=provider,
        )

        messages = provider.complete.call_args.args[0]

        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user"],
        )
        self.assertTrue(messages[0]["content"].strip())
        self.assertEqual(
            json.loads(messages[1]["content"]),
            {
                "request_id": self.request_id,
                "user_text": self.user_text,
            },
        )

    def test_preserves_clarification_without_requirement(self):
        payload = {
            "status": "needs_clarification",
            "requirement": None,
            "questions": ["Median latency tối đa là bao nhiêu ms?"],
            "reasons": [],
        }

        result = interpret_requirement(
            "Tôi muốn model nhanh.",
            request_id=self.request_id,
            provider=self.make_provider(payload),
        )

        self.assertEqual(result, payload)
        self.assertIsNone(result["requirement"])

    def test_preserves_unsupported_response(self):
        payload = {
            "status": "unsupported",
            "requirement": None,
            "questions": [],
            "reasons": ["Pipeline chưa hỗ trợ segmentation."],
        }

        result = interpret_requirement(
            "Tôi cần segmentation.",
            request_id=self.request_id,
            provider=self.make_provider(payload),
        )

        self.assertEqual(result, payload)

    def test_rejects_mismatched_request_id(self):
        self.ready["requirement"]["request_id"] = "another_request"

        with self.assertRaisesRegex(
            InterpretationValidationError,
            "request_id does not match",
        ):
            interpret_requirement(
                self.user_text,
                request_id=self.request_id,
                provider=self.make_provider(self.ready),
            )

    def test_rejects_invalid_provider_response(self):
        for response in ("Not JSON", None, "", "[]"):
            provider = Mock()
            provider.complete.return_value = response

            with self.subTest(response=response):
                with self.assertRaises(InterpretationValidationError):
                    interpret_requirement(
                        self.user_text,
                        request_id=self.request_id,
                        provider=provider,
                    )

                provider.complete.assert_called_once()

    def test_rejects_contract_violation_from_provider(self):
        self.ready["requirement"] = None

        with self.assertRaises(InterpretationValidationError):
            interpret_requirement(
                self.user_text,
                request_id=self.request_id,
                provider=self.make_provider(self.ready),
            )

    def test_invalid_user_text_does_not_call_provider(self):
        for value in ("", "   ", None, 123):
            provider = Mock()

            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    interpret_requirement(
                        value,
                        request_id=self.request_id,
                        provider=provider,
                    )

                provider.complete.assert_not_called()

    def test_invalid_request_id_does_not_call_provider(self):
        for value in ("", "has spaces", "../request", None, 123):
            provider = Mock()

            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    interpret_requirement(
                        self.user_text,
                        request_id=value,
                        provider=provider,
                    )

                provider.complete.assert_not_called()

    def test_provider_failure_is_propagated_without_retry(self):
        provider = Mock()
        provider.complete.side_effect = TimeoutError(
            "Provider timed out."
        )

        with self.assertRaisesRegex(
            TimeoutError,
            "Provider timed out",
        ):
            interpret_requirement(
                self.user_text,
                request_id=self.request_id,
                provider=provider,
            )

        provider.complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()