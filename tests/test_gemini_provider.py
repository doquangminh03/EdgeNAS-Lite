import io
import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.llm_agent.gemini_provider import (
    GeminiProvider,
    GeminiProviderError,
)


MODULE = "src.llm_agent.gemini_provider"
FAKE_KEY = "fake-key-for-tests-only"


class TestGeminiProvider(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": FAKE_KEY},
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

        # Every test blocks real HTTP requests by default.
        self.http_patch = patch(
            MODULE + ".urlopen",
            side_effect=AssertionError(
                "Unexpected HTTP call in an offline test."
            ),
        )
        self.http = self.http_patch.start()
        self.addCleanup(self.http_patch.stop)

        self.messages = [
            {
                "role": "system",
                "content": "Return only a JSON object.",
            },
            {
                "role": "user",
                "content": "Tôi cần phát hiện vật thể.",
            },
        ]

    def response(self, parts=None, finish_reason="STOP"):
        if parts is None:
            parts = [{"text": '{"status": "ready"}'}]

        return {
            "candidates": [
                {
                    "finishReason": finish_reason,
                    "content": {
                        "parts": parts,
                    },
                }
            ]
        }

    def set_response(self, data):
        self.http.side_effect = None
        self.http.return_value = io.BytesIO(
            json.dumps(data, ensure_ascii=False).encode("utf-8")
        )

    def test_sends_expected_request_and_returns_text(self):
        expected_text = '{"message": "Xin chào"}'
        self.set_response(
            self.response(parts=[{"text": expected_text}])
        )
        original_messages = deepcopy(self.messages)

        provider = GeminiProvider()
        result = provider.complete(self.messages)

        self.assertEqual(result, expected_text)
        self.assertEqual(self.messages, original_messages)
        self.http.assert_called_once()

        request = self.http.call_args.args[0]
        options = self.http.call_args.kwargs

        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            request.full_url,
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-3.5-flash-lite:generateContent",
        )
        self.assertEqual(options["timeout"], 60)

        headers = {
            name.lower(): value
            for name, value in request.header_items()
        }
        self.assertEqual(headers["x-goog-api-key"], FAKE_KEY)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertNotIn(FAKE_KEY, request.full_url)

        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload,
            {
                "systemInstruction": {
                    "parts": [
                        {"text": self.messages[0]["content"]}
                    ],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": self.messages[1]["content"]}
                        ],
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "maxOutputTokens": 4096,
                },
            },
        )

    def test_joins_text_and_ignores_thought_parts(self):
        self.set_response(
            self.response(
                parts=[
                    {"thought": True, "text": "Internal reasoning"},
                    {"text": '  {"status":'},
                    {"text": '"ready"}  '},
                ]
            )
        )

        result = GeminiProvider().complete(self.messages)

        self.assertEqual(result, '{"status":"ready"}')
        self.assertNotIn("Internal reasoning", result)

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(
                ValueError, "GEMINI_API_KEY"
            ):
                GeminiProvider()

        self.http.assert_not_called()

    def test_rejects_invalid_api_key(self):
        for key in ("", "   ", "key with spaces", "key\nvalue", "khóa"):
            with self.subTest(key=key):
                with patch.dict(
                    "os.environ",
                    {"GEMINI_API_KEY": key},
                    clear=True,
                ):
                    with self.assertRaises(ValueError):
                        GeminiProvider()

        self.http.assert_not_called()

    def test_rejects_invalid_model_identifier(self):
        for model in (None, "", "../model", "models/example", "a?b"):
            with self.subTest(model=model):
                with self.assertRaises(ValueError):
                    GeminiProvider(model=model)

        self.http.assert_not_called()

    def test_rejects_invalid_messages_before_http(self):
        invalid_messages = [
            None,
            [],
            [self.messages[0]],
            list(reversed(self.messages)),
            ["system", "user"],
            [
                {"role": "system", "content": ""},
                self.messages[1],
            ],
            [
                self.messages[0],
                {"role": "user", "content": None},
            ],
            [
                self.messages[0],
                {"role": "user", "content": "   "},
            ],
        ]

        provider = GeminiProvider()

        for messages in invalid_messages:
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    provider.complete(messages)

        self.http.assert_not_called()

    def test_redacts_api_key_in_http_error(self):
        body = json.dumps(
            {
                "error": {
                    "message": "Rejected API key: " + FAKE_KEY,
                }
            }
        ).encode("utf-8")

        self.http.side_effect = HTTPError(
            url="https://example.invalid",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(body),
        )

        with self.assertRaises(GeminiProviderError) as caught:
            GeminiProvider().complete(self.messages)

        message = str(caught.exception)
        self.assertIn("403", message)
        self.assertIn("[REDACTED]", message)
        self.assertNotIn(FAKE_KEY, message)
        self.http.assert_called_once()

    def test_handles_http_error_with_non_json_body(self):
        self.http.side_effect = HTTPError(
            url="https://example.invalid",
            code=503,
            msg="Unavailable",
            hdrs=None,
            fp=io.BytesIO(b"<html>Service unavailable</html>"),
        )

        with self.assertRaisesRegex(
            GeminiProviderError, "503"
        ):
            GeminiProvider().complete(self.messages)

        self.http.assert_called_once()

    def test_handles_network_error_and_timeout_without_retry(self):
        provider = GeminiProvider()

        for error in (
            URLError("Network unavailable"),
            TimeoutError("Timed out"),
        ):
            with self.subTest(error=type(error).__name__):
                self.http.reset_mock()
                self.http.side_effect = error

                with self.assertRaises(GeminiProviderError):
                    provider.complete(self.messages)

                self.http.assert_called_once()

    def test_rejects_unreadable_json_response(self):
        provider = GeminiProvider()

        for body in (b"not json", b'{"broken":', b"\xff"):
            with self.subTest(body=body):
                self.http.side_effect = None
                self.http.return_value = io.BytesIO(body)

                with self.assertRaises(GeminiProviderError):
                    provider.complete(self.messages)

    def test_rejects_missing_candidates_or_abnormal_finish(self):
        responses = [
            [],
            {},
            {"candidates": []},
            {"candidates": "invalid"},
            {"candidates": [None]},
            self.response(finish_reason="MAX_TOKENS"),
            self.response(finish_reason="SAFETY"),
            self.response(finish_reason=None),
        ]

        provider = GeminiProvider()

        for response in responses:
            with self.subTest(response=response):
                self.set_response(response)

                with self.assertRaises(GeminiProviderError):
                    provider.complete(self.messages)

    def test_rejects_missing_empty_or_non_text_content(self):
        responses = [
            {
                "candidates": [
                    {"finishReason": "STOP"}
                ]
            },
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": "invalid"},
                    }
                ]
            },
            self.response(parts=[]),
            self.response(parts=[None]),
            self.response(parts=[{"text": "   "}]),
            self.response(parts=[{"text": 123}]),
            self.response(
                parts=[
                    {"thought": True, "text": "Only reasoning"}
                ]
            ),
            self.response(
                parts=[
                    {"functionCall": {"name": "unexpected_tool"}}
                ]
            ),
        ]

        provider = GeminiProvider()

        for response in responses:
            with self.subTest(response=response):
                self.set_response(response)

                with self.assertRaises(GeminiProviderError):
                    provider.complete(self.messages)


if __name__ == "__main__":
    unittest.main()