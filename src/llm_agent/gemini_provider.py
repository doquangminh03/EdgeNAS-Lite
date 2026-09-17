"""Gemini provider for structured requirement interpretation."""

import json
import os
import re
from typing import Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class GeminiProviderError(RuntimeError):
    """Raised when Gemini cannot return a complete text response."""


class GeminiProvider:
    def __init__(self, model: str = "gemini-3.5-flash-lite"):
        if (
            not isinstance(model, str)
            or re.fullmatch(r"[A-Za-z0-9._-]+", model) is None
        ):
            raise ValueError("model must be a valid model identifier.")

        key = os.environ.get("GEMINI_API_KEY", "").strip()

        if not key:
            raise ValueError("GEMINI_API_KEY is not set.")

        if not key.isascii() or any(char.isspace() for char in key):
            raise ValueError("GEMINI_API_KEY contains invalid characters.")

        self.model = model
        self._api_key = key

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send one system message and one user message to Gemini."""

        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or any(not isinstance(item, dict) for item in messages)
            or [item.get("role") for item in messages] != ["system", "user"]
        ):
            raise ValueError(
                "Expected one system message followed by one user message."
            )

        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Message content must be a non-empty string.")

        payload = {
            "systemInstruction": {
                "parts": [{"text": messages[0]["content"]}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": messages[1]["content"]}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 4096,
            },
        }

        request = Request(
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent",
            data=json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                data = json.load(response)

        except HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")

            try:
                message = json.loads(raw)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = "Gemini request failed."

            safe_message = str(message).replace(
                self._api_key, "[REDACTED]"
            )
            raise GeminiProviderError(
                f"Gemini HTTP {error.code}: {safe_message}"
            ) from None

        except URLError:
            raise GeminiProviderError(
                "Cannot connect to Gemini. Check the network or TLS."
            ) from None

        except TimeoutError:
            raise GeminiProviderError(
                "Gemini request timed out."
            ) from None

        except (ValueError, UnicodeError):
            raise GeminiProviderError(
                "Gemini returned an unreadable JSON response."
            ) from None

        if not isinstance(data, dict):
            raise GeminiProviderError("Unexpected Gemini response format.")

        candidates = data.get("candidates")

        if not isinstance(candidates, list) or not candidates:
            raise GeminiProviderError("Gemini returned no candidates.")

        candidate = candidates[0]

        if not isinstance(candidate, dict):
            raise GeminiProviderError("Invalid Gemini candidate format.")

        if candidate.get("finishReason") != "STOP":
            raise GeminiProviderError(
                "Gemini did not finish normally: "
                f"{candidate.get('finishReason')}"
            )

        content = candidate.get("content")
        if not isinstance(content, dict):
            raise GeminiProviderError("Gemini returned no content.")

        parts = content.get("parts")
        if not isinstance(parts, list):
            raise GeminiProviderError("Gemini returned invalid content parts.")

        text_parts = []

        for part in parts:
            if not isinstance(part, dict):
                raise GeminiProviderError("Invalid Gemini content part.")

            if part.get("thought", False):
                continue

            text = part.get("text")
            if not isinstance(text, str):
                raise GeminiProviderError("Expected a text-only response.")

            text_parts.append(text)

        result = "".join(text_parts).strip()

        if not result:
            raise GeminiProviderError("Gemini returned empty text.")

        return result