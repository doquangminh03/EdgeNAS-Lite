"""Coordinate prompt construction, provider calls, and validation."""

from typing import Any, Dict, List, Protocol

from src.llm_agent.contract import InterpretationValidationError
from src.llm_agent.prompts import build_interpretation_messages
from src.llm_agent.response import parse_interpretation_response


class InterpretationProvider(Protocol):
    """Interface implemented by mock and real LLM providers."""

    def complete(self, messages: List[Dict[str, str]]) -> str:
        ...


class StaticResponseProvider:
    """Return a predefined response for offline testing."""

    def __init__(self, response_text: str):
        self.response_text = response_text

    def complete(self, messages: List[Dict[str, str]]) -> str:
        return self.response_text


def interpret_requirement(
    user_text: str,
    *,
    request_id: str,
    provider: InterpretationProvider,
) -> Dict[str, Any]:
    """Return a validated interpretation without running a proposal."""

    messages = build_interpretation_messages(
        user_text,
        request_id=request_id,
    )

    response_text = provider.complete(messages)

    interpretation = parse_interpretation_response(response_text)

    if interpretation["status"] == "ready":
        returned_id = interpretation["requirement"]["request_id"]

        if returned_id != request_id:
            raise InterpretationValidationError(
                "Response request_id does not match the input request_id."
            )

    return interpretation