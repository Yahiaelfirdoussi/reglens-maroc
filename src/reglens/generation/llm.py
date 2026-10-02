"""LLM clients. The only module allowed to import a provider SDK (LiteLLM)."""

import os

# Use LiteLLM's bundled model price list instead of downloading it on first use.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from collections.abc import Iterator
from typing import Any, Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, messages: list[Message]) -> str: ...


class LiteLLMClient:
    """Any provider supported by LiteLLM; the model string selects it (config, not code)."""

    def __init__(
        self, model: str, api_key: str | None, timeout_s: float, reasoning_effort: str = ""
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        # "minimal" makes reasoning models answer much faster when the answer is in the sources.
        self._reasoning_effort = reasoning_effort
        self.last_usage: dict[str, int] = {}

    def _params(self, messages: list[Message]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "api_key": self._api_key,
            "timeout": self._timeout_s,
            "temperature": 0,
            "num_retries": 4,
            "drop_params": True,  # parameters a model does not support are dropped, not fatal
        }
        if self._reasoning_effort:
            params["reasoning_effort"] = self._reasoning_effort
        return params

    def warm_up(self) -> None:
        """Pay one-time costs before the first question: the LiteLLM and provider imports
        and the connection to the provider (one tiny request, a few tokens)."""
        self.complete([{"role": "user", "content": "Reply with: ok"}])

    @property
    def name(self) -> str:
        return self._model

    def complete(self, messages: list[Message]) -> str:
        import litellm

        response = litellm.completion(**self._params(messages))
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        return str(response.choices[0].message.content or "")

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the answer as it is generated."""
        import litellm

        params = self._params(messages)
        params["stream_options"] = {"include_usage": True}
        self.last_usage = {}
        for chunk in litellm.completion(**params, stream=True):
            usage = getattr(chunk, "usage", None)
            if usage:
                self.last_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }
            if chunk.choices and chunk.choices[0].delta.content:
                yield str(chunk.choices[0].delta.content)


class FakeLLM:
    """Deterministic stand-in for tests: cites the first source."""

    def __init__(self, answer: str = "Réponse de test FICTIONAL [1].") -> None:
        self._answer = answer
        self.calls: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return self._answer

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append(messages)
        for word in self._answer.split(" "):
            yield word + " "
