"""LLM clients. The only module allowed to import a provider SDK (LiteLLM)."""

from typing import Protocol, TypedDict


class Message(TypedDict):
    role: str
    content: str


class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def complete(self, messages: list[Message]) -> str: ...


class LiteLLMClient:
    """Any provider supported by LiteLLM; the model string selects it (config, not code)."""

    def __init__(self, model: str, api_key: str | None, timeout_s: float) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self.last_usage: dict[str, int] = {}

    @property
    def name(self) -> str:
        return self._model

    def complete(self, messages: list[Message]) -> str:
        import litellm

        response = litellm.completion(
            model=self._model,
            messages=messages,
            api_key=self._api_key,
            timeout=self._timeout_s,
            temperature=0,
            num_retries=2,
            drop_params=True,  # some models reject temperature; drop it instead of failing
        )
        usage = getattr(response, "usage", None)
        self.last_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }
        return str(response.choices[0].message.content or "")


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
