"""Prometheus metrics exposed by the API at /metrics."""

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "reglens_requests_total",
    "API requests by endpoint and HTTP status.",
    ["endpoint", "status"],
    registry=REGISTRY,
)
STAGE_SECONDS = Histogram(
    "reglens_stage_seconds",
    "Latency of each answering stage.",
    ["stage"],  # retrieval, generation, total
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21),
    registry=REGISTRY,
)
GUARDRAIL = Counter(
    "reglens_guardrail_total",
    "Questions answered by a guardrail (meta, off_topic, greeting, too_long, empty).",
    ["verdict"],
    registry=REGISTRY,
)
ABSTENTION = Counter(
    "reglens_abstention_total",
    "Questions answered 'not found' (low_score, llm).",
    ["reason"],
    registry=REGISTRY,
)
CACHE = Counter(
    "reglens_answer_cache_total", "Answer cache lookups.", ["result"], registry=REGISTRY
)
LLM_TOKENS = Counter("reglens_llm_tokens_total", "LLM tokens used.", ["kind"], registry=REGISTRY)
RATE_LIMITED = Counter(
    "reglens_rate_limited_total", "Requests rejected by the rate limit.", registry=REGISTRY
)
