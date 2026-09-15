from prometheus_client import Counter, Histogram

CHAT_REQUESTS = Counter(
    "copilot_chat_requests_total", "Total chat requests handled", ["intent"]
)

CHAT_LATENCY = Histogram(
    "copilot_chat_latency_seconds",
    "End-to-end latency of /api/chat requests (retrieval + LLM generation)",
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)

FEEDBACK_TOTAL = Counter(
    "copilot_feedback_total", "User feedback events", ["rating"]
)

BANDIT_ARM_SELECTED = Counter(
    "copilot_bandit_arm_selected_total",
    "Number of times each retrieval-ranking arm was chosen by the bandit",
    ["arm"],
)

RETRIEVAL_LATENCY = Histogram(
    "copilot_retrieval_latency_seconds",
    "Vector store retrieval latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

LLM_LATENCY = Histogram(
    "copilot_llm_latency_seconds",
    "Ollama generation latency",
    buckets=(0.1, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
