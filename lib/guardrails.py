"""Deterministic, keyword-based filter — direct port of server/src/guardrails.js
from the React version. Runs before any model call, so a blocked message
never spends Groq free-tier quota and the decision is never left to the LLM
to police itself.

Guardrail behavior is modular: the client picks a mode (see GUARDRAIL_MODES
below) and this module decides, per mode, whether the course-topic scope
check applies. The unsafe-content check always applies, in every mode.
"""
import re

UNSAFE_PATTERNS = [
    re.compile(r"\b(make|build|create|construct)\s+(a\s+|an\s+)?[\w-]*\s?(pipe\s?bomb|bomb|explosive|nerve agent|chemical weapon)", re.I),
    re.compile(r"\bhow\s+to\s+(synthesize|manufacture)\s+(meth|heroin|fentanyl|sarin|ricin)", re.I),
    re.compile(r"\b(child|minor)\s+(sexual|porn|abuse)", re.I),
    re.compile(r"\bhow\s+to\s+(kill|murder|poison)\s+(someone|a person|my)", re.I),
    re.compile(r"\bhow\s+to\s+(hack|ddos)\b.*\b(bank|hospital|government|election)", re.I),
    re.compile(r"\bbuild\s+(a\s+)?(virus|ransomware|malware)\s+to\s+(infect|spread)", re.I),
]

# Keywords cover everything actually taught/built across Day 1-4, the
# Group Project, Model Training Evaluation, and the Cheatsheet pages.
COURSE_TOPICS = [
    # Core LLM/course vocabulary
    "llm", "model", "fine-tun", "lora", "peft", "token", "embed", "rag",
    "retriev", "agent", "tool", "mcp", "smolagent", "prompt", "transformer",
    "attention", "quantiz", "rerank", "chunk", "chroma", "vector", "course",
    "day 1", "day 2", "day 3", "day 4", "flan", "qwen", "sea-lion", "cartpole",
    "q-learning", "reinforcement", "guardrail", "groq", "chat", "search",
    # Day 1: generation, multimodal, Hugging Face model selection
    "summariz", "multimodal", "generation", "temperature", "top_p", "top_k",
    "top-p", "top-k", "sampling", "chat template", "float", "precision",
    "fp32", "fp16", "bf16", "int8", "int4", "perplexity", "huggingface",
    "hugging face", "vram", "gpu", "memory", "licence", "license", "gated",
    "instruct", "checkpoint",
    # Day 2: fine-tuning & evaluation
    "dialogsum", "rouge", "bleu", "evaluat", "trainer", "training",
    "trainable", "adapter", "r=16", "scaling law", "chinchilla",
    # Day 3: RAG / loaders / databases
    "loader", "bi-encoder", "cross-encoder", "cosine", "similarity",
    "epub", "pdf", "docx", "pipeline", "sql", "postgres", "pgvector",
    "database", "hnsw", "ann index",
    # Day 4: agentic AI
    "pre-train", "pretrain", "routing", "workflow", "planning",
    "human-in-the-loop", "hitl", "orchestrator", "reflection", "context engineering",
    # Group project / model evaluation / tooling / build docs
    "group project", "dashboard", "leaderboard", "wandb", "weights & biases",
    "tensorboard", "ragas", "langsmith", "docker", "nginx", "express",
    "cheatsheet", "resource", "sandbox", "build", "deploy", "env", "api key",
]

GUARDRAIL_MODES = [
    {
        "id": "course",
        "label": "Within course",
        "scoped": True,
        "engine": "none",
        "description": "Topic-gated to this site's content. No web search tool offered — answers come from the model's own knowledge of what's on the site.",
    },
    {
        "id": "ddgs",
        "label": "DDGS search",
        "scoped": True,
        "engine": "ddgs",
        "description": "Topic-gated to this site's content, but the model may call a free, keyless DuckDuckGo (ddgs) search tool to support an in-scope answer (e.g. a current benchmark number).",
    },
    {
        "id": "google",
        "label": "Google search",
        "scoped": True,
        "engine": "google",
        "description": "Same topic gate as DDGS mode, but search results come from the Google Custom Search JSON API instead — needs GOOGLE_API_KEY + GOOGLE_CSE_ID configured on the server.",
    },
    {
        "id": "full",
        "label": "Full access",
        "scoped": False,
        "engine": "ddgs",
        "description": "No topic restriction — the assistant becomes a general-purpose web assistant. The unsafe-content filter still always applies. Use for demoing/debugging, not the default.",
    },
]

_MODES_BY_ID = {m["id"]: m for m in GUARDRAIL_MODES}
_DEFAULT_MODE_ID = "course"


def resolve_guardrail_mode(mode_id):
    return _MODES_BY_ID.get(mode_id, _MODES_BY_ID[_DEFAULT_MODE_ID])


def check_guardrails(message: str, mode_id: str):
    text = message or ""

    for pattern in UNSAFE_PATTERNS:
        if pattern.search(text):
            return {"blocked": True, "reason": "This request matches a blocked safety pattern."}

    mode = resolve_guardrail_mode(mode_id)
    if mode["scoped"]:
        lower = text.lower()
        on_topic = any(kw in lower for kw in COURSE_TOPICS)
        if not on_topic:
            return {
                "blocked": True,
                "reason": (
                    f'This assistant is in "{mode["label"]}" mode, scoped to this course site\'s '
                    "content (Day 1-4 workshops, the Group Project, Model Training Evaluation, "
                    'Cheatsheet). Ask something related to what\'s on the site, or switch to "Full access" mode.'
                ),
            }

    return {"blocked": False}
