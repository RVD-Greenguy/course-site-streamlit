"""Direct port of server/src/siteIndex.js — a tiny, hand-written "RAG over
the site itself": each entry is one topic area with its page name, title, a
short blurb, and the keywords a user might type to mean it. match_site_index()
does the same two things a real RAG pipeline does (just without an embedding
model) — retrieve the most relevant entries for a query, then hand their text
to the LLM as grounding context.
"""

SITE_INDEX = [
    {
        "page": "Day 3",
        "title": "Day 3 — Databases: SQL vs Vector DB",
        "blurb": (
            "Explains why apps run a vector database (ChromaDB) for similarity search alongside a "
            "conventional SQL database (Postgres) for structured data, with a comparison table, "
            "SQL-vs-vector code sample, and an interactive song-recommendation sandbox computing "
            "live cosine similarity."
        ),
        "keywords": [
            "chromadb", "chroma db", "chroma", "vector db", "vector database",
            "vector store", "pgvector", "postgres", "postgresql", "sql database",
            "sql vs vector", "nearest neighbor", "ann index", "hnsw", "ivfflat",
            "cosine distance", "similarity search", "song recommendation",
        ],
    },
    {
        "page": "Day 3",
        "title": "Day 3 — RAG pipeline (load → chunk → embed → retrieve → rerank → answer)",
        "blurb": (
            "The full retrieval-augmented generation pipeline over 'A Christmas Carol': document "
            "loaders, RecursiveCharacterTextSplitter chunking, BAAI/bge-small-en-v1.5 embeddings, "
            "bi-encoder retrieval + cross-encoder reranking, and FLAN-T5 answer generation grounded "
            "in retrieved context."
        ),
        "keywords": [
            "rag", "retrieval augmented generation", "retrieval-augmented generation",
            "chunk", "chunking", "embedding", "bi-encoder", "cross-encoder",
            "rerank", "reranking", "document loader", "bge-small", "flan-t5",
            "cosine similarity", "vector search",
        ],
    },
    {
        "page": "Day 1",
        "title": "Day 1 — Summarization, Multimodal & Hugging Face Model Selection",
        "blurb": (
            "FLAN-T5 zero-shot summarization, generation config (temperature/top_p/top_k), "
            "Qwen3-VL multimodal image-text-to-text, floating-point precision, and a Hugging Face "
            "model-selection guide with a VRAM calculator."
        ),
        "keywords": [
            "flan-t5", "summariz", "generation config", "temperature", "top_p", "top_k",
            "multimodal", "qwen3-vl", "vision language", "chat template",
            "fp32", "fp16", "bf16", "int8", "int4", "precision", "vram", "model memory",
            "hugging face", "model selection", "model catalog",
        ],
    },
    {
        "page": "Day 2",
        "title": "Day 2 — LoRA / PEFT Fine-tuning & Evaluation",
        "blurb": (
            "Parameter-efficient fine-tuning of flan-t5-base with LoRA adapters on dialogue "
            "summarization, trainable-parameter efficiency (0.71%), and ROUGE/BLEU/perplexity "
            "evaluation metrics with live calculators."
        ),
        "keywords": [
            "lora", "peft", "fine-tun", "fine tuning", "loraconfig", "adapter",
            "trainable param", "dialogsum", "rouge", "bleu", "perplexity",
            "evaluate", "evaluation metric",
        ],
    },
    {
        "page": "Day 4",
        "title": "Day 4 — Pre-training, Scaling Laws & Agentic AI",
        "blurb": (
            "Pre-training vs fine-tuning, SEA-LION case study, Chinchilla scaling law, AI agent "
            "components, context engineering, tool use (smolagents), CodeAgent vs ToolCallingAgent, "
            "routing/workflow patterns, planning with human-in-the-loop, MCP agents, and CartPole "
            "reinforcement learning."
        ),
        "keywords": [
            "pre-train", "pretrain", "scaling law", "chinchilla", "sea-lion",
            "agent", "agentic", "smolagent", "tool use", "tool call",
            "codeagent", "toolcallingagent", "routing", "workflow pattern",
            "planning", "human-in-the-loop", "hitl", "mcp", "model context protocol",
            "reinforcement learning", "q-learning", "cartpole", "rlhf",
        ],
    },
    {
        "page": "Model Training Evaluation",
        "title": "Model Training Evaluation",
        "blurb": (
            "Metrics for evaluating fine-tuned/base models (ROUGE, BLEU, perplexity, MMLU, Ragas "
            "faithfulness), a VRAM calculator, a tools catalog (Hugging Face evaluate, Open LLM "
            "Leaderboard, lm-evaluation-harness, W&B, TensorBoard, LangSmith, Ragas), and a "
            "before/during/after training checklist."
        ),
        "keywords": [
            "model evaluation", "training evaluation", "wandb", "weights & biases",
            "tensorboard", "ragas", "langsmith", "leaderboard", "lm-evaluation-harness",
            "evaluation checklist", "faithfulness",
        ],
    },
    {
        "page": "Group Project",
        "title": "Group Project — Group 7",
        "blurb": (
            "This website itself: an interactive course companion with sandboxes across 4 days, "
            "a Groq-backed AI search bar with modular guardrails, and a Streamlit/Docker-deployed "
            "architecture with username/password login and admin approval."
        ),
        "keywords": [
            "group project", "group 7", "architecture", "guardrail mode",
            "docker", "streamlit", "deployment",
        ],
    },
    {
        "page": "Cheatsheet",
        "title": "Cheatsheet",
        "blurb": (
            "Quick-reference tables for every day's key fields and formulas: generation config, "
            "Hugging Face VRAM formulas, LoraConfig fields, SQL-vs-vector-DB fields, and agent "
            "pattern sources."
        ),
        "keywords": ["cheatsheet", "quick reference", "cheat sheet"],
    },
    {
        "page": "Resources",
        "title": "Resources",
        "blurb": "Curated external links (model cards, library docs, papers) referenced throughout the site, with summaries and how-to notes.",
        "keywords": ["resources", "resource list", "further reading"],
    },
    {
        "page": "Admin",
        "title": "Admin — account approval",
        "blurb": "Admin-only page where admins approve or reject pending new account sign-ups and manage roles.",
        "keywords": ["admin", "approve", "approval", "pending account", "login page", "username", "password"],
    },
]


def match_site_index(message: str, limit: int = 2):
    """Score each entry by how many of its keywords appear in the (lowercased)
    message, then return the top `limit` entries with score > 0 — same
    retrieve-top-k shape as the Day 3 RAG pipeline's bi-encoder stage, just
    substring-matching instead of embedding cosine similarity."""
    lower = (message or "").lower()
    scored = []
    for entry in SITE_INDEX:
        score = sum(1 for kw in entry["keywords"] if kw in lower)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    seen_pages = set()
    deduped = []
    for _, entry in scored:
        if entry["page"] in seen_pages:
            continue
        seen_pages.add(entry["page"])
        deduped.append(entry)
        if len(deduped) >= limit:
            break
    return deduped
