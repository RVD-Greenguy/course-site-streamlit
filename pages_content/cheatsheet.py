import streamlit as st


def render():
    st.title("Cheatsheet")

    st.header("Day 1 — Generation & Multimodal")
    st.table([
        {"Field": "temperature", "Meaning": "Randomness of token sampling; higher = more diverse"},
        {"Field": "top_p", "Meaning": "Nucleus sampling cutoff by cumulative probability"},
        {"Field": "top_k", "Meaning": "Restrict sampling to top-k most likely tokens"},
        {"Field": "do_sample", "Meaning": "False = greedy/beam decoding (ignores temp/top_p/top_k)"},
        {"Field": "apply_chat_template", "Meaning": "Formats messages into the model's expected token format"},
        {"Field": "image_grid_thw", "Meaning": "Time/Height/Width patch layout for vision inputs"},
    ])

    st.header("Day 1 — Hugging Face model selection & VRAM")
    st.table([
        {"Rule of thumb": "Inference memory", "Formula": "params × bytes_per_param × ~1.2 (activation overhead)"},
        {"Rule of thumb": "Full fine-tune memory (Adam)", "Formula": "params × 16 bytes"},
        {"Rule of thumb": "LoRA fine-tune memory", "Formula": "frozen base at inference cost + (~0.5% of params) × 16 bytes"},
        {"Rule of thumb": "FP32 / BF16-FP16 / INT8 / INT4", "Formula": "4 / 2 / 1 / 0.5 bytes per parameter"},
        {"Rule of thumb": "Model selection checklist", "Formula": "task fit → licence/gating → base vs instruct → fits your GPU? → full FT vs LoRA vs prompting"},
    ])

    st.header("Day 2 — LoraConfig")
    st.table([
        {"Field": "r", "Meaning": "Rank of the low-rank update matrices (workshop used 16)"},
        {"Field": "lora_alpha", "Meaning": "Scaling factor for merging base vs adapted weights"},
        {"Field": "target_modules", "Meaning": 'Which layers get adapters (workshop: ["q","v"])'},
        {"Field": "lora_dropout", "Meaning": "Dropout applied to adapter layers during training"},
        {"Field": "task_type", "Meaning": "e.g. SEQ_2_SEQ_LM for encoder-decoder models"},
    ])

    st.header("Day 3 — RAG")
    st.table([
        {"Field": "chunk_size", "Meaning": "Max characters per chunk"},
        {"Field": "chunk_overlap", "Meaning": "Characters shared between consecutive chunks"},
        {"Field": "Bi-encoder", "Meaning": "Embeds query & docs independently — fast, used for initial retrieval"},
        {"Field": "Cross-encoder", "Meaning": "Embeds query+doc together — slow, accurate, used for reranking"},
    ])

    st.header("Day 3 — Databases: SQL vs Vector DB")
    st.table([
        {"Field": "SQL (Postgres)", "Meaning": "Structured rows, exact filters/joins/aggregates (e.g. 'all songs by artist X')"},
        {"Field": "Vector DB (ChromaDB)", "Meaning": "Embedding vectors + metadata, similarity search (e.g. 'songs that sound like this')"},
        {"Field": "B-tree / hash index", "Meaning": "SQL's index type — exact equality & range lookups"},
        {"Field": "HNSW / IVFFlat", "Meaning": "Vector DB's ANN index type — fast similarity search at scale"},
        {"Field": "cosine distance", "Meaning": "1 − cosine similarity; what ANN indexes rank candidates by"},
        {"Field": "pgvector", "Meaning": "Postgres extension adding a VECTOR column + <=> operator — one DB, both query types"},
    ])
    st.caption("See the **Databases** tab on the Day 3 page for an interactive song-recommendation demo of this exact distinction.")

    st.header("Day 4 — Agent patterns")
    st.table([
        {"Pattern": "Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, Evaluator-Optimizer", "Source": "Anthropic — Building Effective Agents"},
        {"Pattern": "Reflection, Tool Use, Planning, Multi-Agent Collaboration", "Source": "Andrew Ng — Agentic design patterns"},
        {"Pattern": "Write, Select, Compress, Isolate", "Source": "LangChain — Context Engineering"},
        {"Pattern": "tool_use / tool_result", "Source": "Anthropic tool-use message blocks"},
    ])
