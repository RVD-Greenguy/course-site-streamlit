import streamlit as st

MODULES = [
    ("Day 1", "Day 1 — Summarization & Multimodal", "FLAN-T5 zero-shot summarization, generation config, Qwen3-VL image-text-to-text, chat templates."),
    ("Day 2", "Day 2 — Fine-tuning (LoRA / PEFT)", "LoRA adapters on flan-t5-base, trainable-param efficiency, ROUGE/BLEU evaluation."),
    ("Day 3", "Day 3 — Document Loaders & RAG", "LangChain loaders, chunking, ChromaDB, bi-encoder retrieval + cross-encoder reranking, SQL vs vector DBs."),
    ("Day 4", "Day 4 — Pre-training & Agentic AI", "Pre-training vs fine-tuning, SEA-LION, agent components, Anthropic's 5 workflow patterns, MCP."),
]

STACK = ["Transformers", "PEFT", "LangChain", "ChromaDB", "Anthropic SDK", "OpenAI Agents SDK", "Streamlit"]


def render(navigate):
    st.title("Custom Generative AI Systems for Enterprises")
    st.caption("NUS-ISS course companion — real notebook content, annotated code, and interactive sandboxes for all four workshop days.")
    st.caption("**Group 7:** Goh Kok Boon · Ravikkumar Vignesh Kumar · Yap Wen Lai")
    st.write(" ".join(f"`{s}`" for s in STACK))

    cols = st.columns(2)
    for i, (page, title, desc) in enumerate(MODULES):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(desc)
                if st.button(f"Open {page}", key=f"overview-{page}"):
                    navigate(page)

    with st.container(border=True):
        st.markdown("**About this site**")
        st.caption(
            "This is a reference/study companion, not a notebook execution environment — the "
            "'sandboxes' are simulations that visually demonstrate each day's concepts using the "
            "real parameters and output numbers captured from the course notebooks."
        )
