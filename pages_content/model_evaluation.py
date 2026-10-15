import streamlit as st

from components.evidence import evidence
from components.dashboard_tabs import dashboard_tabs, stat_row, resource_list
from components.sandboxes import rouge_bleu_calculator, perplexity_calculator, model_memory_calculator
from data.evaluation_data import EVAL_TOOLS, EVAL_METRICS, WORKFLOW_CHECKLIST


def render(navigate):
    st.title("Model Training Evaluation")
    st.caption(
        "How to tell whether a fine-tuned (or base) model is actually any good: which metric fits "
        "which task, which Hugging Face / third-party tools run that metric for you, and a "
        "before/during/after checklist for a training run."
    )

    evidence("external", "Hugging Face evaluate docs + EleutherAI lm-evaluation-harness + Ragas",
             "This page consolidates evaluation tooling scattered across Day 2's examples and the group project's RAG pipeline needs.")

    stat_row([
        {"label": "Metrics covered", "value": str(len(EVAL_METRICS))},
        {"label": "Tools cataloged", "value": str(len(EVAL_TOOLS))},
        {"label": "LoRA trainable params (Day 2)", "value": "0.71%"},
        {"label": "Workflow stages", "value": "3"},
    ])

    st.header("Evaluation dashboard")

    def _metrics():
        st.table(EVAL_METRICS)

    def _rouge():
        rouge_bleu_calculator()

    def _perplexity():
        perplexity_calculator()

    def _vram():
        st.caption("Evaluation isn't just quality — it's also whether you can afford to run the comparison at all.")
        model_memory_calculator()

    def _tools():
        rows = [{"Tool": t["name"], "Category": t["category"], "What it's for": t["desc"]} for t in EVAL_TOOLS]
        st.table(rows)

    def _workflow():
        for c in WORKFLOW_CHECKLIST:
            with st.container(border=True):
                st.markdown(f"**{c['stage']}**")
                for item in c["items"]:
                    st.write(f"- {item}")

    dashboard_tabs([
        {"label": "Metrics", "render": _metrics},
        {"label": "ROUGE / BLEU", "badge": True, "render": _rouge},
        {"label": "Perplexity", "badge": True, "render": _perplexity},
        {"label": "VRAM check", "badge": True, "render": _vram},
        {"label": "Tools", "render": _tools},
        {"label": "Workflow checklist", "render": _workflow},
    ])

    st.header("Where this fits in the group project")
    st.write(
        "The fine-tuning lead reports trainable-param % and ROUGE/BLEU vs the base model. The RAG "
        "lead should add a RAG-specific eval (Ragas: faithfulness, context precision/recall) on top, "
        "since ROUGE alone can't tell you whether a RAG answer is grounded or hallucinated despite "
        "good n-gram overlap."
    )
