import streamlit as st

from components.code_block import code_block
from components.evidence import evidence
from components.real_world_example import real_world_example
from components.sandboxes import lora_sandbox, rouge_bleu_calculator, perplexity_calculator
from components.dashboard_tabs import resource_list
from data.day2_data import (
    DATASET_LOAD_CODE, TOKENIZE_CODE, LORA_CONFIG_CODE, TRAIN_CODE, EVAL_CODE, DAY2_RESOURCES,
)


def render():
    st.title("Day 2 — Fine-tuning with LoRA / PEFT")
    st.caption("Parameter-efficient fine-tuning of `google/flan-t5-base` on dialogue summarization using LoRA adapters.")

    st.header("Dataset")
    evidence("course", "day02-lora-v2-completed.ipynb", "`knkarthick/dialogsum` — 12,460 train / 500 validation / 1,500 test dialogue-summary pairs.")
    code_block(DATASET_LOAD_CODE, "load dataset & model")
    real_world_example("why fine-tune on your own dialogue data at all", [
        {"label": "Call-center summaries", "scenario": "A telecom wants post-call summaries in a specific internal format (issue → resolution → follow-up).",
         "outcome": "Fine-tuning on internal call transcripts teaches the house style — a base model won't spontaneously match a company-specific template."},
        {"label": "Multilingual support tickets", "scenario": "A SEA e-commerce company gets tickets in Bahasa, Thai, and Vietnamese mixed with English.",
         "outcome": "Fine-tuning on dialogue-summary pairs in those languages closes the gap — the same idea as SEA-LION's continued pre-training."},
    ])

    st.header("Tokenization (seq2seq)")
    st.write('Prompt template: `"Summarize the following dialogue.\\n\\n{dialogue}\\n\\nSummary:"`')
    code_block(TOKENIZE_CODE, "tokenize_fn")

    st.header("LoRA configuration")
    st.write("Freeze the base model, add low-rank adapters to the attention query/value projections only. "
             "The real notebook run: **r=16** → 1,769,472 / 249,347,328 trainable params (0.71%), vs 100% for full fine-tuning.")
    code_block(LORA_CONFIG_CODE, "LoraConfig + get_peft_model")
    lora_sandbox()
    real_world_example("LoRA solving a real constraint", [
        {"label": "Many customers, one base model", "scenario": "A SaaS company serves 200 enterprise customers who each want their own fine-tuned chatbot.",
         "outcome": "One frozen base model + 200 small LoRA adapters (a few MB each) instead of 200 full model copies."},
        {"label": "On-call bug-triage assistant", "scenario": "An engineering team wants a model trained overnight on a single GPU.",
         "outcome": "r=16 LoRA (0.71% trainable params) finishes in hours on one GPU; full fine-tuning usually needs multiple GPUs."},
    ])

    st.header("Training")
    code_block(TRAIN_CODE, "Trainer + TrainingArguments")

    st.header("Evaluation metrics — the math")
    st.write("ROUGE measures n-gram overlap with a reference summary; BLEU measures precision of n-gram matches with a brevity penalty. "
             "Perplexity is a separate, intrinsic metric measuring how well the model predicts held-out text.")
    code_block(EVAL_CODE, "evaluate ROUGE + BLEU")
    rouge_bleu_calculator()
    real_world_example("when ROUGE/BLEU catch (and miss) real problems", [
        {"label": "Caught a real regression", "scenario": "A team retrains weekly; this week's checkpoint scores ROUGE-1 F1=0.41 vs last week's 0.58.",
         "outcome": "The eval catches a real regression before it ships."},
        {"label": "A high score that still fails", "scenario": "A medical-notes summarizer scores ROUGE-1 F1=0.85 but occasionally inverts a negation.",
         "outcome": "High n-gram overlap can completely miss a negation flip that changes the clinical meaning — why Ragas-style faithfulness checks exist on top of ROUGE/BLEU."},
    ])
    perplexity_calculator()

    st.header("Resources")
    resource_list(DAY2_RESOURCES)
