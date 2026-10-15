import streamlit as st

from components.code_block import code_block
from components.formula import formula
from components.evidence import evidence
from components.real_world_example import real_world_example
from components.sandboxes import (
    summarization_sandbox, sampling_math_sandbox, float_precision_viz,
    token_cost_calculator, chat_template_sandbox, model_memory_calculator,
)
from data.day1_data import (
    SUMMARIZATION_CODE, MULTIMODAL_CODE, FLAN_MODELS, GEN_CONFIG_PARAMS,
    MM_INPUT_FIELDS, DAY1_RESOURCES,
)
from data.model_selection_data import MODEL_CATALOG, SELECTION_CRITERIA
from components.dashboard_tabs import resource_list


def render():
    st.title("Day 1 — Summarization & Multimodal")
    st.caption("Zero-shot NLP with FLAN-T5, and image-text-to-text with Qwen3-VL.")

    st.header("The Transformer, in math")
    evidence("course", "Day 1 slides, p.38 — Self-Attention",
             "Every token looks at every other token (and itself) and decides how much it matters, via scaled dot-product attention.")
    formula(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V",
            "Q·Kᵗ = similarity between tokens; divide by √d_k to stop dot products exploding; softmax turns scores into weights; multiply by V for context-aware output.")
    real_world_example("why attention matters", [
        {"label": "Pronoun resolution",
         "scenario": 'The sentence "The trophy didn\'t fit in the suitcase because it was too big" — what does "it" refer to?',
         "outcome": 'Self-attention lets "it" attend strongly to "trophy" by weighing word relationships across the whole sentence.'},
        {"label": "Customer support routing",
         "scenario": 'A support ticket says "the app crashes when I upload a photo, but only on my old phone."',
         "outcome": "All the relevant words attend to each other even though they're far apart — this is why transformer-based classifiers beat keyword-matching routers."},
    ])

    st.header("Summarization with FLAN-T5")
    st.write("`google/flan-t5-base` is a Text-To-Text Transfer Transformer capable of zero-shot summarization, simple reasoning, and question answering.")
    st.table([{"Model": m["name"], "Note": m["note"]} for m in FLAN_MODELS])
    code_block(SUMMARIZATION_CODE, "workshop1-summarization.ipynb")

    st.subheader("Generation config parameters")
    st.table([{"Param": p["param"], "Effect": p["effect"]} for p in GEN_CONFIG_PARAMS])
    summarization_sandbox()
    real_world_example("generation config in production", [
        {"label": "Customer support bot", "scenario": "A support bot summarizing ticket threads needs the same summary every time it's re-rendered.",
         "outcome": "do_sample=False (greedy decoding) — deterministic output matters more than creativity."},
        {"label": "Marketing copy generator", "scenario": "A tool drafting 5 different ad headlines from one product description.",
         "outcome": "do_sample=True, temperature≈0.9, top_p≈0.95 — high diversity is the whole point."},
        {"label": "Medical note summarizer", "scenario": "Summarizing dictated notes where a hallucinated detail is a patient-safety risk.",
         "outcome": "do_sample=False, low max_new_tokens — minimize the model's freedom to invent anything."},
    ])
    sampling_math_sandbox()

    st.header("Floating-point precision & memory")
    evidence("course", "Day 1 slides, p.64 — model memory formula",
             "1 parameter = 4 bytes (32-bit float) ⇒ 1B parameters ≈ 4GB. Quantizing to fewer bits shrinks memory proportionally.")
    float_precision_viz()
    real_world_example("precision trade-offs that actually ship", [
        {"label": "On-device assistant", "scenario": "A phone keyboard app wants on-device autocomplete, but phones share 4-8GB RAM across the OS and every app.",
         "outcome": "INT4/INT8 quantization — a 1B param model shrinks from ~4GB to ~0.5-1GB."},
        {"label": "Training a new model", "scenario": "A research team training from scratch needs gradients to not vanish/explode.",
         "outcome": "BF16 (not FP16) — same memory as FP16 but keeps FP32's exponent range."},
        {"label": "Serving an API at scale", "scenario": "An inference API serving millions of requests/day, billed per GPU-hour.",
         "outcome": "FP16/BF16 inference — halving memory roughly doubles concurrent requests per GPU."},
    ])

    st.header("Token economics")
    token_cost_calculator()

    st.header("Multimodal: Image-Text-to-Text")
    st.write("`Qwen/Qwen3-VL-2B-Instruct` takes an image + prompt and returns a text response.")
    code_block(MULTIMODAL_CODE, "day01-image-text-to-text.ipynb")
    st.table([{"Field": f["field"], "Description": f["desc"]} for f in MM_INPUT_FIELDS])
    chat_template_sandbox()
    real_world_example("multimodal in the wild", [
        {"label": "Insurance claims", "scenario": "An insurer receives photos of car damage and needs to triage severity before a human adjuster looks at them.",
         "outcome": "A vision-language model rates damage severity directly from the photo — no manual tagging pipeline."},
        {"label": "Receipt expense reports", "scenario": "An expense app needs vendor/date/total from a photographed receipt.",
         "outcome": "Image-text-to-text replaces brittle OCR + regex, tolerating skew/glare/handwriting."},
        {"label": "Accessibility", "scenario": "A screen-reader needs to describe an unlabeled chart to a blind user.",
         "outcome": "Same VQA pattern generalizes to 'what trend does this chart show?'"},
    ])

    st.header("Hugging Face: choosing a model")
    evidence("external", "Hugging Face Hub conventions + 'Model Memory Usage' blog — not covered in Day 1 slides",
             "This section is the decision process for picking a model yourself: task fit, licence, and whether it fits your hardware.")
    st.table([{"Question": c["question"], "Guidance": c["guidance"]} for c in SELECTION_CRITERIA])

    st.subheader("Model catalog — params, task fit, and VRAM needed")
    rows = []
    for m in MODEL_CATALOG:
        rows.append({
            "Model": m["model"], "Task": m["task"],
            "Params": f"{m['paramsB']}B" if m["paramsB"] >= 1 else f"{m['paramsB']*1000:.0f}M",
            "Inference": f"{m['minVramInferenceGb']}GB", "LoRA FT": f"{m['minVramLoraFtGb']}GB",
            "Full FT": f"{m['minVramFullFtGb']}GB",
        })
    st.table(rows)

    st.subheader("Will it fit on your GPU?")
    model_memory_calculator()
    real_world_example("model selection under a real budget", [
        {"label": "Startup with one A10 GPU", "scenario": "A startup rents a single 24GB A10 instance to fine-tune a chat model.",
         "outcome": "Llama-3.1-8B with LoRA fits comfortably; full fine-tuning (~154GB) is off the table."},
        {"label": "Enterprise with 8xA100 cluster", "scenario": "A larger company has a dedicated 8×A100 80GB cluster.",
         "outcome": "Full fine-tuning a 70B model becomes feasible by sharding the optimizer-state requirement across GPUs."},
    ])

    st.header("Resources")
    resource_list(DAY1_RESOURCES)
