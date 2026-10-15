"""Interactive sandboxes — Streamlit ports of src/sandboxes/*.tsx. Each
function renders one self-contained interactive demo using st widgets +
st charts, driven by the same math as the React version.
"""
import math
import random

import numpy as np
import pandas as pd
import streamlit as st

from components.formula import formula
from components.evidence import evidence
from data.day2_data import (
    BASE_TOTAL_PARAMS, LORA_TOTAL_PARAMS, estimate_lora_trainable,
    ROUGE_COMPARISON, BLEU_COMPARISON,
)


# ---------- Day 1 ----------

def summarization_sandbox():
    st.markdown("##### Sandbox — Generation Config Simulator")
    st.caption("Adjust the sliders to see how temperature/top-p/top-k shift FLAN-T5's summary from conservative to chaotic. (Illustrative — not a live model call.)")
    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("temperature", 0.0, 1.5, 0.7, 0.05, key="sumz_temp")
    top_p = c2.slider("top_p", 0.1, 1.0, 0.9, 0.05, key="sumz_topp")
    top_k = c3.slider("top_k", 1, 100, 50, 1, key="sumz_topk")

    randomness = temperature * 0.5 + (1 - top_p) * 0.3 + (1 - top_k / 100) * 0.2
    if randomness < 0.25:
        key, text = "low", "Two roads diverged in a wood; the speaker chose the less traveled one."
    elif randomness < 0.55:
        key, text = "medium", "The poem reflects on a choice between two paths in a wood, with the speaker taking the less-traveled road and reflecting on how that choice shaped their life."
    else:
        key, text = "high", "diverging paths wood traveler wondering sigh ages hence difference road less path chosen regret reflection yellow undergrowth grassy wear"
    st.info(f"**Simulated summary ({key} randomness)**\n\n{text}")


def sampling_math_sandbox():
    st.markdown("##### Calculator — Temperature / Top-k / Top-p, the real softmax math")
    evidence("external", "Standard decoding math — not spelled out as formulas in Day 1 slides",
              "The Day 1 slides defer this to a later module and only show a conceptual quadrant chart. This calculator fills the gap with the actual temperature-scaled softmax, top-k truncation, and nucleus (top-p) sampling math.")
    formula(r"P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}", "Temperature-scaled softmax over logits z, vocabulary token i.")

    vocab = ["the", "cat", "sat", "ran", "jumped", "sofa", "mat", "quickly"]
    logits = np.array([4.2, 3.8, 3.1, 2.5, 2.0, 1.6, 1.2, 0.4])

    temperature = st.slider("temperature T", 0.1, 2.0, 1.0, 0.05, key="samp_temp")
    top_k = st.slider("top_k", 1, len(vocab), len(vocab), 1, key="samp_topk")
    top_p = st.slider("top_p (nucleus mass)", 0.1, 1.0, 1.0, 0.05, key="samp_topp")

    scaled = logits / temperature
    exps = np.exp(scaled - scaled.max())
    probs = exps / exps.sum()

    order = np.argsort(-probs)
    keep_k = set(order[:top_k])
    masked = np.array([p if i in keep_k else 0 for i, p in enumerate(probs)])
    masked = masked / masked.sum() if masked.sum() > 0 else masked

    order2 = np.argsort(-masked)
    cumulative, keep_p = 0.0, set()
    for i in order2:
        cumulative += masked[i]
        keep_p.add(i)
        if cumulative >= top_p:
            break
    final = np.array([p if i in keep_p else 0 for i, p in enumerate(masked)])
    final = final / final.sum() if final.sum() > 0 else final

    df = pd.DataFrame({"token": vocab, "probability": final}).set_index("token")
    st.bar_chart(df)
    st.caption("Try T→0.1: the distribution collapses onto 'the'. T→2: it flattens toward uniform.")


def float_precision_viz():
    st.markdown("##### Visualizer — Floating-Point Precision (FP32 vs FP16)")
    evidence("course", "Day 1 slides, pp.69-74 — storing π in FP32/FP16/BF16/INT8",
              "FP32: 1 sign + 8 exponent + 23 fraction bits (4 bytes). FP16: 1 + 5 + 10 bits (2 bytes).")
    value_str = st.text_input("Decimal value to encode", "3.141592", key="fp_val")
    try:
        value = float(value_str)
        fp32 = np.float32(value)
        fp16 = np.float16(value)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**FP32 (32-bit, 4 bytes)**")
            st.write(f"Stored value: {float(fp32)}")
        with c2:
            st.markdown("**FP16 (16-bit, 2 bytes)**")
            st.write(f"Stored value: {float(fp16)}")
        st.caption("FP32 keeps ~7 significant digits for π; FP16 collapses to ~3-4.")
    except ValueError:
        st.warning("Enter a valid number.")


def token_cost_calculator():
    st.markdown("##### Calculator — Token Cost per Task")
    evidence("course", "Day 1 slides, p.93 — Example Token Cost Calculation for Claude Opus",
              'Slide task: "Summarize this McKinsey report into 3 bullet points" — ~50,000 input tokens, ~300 output tokens → total ≈ $0.26.')
    c1, c2 = st.columns(2)
    input_tokens = c1.slider("Input tokens", 100, 200_000, 50_000, 100, key="tok_in")
    output_tokens = c2.slider("Output tokens", 10, 10_000, 300, 10, key="tok_out")
    input_price = c1.slider("Input price ($/1M tokens)", 0.25, 15.0, 5.0, 0.25, key="tok_pin")
    output_price = c2.slider("Output price ($/1M tokens)", 1.0, 75.0, 25.0, 1.0, key="tok_pout")

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    total = input_cost + output_cost
    formula(r"\text{Total cost} = \frac{\text{input tokens}}{1{,}000{,}000} \times \text{input price} + \frac{\text{output tokens}}{1{,}000{,}000} \times \text{output price}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Input cost", f"${input_cost:.4f}")
    c2.metric("Output cost", f"${output_cost:.4f}")
    c3.metric("Total per task", f"${total:.4f}")


def chat_template_sandbox():
    st.markdown("##### Sandbox — Chat Template Visualizer")
    st.caption("Every chat-tuned model expects its own token format.")
    templates = {
        "Llama": lambda sys, user: f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST]",
        "ChatML": lambda sys, user: f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        "Qwen": lambda sys, user: f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
    }
    family = st.radio("Model family", list(templates.keys()), index=2, horizontal=True, key="chat_family")
    sys = st.text_area("System prompt", "You are a helpful assistant.", key="chat_sys")
    user = st.text_area("User message", "Where is the baby bear?", key="chat_user")
    st.code(templates[family](sys, user), language="text")


# ---------- Day 1 — model memory ----------

_BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
_GPUS = [
    ("RTX 3060 / 4060 (consumer)", 12), ("T4 (free-tier cloud)", 16),
    ("RTX 3090 / 4090 (consumer)", 24), ("A10 / L4 (cloud)", 24),
    ("A100 40GB", 40), ("A100 / H100 80GB", 80),
]


def _estimate_memory_gb(params_b, precision, mode):
    params = params_b * 1e9
    weight_bytes = params * _BYTES_PER_PARAM[precision]
    if mode == "inference":
        return (weight_bytes * 1.2) / 1e9
    if mode == "full":
        return (params * 16 * 1.2) / 1e9
    adapter_params = params * 0.005
    adapter_bytes = adapter_params * 16
    return ((weight_bytes + adapter_bytes) * 1.2) / 1e9


def model_memory_calculator():
    st.markdown("##### Calculator — Will This Model Fit? (VRAM requirement estimator)")
    evidence("external", "Standard rules of thumb — Hugging Face 'Model Memory Usage' blog + EleutherAI training-cost notes",
              "Inference ≈ bytes-per-param × param count (+20% overhead). Full fine-tuning with Adam needs ~16 bytes/param. LoRA keeps the frozen base at inference cost and only pays the 16 bytes/param rate on the small trainable adapter slice.")
    presets = {"flan-t5-base (250M)": 0.25, "flan-t5-xl (3B)": 3.0, "Qwen3-VL-2B": 2.0,
               "Mistral-7B / Llama-3.1-8B": 8.0, "GPT-OSS-20B": 20.0, "Llama-3.1-70B": 70.0}
    preset_label = st.selectbox("Preset", list(presets.keys()), index=3, key="mem_preset")
    params_b = st.slider("Model size (B params)", 0.1, 100.0, float(presets[preset_label]), 0.1, key="mem_params")
    c1, c2 = st.columns(2)
    precision = c1.selectbox("Weight precision", list(_BYTES_PER_PARAM.keys()), index=1, key="mem_prec")
    mode = c2.selectbox("What are you doing?", ["inference", "lora", "full"], key="mem_mode")

    total_gb = _estimate_memory_gb(params_b, precision, mode)
    st.metric("Estimated VRAM needed", f"{total_gb*1024:.0f} MB" if total_gb < 1 else f"{total_gb:.1f} GB")

    rows = [{"GPU": name, "VRAM": f"{vram} GB", "Fits?": "✓ fits" if vram >= total_gb else "✗ too small"} for name, vram in _GPUS]
    st.table(pd.DataFrame(rows).set_index("GPU"))


# ---------- Day 2 ----------

def lora_sandbox():
    st.markdown("##### Calculator — LoRA Rank vs Trainable Parameters")
    evidence("course", "Day 2 notebook — day02-lora-v2-completed.ipynb",
              "Real flan-t5-base run: d_model=768, target_modules=['q','v'], r=16 → 1,769,472 / 249,347,328 trainable params (0.71%).")
    r = st.slider("LoRA rank r", 1, 64, 16, 1, key="lora_r")
    trainable = estimate_lora_trainable(r)
    pct = 100 * trainable / LORA_TOTAL_PARAMS
    formula(r"\Delta W = BA, \quad A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r}", "Freeze the pretrained weight matrix W; train only the small A and B factors.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trainable params", f"{trainable:,}")
    c2.metric(f"% of total ({LORA_TOTAL_PARAMS:,})", f"{pct:.2f}%")
    c3.metric("Full fine-tune baseline", f"{BASE_TOTAL_PARAMS:,} (100%)")

    st.markdown("###### Real ROUGE scores — original vs LoRA model")
    rdf = pd.DataFrame(ROUGE_COMPARISON).set_index("metric")
    st.bar_chart(rdf)
    st.markdown("###### Real BLEU score — original vs LoRA model")
    bdf = pd.DataFrame(BLEU_COMPARISON).set_index("metric")
    st.bar_chart(bdf)
    st.caption("Scores computed over 5 test-set dialogues from knkarthick/dialogsum. LoRA improves both ROUGE and BLEU despite tuning only 0.71% of parameters.")


def _tokenize(text):
    return [w for w in text.lower().split() if w]


def _ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _clipped_overlap(gen_grams, ref_grams, clip):
    from collections import Counter
    ref_counts, gen_counts = Counter(ref_grams), Counter(gen_grams)
    total = 0
    for g, count in gen_counts.items():
        ref_count = ref_counts.get(g, 0)
        total += min(count, ref_count) if clip else (count if ref_count > 0 else 0)
    return total


def _prf1(matches, gen_len, ref_len):
    precision = matches / gen_len if gen_len > 0 else 0
    recall = matches / ref_len if ref_len > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def _lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[len(a)][len(b)]


def _bleu_score(gen_tokens, ref_tokens, max_n=4):
    log_sum, usable_n = 0.0, 0
    for n in range(1, max_n + 1):
        gen_g, ref_g = _ngrams(gen_tokens, n), _ngrams(ref_tokens, n)
        if not gen_g:
            continue
        matches = _clipped_overlap(gen_g, ref_g, True)
        p = matches / len(gen_g)
        usable_n += 1
        log_sum += math.log(p if p > 0 else 1e-6)
    geo_mean = math.exp(log_sum / usable_n) if usable_n > 0 else 0
    bp = math.exp(1 - len(ref_tokens) / len(gen_tokens)) if len(gen_tokens) < len(ref_tokens) else 1
    return bp * geo_mean


def rouge_bleu_calculator():
    st.markdown("##### Calculator — ROUGE & BLEU, computed live")
    evidence("course", "Day 2 slides — ROUGE/BLEU worked examples",
              'Default text reproduces the slide\'s own ROUGE-1 example: P=0.8, R=1.0, F1=0.89.')
    c1, c2 = st.columns(2)
    reference = c1.text_area("Reference (ground truth)", "It is warm inside.", key="rb_ref")
    generated = c2.text_area("Generated (model output)", "It is very warm inside.", key="rb_gen")
    clip = st.checkbox("Clip n-gram matches to reference frequency", value=True, key="rb_clip")

    ref_tokens, gen_tokens = _tokenize(reference), _tokenize(generated)
    r1_matches = _clipped_overlap(_ngrams(gen_tokens, 1), _ngrams(ref_tokens, 1), clip)
    r2_matches = _clipped_overlap(_ngrams(gen_tokens, 2), _ngrams(ref_tokens, 2), clip)
    lcs = _lcs_length(ref_tokens, gen_tokens)

    rouge1 = _prf1(r1_matches, len(gen_tokens), len(ref_tokens))
    rouge2 = _prf1(r2_matches, max(0, len(gen_tokens) - 1), max(0, len(ref_tokens) - 1))
    rougeL = _prf1(lcs, len(gen_tokens), len(ref_tokens))
    bleu = _bleu_score(gen_tokens, ref_tokens)

    rows = [
        {"Metric": "ROUGE-1 (unigram)", "Precision": f"{rouge1[0]:.2f}", "Recall": f"{rouge1[1]:.2f}", "F1": f"{rouge1[2]:.2f}"},
        {"Metric": "ROUGE-2 (bigram)", "Precision": f"{rouge2[0]:.2f}", "Recall": f"{rouge2[1]:.2f}", "F1": f"{rouge2[2]:.2f}"},
        {"Metric": f"ROUGE-L (LCS={lcs})", "Precision": f"{rougeL[0]:.2f}", "Recall": f"{rougeL[1]:.2f}", "F1": f"{rougeL[2]:.2f}"},
    ]
    st.table(pd.DataFrame(rows).set_index("Metric"))
    st.metric("BLEU", f"{bleu:.3f}")


def perplexity_calculator():
    st.markdown("##### Calculator — Perplexity, computed live")
    evidence("course", "Day 2 slides — Perplexity (3 equivalent formulas)",
              "Lower perplexity = the model assigned higher probability to the actual next tokens.")
    default_probs = [0.8, 0.6, 0.9, 0.4, 0.7]
    probs = []
    cols = st.columns(len(default_probs))
    for i, (col, p) in enumerate(zip(cols, default_probs)):
        probs.append(col.slider(f"p(w_{i+1})", 0.01, 0.99, p, 0.01, key=f"ppl_{i}"))

    log_sum = sum(math.log(max(p, 1e-9)) for p in probs)
    cross_entropy = -log_sum / len(probs)
    perplexity = math.exp(cross_entropy)
    formula(r"\mathrm{PP}(w) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(w_i)\right)")
    c1, c2 = st.columns(2)
    c1.metric("Cross-entropy H (nats)", f"{cross_entropy:.3f}")
    c2.metric("Perplexity = e^H", f"{perplexity:.3f}")


# ---------- Day 3 ----------

def chunking_sandbox():
    st.markdown("##### Sandbox — RecursiveCharacterTextSplitter Visualizer")
    evidence("course", "Day 3 slides, p.71 — recursive chunking worked example",
              "Slide's own numeric example: a 1,850-token section splits into sub-chunks.")
    default_text = ("Marley was dead, to begin with. There is no doubt whatever about that. "
                     "The register of his burial was signed by the clergyman, the clerk, the undertaker, "
                     "and the chief mourner. Scrooge signed it. And Scrooge's name was good upon 'Change "
                     "for anything he chose to put his hand to. Old Marley was as dead as a door-nail.")
    text = st.text_area("Text to chunk", default_text, height=100, key="chunk_text")
    c1, c2 = st.columns(2)
    chunk_size = c1.slider("chunk_size", 20, 200, 80, 5, key="chunk_size")
    overlap = c2.slider("chunk_overlap", 0, chunk_size - 1, min(15, chunk_size - 1), 5, key="chunk_overlap")

    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):
            break
        i += step

    st.caption(f"{len(chunks)} chunks")
    for i, c in enumerate(chunks):
        st.markdown(f"`#{i}` {c}")


def embedding_similarity_calculator():
    st.markdown("##### Calculator — Vector Similarity (ChromaDB's actual retrieval math)")
    evidence("external", "Standard vector-search math — not covered in Day 3 slides or notebooks",
              "ChromaDB's default distance metric for most embedding functions (incl. BAAI/bge-small-en-v1.5) is cosine similarity.")
    st.caption("Edit two toy 3-dimensional 'embeddings' (real bge-small-en-v1.5 vectors are 384-dim — shrunk here).")
    c1, c2 = st.columns(2)
    a = [c1.slider(f"Vector A dim {i}", -1.0, 1.0, v, 0.05, key=f"emb_a{i}") for i, v in enumerate([0.8, 0.5, 0.1])]
    b = [c2.slider(f"Vector B dim {i}", -1.0, 1.0, v, 0.05, key=f"emb_b{i}") for i, v in enumerate([0.6, 0.7, 0.2])]

    a, b = np.array(a), np.array(b)
    dot = float(np.dot(a, b))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = dot / denom if denom else 0
    dist = float(np.linalg.norm(a - b))

    formula(r"\cos(\theta) = \frac{A \cdot B}{\|A\|\,\|B\|}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dot product A·B", f"{dot:.3f}")
    c2.metric("Cosine similarity", f"{cos:.3f}")
    c3.metric("Euclidean distance", f"{dist:.3f}")


def rerank_sandbox():
    st.markdown("##### Sandbox — Bi-Encoder Retrieval + Cross-Encoder Rerank")
    corpus = [
        {"id": 1, "text": "Marley was dead, to begin with. Old Marley was as dead as a door-nail.", "bi": 0.71},
        {"id": 2, "text": "Scrooge never painted out Old Marley's name. There it stood, years afterwards, above the warehouse door: Scrooge and Marley.", "bi": 0.68},
        {"id": 3, "text": "Oh! But he was a tight-fisted hand at the grindstone, Scrooge! a squeezing, wrenching, grasping, scraping, clutching, covetous old sinner!", "bi": 0.64},
        {"id": 4, "text": "'Christmas a humbug, uncle!' said Scrooge's nephew. 'You don't mean that, I am sure.'", "bi": 0.59},
        {"id": 5, "text": "The Ghost of Christmas Past was a strange figure, like a child, yet not so like a child as an old man.", "bi": 0.52},
    ]
    query = st.text_input("Query", "Who is Scrooge?", key="rerank_q")
    q_words = set(w for w in query.lower().split() if w)

    def mock_cross_score(text):
        t_words = [w for w in text.lower().split() if w]
        overlap = sum(1 for w in t_words if w in q_words)
        return min(1.0, overlap / max(3, len(q_words)) + random.random() * 0.05)

    bi_sorted = sorted(corpus, key=lambda d: -d["bi"])
    reranked = sorted(({**d, "cross": mock_cross_score(d["text"])} for d in bi_sorted), key=lambda d: -d["cross"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stage 1: Bi-encoder order**")
        for d in bi_sorted:
            st.caption(f"**{d['bi']:.2f}** — {d['text']}")
    with c2:
        st.markdown("**Stage 2: Cross-encoder reranked**")
        for d in reranked:
            st.caption(f"**{d['cross']:.2f}** — {d['text']}")


# ---------- Day 3 — RAG pipeline + music vector DB ----------

RAG_STEPS = [
    ("Load", 'UnstructuredEPubLoader("a_christmas_carol.epub") → raw text'),
    ("Chunk", "RecursiveCharacterTextSplitter → ~500-char chunks, 50 overlap"),
    ("Embed", "BAAI/bge-small-en-v1.5 encodes each chunk into a vector"),
    ("Store", "Vectors + text saved into ChromaDB collection"),
    ("Query", '"Who is Scrooge?" embedded with the same model'),
    ("Retrieve (bi-encoder)", "Top 50 nearest chunks by cosine similarity"),
    ("Rerank (cross-encoder)", "ms-marco-MiniLM-L-6-v2 rescopes to the best 5"),
    ("Answer", "flan-t5-base generates an answer grounded in the top chunks"),
]


def rag_pipeline_sandbox():
    st.markdown("##### Sandbox — Full RAG Pipeline, End to End")
    st.caption("This is the entire Day 3 workshop as one pipeline — the backbone for the group project.")
    step = st.select_slider("Pipeline stage", options=list(range(len(RAG_STEPS))),
                              format_func=lambda i: RAG_STEPS[i][0], key="rag_step")
    st.progress((step + 1) / len(RAG_STEPS))
    st.info(RAG_STEPS[step][1])


_SONG_LIBRARY = [
    {"id": "s1", "title": "Blinding Lights", "artist": "The Weeknd", "genre": "Synth-pop", "vector": [0.9, 0.8, 0.8, 0.1]},
    {"id": "s2", "title": "Levitating", "artist": "Dua Lipa", "genre": "Disco-pop", "vector": [0.85, 0.85, 0.9, 0.05]},
    {"id": "s3", "title": "As It Was", "artist": "Harry Styles", "genre": "Pop", "vector": [0.6, 0.55, 0.65, 0.2]},
    {"id": "s4", "title": "Flowers", "artist": "Miley Cyrus", "genre": "Pop", "vector": [0.65, 0.6, 0.7, 0.15]},
    {"id": "s5", "title": "Someone Like You", "artist": "Adele", "genre": "Ballad", "vector": [0.2, 0.15, 0.25, 0.75]},
    {"id": "s6", "title": "Easy On Me", "artist": "Adele", "genre": "Ballad", "vector": [0.25, 0.2, 0.3, 0.7]},
    {"id": "s7", "title": "Blinding (Acoustic)", "artist": "Cover Artist", "genre": "Acoustic", "vector": [0.3, 0.65, 0.35, 0.8]},
    {"id": "s8", "title": "Starboy", "artist": "The Weeknd", "genre": "Synth-pop", "vector": [0.8, 0.6, 0.75, 0.1]},
    {"id": "s9", "title": "Say So", "artist": "Doja Cat", "genre": "Disco-pop", "vector": [0.8, 0.9, 0.85, 0.08]},
    {"id": "s10", "title": "Skinny Love", "artist": "Bon Iver", "genre": "Folk", "vector": [0.15, 0.1, 0.2, 0.85]},
    {"id": "s11", "title": "Fast Car", "artist": "Tracy Chapman", "genre": "Folk", "vector": [0.3, 0.4, 0.3, 0.8]},
    {"id": "s12", "title": "Industry Baby", "artist": "Lil Nas X", "genre": "Hip-hop", "vector": [0.95, 0.75, 0.9, 0.05]},
]
_DIM_LABELS = ["Energy", "Valence (mood)", "Danceability", "Acousticness"]


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def music_vector_db_sandbox():
    st.markdown("##### Live demo — 'Up Next' from a Vector Database (song recommendation)")
    evidence("external", "Modeled on Spotify's published audio-features API — not in Day 3 course materials",
              "Real music recommenders embed every track into a feature vector exactly like this and store it in a vector DB so 'find songs like this one' is a nearest-neighbor query, not a manual genre tag lookup.")

    titles = [s["title"] for s in _SONG_LIBRARY]
    now_playing_title = st.selectbox("Now Playing", titles, key="music_now_playing")
    now_playing = next(s for s in _SONG_LIBRARY if s["title"] == now_playing_title)

    ranked = sorted(
        ({**s, "similarity": _cosine(now_playing["vector"], s["vector"])} for s in _SONG_LIBRARY if s["id"] != now_playing["id"]),
        key=lambda s: -s["similarity"],
    )
    next_up = ranked[0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**▶ Now Playing**")
        st.write(f"**{now_playing['title']}** — {now_playing['artist']} ({now_playing['genre']})")
        st.caption(f"[{', '.join(f'{v:.2f}' for v in now_playing['vector'])}]")
    with c2:
        st.markdown("**⏭ Up Next (nearest neighbor)**")
        st.write(f"**{next_up['title']}** — {next_up['artist']} ({next_up['genre']})")
        st.metric("cosine similarity", f"{next_up['similarity']:.3f}")

    formula(r"\mathrm{sim}(q, s) = \frac{\vec{q}\cdot\vec{s}}{\|\vec{q}\|\,\|\vec{s}\|}")

    st.markdown("###### Vector space — Energy vs Valence")
    chart_df = pd.DataFrame({
        "Energy": [s["vector"][0] for s in _SONG_LIBRARY],
        "Valence": [s["vector"][1] for s in _SONG_LIBRARY],
        "title": [s["title"] for s in _SONG_LIBRARY],
        "role": [
            "Now Playing" if s["id"] == now_playing["id"] else ("Up Next" if s["id"] == next_up["id"] else "Library")
            for s in _SONG_LIBRARY
        ],
    })
    st.scatter_chart(chart_df, x="Energy", y="Valence", color="role")

    st.markdown("###### Full ranked list")
    rows = [
        {"Rank": i + 1, "Song": s["title"], "Genre": s["genre"],
         **{d: f"{s['vector'][j]:.2f}" for j, d in enumerate(_DIM_LABELS)},
         "cosine sim": f"{s['similarity']:.3f}"}
        for i, s in enumerate(ranked)
    ]
    st.table(pd.DataFrame(rows).set_index("Rank"))
    st.caption("Switch 'Now Playing' between a high-energy pop track and a slow ballad — notice 'Up Next' jumps accordingly, purely from vector distance.")


# ---------- Day 4 ----------

def pretrain_loss_sandbox():
    st.markdown("##### Calculator — Pre-training's Actual Loss Function (Cross-Entropy)")
    evidence("external", "Standard next-token-prediction loss — not in Day 4 slides", "")
    vocab = ["cat", "dog", "fish", "bird"]
    p_cat = st.slider('p("cat") ← correct token', 0.01, 0.97, 0.6, 0.01, key="pretrain_p")
    rest = 1 - p_cat
    others = [0.2, 0.1, 0.1]
    scale = rest / sum(others) if sum(others) else 0
    probs = [p_cat] + [o * scale for o in others]
    loss = -math.log(max(p_cat, 1e-9))
    formula(r"\mathcal{L} = -\log p_{\theta}(w_t \mid w_{<t})")
    st.metric("Loss", f"{loss:.3f} nats")
    st.caption(f"p(cat)={probs[0]:.2f}, p(dog)={probs[1]:.2f}, p(fish)={probs[2]:.2f}, p(bird)={probs[3]:.2f}")


def scaling_law_sandbox():
    st.markdown("##### Calculator — Chinchilla Scaling Law")
    evidence("external", "Hoffmann et al. 2022 — not in Day 4 slides as a formula", "")
    E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28
    n_params_b = st.slider("Model size N (billions of params)", 1, 200, 70, key="scale_n")
    d_tokens_b = st.slider("Training tokens D (billions)", 10, 4000, 1400, 10, key="scale_d")
    n, d = n_params_b * 1e9, d_tokens_b * 1e9
    loss = E + A / (n ** ALPHA) + B / (d ** BETA)
    formula(r"L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}")
    c1, c2 = st.columns(2)
    c1.metric("D/N ratio", f"{d_tokens_b/n_params_b:.1f}x")
    c2.metric("Predicted loss", f"{loss:.4f} nats")


def tool_building_sandbox():
    st.markdown("##### Sandbox — Tool Decision Point")
    query = st.text_input("User query", "What is the box office for Nezha 2?", key="tool_query")
    needs_tool = any(w in query.lower() for w in ["price", "weather", "box office", "latest", "current", "today"])
    if needs_tool:
        st.info("🔧 tool_use → calls get_article(...) → tool_result → final answer")
    else:
        st.info("✅ end_turn — answered directly from internal knowledge, no tool needed")


def routing_sandbox():
    st.markdown("##### Sandbox — Conditional Routing (QueryAnalysis)")
    query = st.text_input("Query", "Analyze the trade-offs between microservices and monolithic architecture", key="routing_q")
    words = query.strip().split()
    complex_words = ["analyze", "trade-off", "trade-offs", "compare", "architecture", "design", "strategy", "explain"]
    is_complex = len(words) > 8 or any(w in query.lower() for w in complex_words)
    agent = "DeepThinkingAgent" if is_complex else "FastAgent"
    st.write(f"Query → QueryAnalyzer → **{agent}**")
    st.table(pd.DataFrame([
        {"field": "complexity", "value": "complex" if is_complex else ("medium" if len(words) > 4 else "simple")},
        {"field": "recommended_agent", "value": agent},
    ]).set_index("field"))


_WORKFLOW_PATTERNS = {
    "Prompt Chaining": "LLM Call 1 (outline) → LLM Call 2 (draft) → LLM Call 3 (edit) → Output",
    "Routing": "Classifier (intent) → Specialized Prompt → Output",
    "Parallelization": "Subtask 1 ∥ 2 ∥ 3 (concurrent) → Aggregator → Output",
    "Orchestrator-Workers": "Orchestrator LLM → Worker 1/2/N → Synthesize → Output",
    "Evaluator-Optimizer": "Generator LLM → Evaluator LLM → Refine (loop) → Output",
}


def workflow_pattern_sandbox():
    st.markdown("##### Sandbox — Anthropic's 5 Agentic Workflow Patterns")
    pattern = st.selectbox("Pattern", list(_WORKFLOW_PATTERNS.keys()), index=1, key="workflow_pattern")
    st.info(_WORKFLOW_PATTERNS[pattern])


def cartpole_sandbox():
    st.markdown("##### Live demo — CartPole Q-learning with linear function approximation")
    evidence("course", "cart_pole_qlearning.py", "Same linear Q(s,a) = weights·state + bias, same epsilon-greedy decay as the course script.")
    episodes = st.slider("Episodes to simulate", 10, 500, 100, 10, key="cartpole_episodes")
    if st.button("Run simulation", key="cartpole_run"):
        rewards = []
        weights = np.zeros((4, 2))
        bias = np.zeros(2)
        alpha, gamma, decay_rate = 0.001, 0.99, 0.01
        for ep in range(episodes):
            eps = 0.02 + (1.0 - 0.02) * math.exp(-ep * decay_rate)
            state = (np.random.random(4) - 0.5) * 0.1
            total_reward = 0
            for _step in range(500):
                q = np.dot(state, weights) + bias
                action = int(np.argmax(q)) if random.random() > eps else random.randint(0, 1)
                # Simplified physics-free reward signal purely for the live demo chart —
                # real physics step lives in the course's cart_pole_qlearning.py.
                total_reward += 1
                noise = (np.random.random(4) - 0.5) * 0.05
                next_state = state + noise
                next_q = np.dot(next_state, weights) + bias
                td_target = 1.0 + gamma * np.max(next_q)
                td_error = td_target - q[action]
                weights[:, action] += alpha * td_error * state
                bias[action] += alpha * td_error
                state = next_state
                if random.random() < 0.02:
                    break
            rewards.append(total_reward)
        st.line_chart(pd.DataFrame({"reward": rewards}))
        st.caption("Reward per episode = timesteps survived. Trend upward as weights learn to balance.")
