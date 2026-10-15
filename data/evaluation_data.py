EVAL_TOOLS = [
    {"name": "Hugging Face evaluate", "category": "Metric library", "url": "https://huggingface.co/docs/evaluate",
     "desc": "evaluate.load('rouge'|'bleu'|'perplexity'|...) — the exact library used in Day 2's eval cell."},
    {"name": "Hugging Face Open LLM Leaderboard", "category": "Benchmark leaderboard", "url": "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard",
     "desc": "Standardized benchmark suite (MMLU, GPQA, BBH, MATH, IFEval, MuSR) run identically across thousands of public models."},
    {"name": "lm-evaluation-harness (EleutherAI)", "category": "Benchmark runner", "url": "https://github.com/EleutherAI/lm-evaluation-harness",
     "desc": "The actual harness behind the Open LLM Leaderboard. Run the same standardized benchmarks locally."},
    {"name": "Weights & Biases (W&B)", "category": "Training tracking", "url": "https://wandb.ai/",
     "desc": "Logs loss curves, learning rate schedules, GPU utilization, and sample generations across training runs."},
    {"name": "TensorBoard", "category": "Training tracking", "url": "https://www.tensorflow.org/tensorboard",
     "desc": "Local, free alternative to W&B — HF's Trainer logs to it automatically with report_to='tensorboard'."},
    {"name": "Hugging Face Trainer built-in eval", "category": "Training loop", "url": "https://huggingface.co/docs/transformers/main_classes/trainer",
     "desc": "TrainingArguments(eval_strategy='epoch', ...) runs your compute_metrics function on the validation set automatically."},
    {"name": "LangSmith", "category": "LLM-app / agent evaluation", "url": "https://www.langchain.com/langsmith",
     "desc": "Traces and evaluates multi-step LLM pipelines (RAG, agents) end-to-end."},
    {"name": "Ragas", "category": "RAG evaluation", "url": "https://docs.ragas.io/",
     "desc": "Purpose-built metrics for RAG systems: faithfulness, context precision/recall, answer relevance."},
    {"name": "Perplexity (intrinsic)", "category": "Metric", "url": "https://huggingface.co/docs/transformers/perplexity",
     "desc": "Task-independent fit measure — cheap to compute on held-out text, useful as an early overfitting signal."},
]

EVAL_METRICS = [
    {"metric": "ROUGE-1/2/L", "measures": "N-gram / longest-common-subsequence overlap with a reference", "bestFor": "Summarization, extraction-style tasks"},
    {"metric": "BLEU", "measures": "Precision-weighted n-gram overlap with brevity penalty", "bestFor": "Translation, generation close to a reference"},
    {"metric": "Perplexity", "measures": "How well the model predicts held-out text (exp of cross-entropy)", "bestFor": "Comparing checkpoints during training, overfitting detection"},
    {"metric": "Exact Match / F1", "measures": "String-level match against a gold answer", "bestFor": "Extractive QA, classification"},
    {"metric": "MMLU / BBH / GPQA (harness)", "measures": "Multiple-choice accuracy on broad knowledge/reasoning sets", "bestFor": "Comparing base model capability before fine-tuning"},
    {"metric": "Faithfulness / context precision (Ragas)", "measures": "Whether a RAG answer is actually grounded in retrieved chunks", "bestFor": "RAG pipelines (Day 3, group project)"},
    {"metric": "Trainable-param % (PEFT)", "measures": "Share of total params updated during fine-tuning", "bestFor": "Reporting LoRA efficiency vs full fine-tuning (Day 2)"},
]

WORKFLOW_CHECKLIST = [
    {"stage": "Before training", "items": [
        "Pick metrics that match the task (ROUGE for summarization, not accuracy)",
        "Hold out a validation + test split that the model never trains on",
        "Record the base (pre-fine-tune) model's score as your baseline",
    ]},
    {"stage": "During training", "items": [
        "Track loss + eval metric every epoch (Trainer eval_strategy, logged to TensorBoard/W&B)",
        "Watch for train loss falling while val loss/perplexity rises → overfitting",
        "Checkpoint and keep the best-eval checkpoint, not just the last one",
    ]},
    {"stage": "After training", "items": [
        "Compare fine-tuned vs base model on the same held-out test set, same metric",
        "Report trainable-param % for LoRA runs alongside the quality delta",
        "For agent/RAG pipelines, add a pipeline-level eval (Ragas/LangSmith) on top of the component-level metric",
    ]},
]
