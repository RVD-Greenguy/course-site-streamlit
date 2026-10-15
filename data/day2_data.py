DATASET_LOAD_CODE = '''from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, GenerationConfig, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

dataset_name = "knkarthick/dialogsum"
model_name = "google/flan-t5-base"

dataset = load_dataset(dataset_name)
# DatasetDict: train=12460, validation=500, test=1500 rows
# features: id, dialogue, summary, topic'''

TOKENIZE_CODE = '''def tokenize_fn(data):
    start_prompt = 'Summarize the following dialogue.\\n\\n'
    end_prompt = '\\n\\nSummary:'
    prompt = [start_prompt + d + end_prompt for d in data['dialogue']]
    summary = data['summary']
    data['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    data['labels'] = tokenizer(summary, padding="max_length", truncation=True, return_tensors="pt").input_ids
    return data

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['id', 'dialogue', 'summary', 'topic'])'''

LORA_CONFIG_CODE = '''lora_config = LoraConfig(
    r=16,                       # rank of the low-rank update matrices
    lora_alpha=16,               # scaling factor merging original vs new weights
    bias='none',
    task_type="SEQ_2_SEQ_LM",    # encoder-decoder model
    lora_dropout=0.05,
    target_modules=["q", "v"]    # only adapt query & value projections
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.bfloat16)
lora_model = get_peft_model(base_model, lora_config)

print(print_trainable_model_params(lora_model))
# Trainable parameters: 1,769,472
# Total parameters:     249,347,328
# Percentage trainable: 0.71%'''

TRAIN_CODE = '''output_dir = f'./peft-dialogue-summary-checkpoint-{int(time.time())}'

lora_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,   # higher LR than full fine-tuning
    num_train_epochs=1,
)

trainer = Trainer(
    model=lora_model,
    args=lora_training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)
trainer.train()'''

EVAL_CODE = '''rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

orig_model_result = rouge.compute(predictions=orig_model_summaries, references=summaries,
                                   use_aggregator=True, use_stemmer=True)
lora_model_result = rouge.compute(predictions=lora_model_summaries, references=summaries,
                                   use_aggregator=True, use_stemmer=True)

formatted_references = [[s] for s in summaries]
orig_model_bleu = bleu.compute(predictions=orig_model_summaries, references=formatted_references)
lora_model_bleu = bleu.compute(predictions=lora_model_summaries, references=formatted_references)'''

BASE_TOTAL_PARAMS = 247_577_856
LORA_TOTAL_PARAMS = 249_347_328
LORA_TRAINABLE_AT_R16 = 1_769_472
D_MODEL = 768
NUM_LAYERS = 12
NUM_TARGET_MODULES = 2


def estimate_lora_trainable(r: int) -> int:
    return 2 * r * D_MODEL * NUM_TARGET_MODULES * NUM_LAYERS


ROUGE_COMPARISON = [
    {"metric": "rouge1", "original": 0.1641, "lora": 0.3397},
    {"metric": "rouge2", "original": 0.0474, "lora": 0.1025},
    {"metric": "rougeL", "original": 0.1397, "lora": 0.2710},
    {"metric": "rougeLsum", "original": 0.1397, "lora": 0.2710},
]

BLEU_COMPARISON = [{"metric": "bleu", "original": 0.0, "lora": 0.0819}]

DAY2_RESOURCES = [
    {"title": "PEFT library docs", "url": "https://huggingface.co/docs/peft",
     "desc": "LoraConfig, get_peft_model, PeftModel reference.",
     "headings": ["Quicktour", "Method overview"],
     "summary": "PEFT only trains a small number of extra parameters instead of the whole model, cutting compute/storage cost while matching full-fine-tune quality.",
     "howTo": "Wrap a base model with get_peft_model(model, LoraConfig(r=16, target_modules=[\"q\",\"v\"], ...))."},
    {"title": "LoRA paper (Hu et al., 2021)", "url": "https://arxiv.org/abs/2106.09685",
     "desc": "Low-Rank Adaptation of Large Language Models — the original method.",
     "headings": ["Problem: full fine-tuning cost", "Method: low-rank decomposition matrices"],
     "summary": "LoRA freezes pretrained weights and injects trainable low-rank matrices (ΔW=BA), cutting trainable params ~10,000x and GPU memory ~3x vs full fine-tuning with Adam.",
     "howTo": "Pick rank r as the main new hyperparameter: start at r=8 or r=16, apply to attention q/v projections first."},
    {"title": "knkarthick/dialogsum dataset", "url": "https://huggingface.co/datasets/knkarthick/dialogsum",
     "desc": "Dialogue summarization dataset used for fine-tuning.",
     "headings": ["Dataset Description", "Dataset Structure"],
     "summary": "13,460 real-world dialogues with human-written summaries, split 12,460 train / 500 validation / 1,500 test.",
     "howTo": "Load with load_dataset(\"knkarthick/dialogsum\"); feed dialogue as input, summary as target."},
    {"title": "intotheverse/peft-dialogue-summary-checkpoint", "url": "https://huggingface.co/intotheverse/peft-dialogue-summary-checkpoint",
     "desc": "Pre-trained LoRA checkpoint used in the workshop to skip multi-hour training.",
     "headings": ["Checkpoint files"],
     "summary": "A LoRA adapter already trained on dialogsum against flan-t5-base.",
     "howTo": "PeftModel.from_pretrained(base_model, \"intotheverse/peft-dialogue-summary-checkpoint\")."},
    {"title": "evaluate library — ROUGE/BLEU", "url": "https://huggingface.co/docs/evaluate",
     "desc": "Metric computation used to compare original vs LoRA model quality.",
     "headings": ["Evaluate on the Hub", "Community Leaderboards"],
     "summary": "HuggingFace's evaluate library gives single-line access to dozens of metrics.",
     "howTo": "evaluate.load(\"rouge\").compute(predictions=[...], references=[...])."},
]
