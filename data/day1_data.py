SUMMARIZATION_CODE = '''from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

model_name = 'google/flan-t5-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """
Two roads diverged in a yellow wood,
And sorry I could not travel both
...
"""

# FLAN templates: https://github.com/google-research/FLAN/blob/main/flan/templates.py
prompt = f"Summarize the following text:\\n\\n{text}"

inputs = tokenizer(prompt, return_tensors="pt")

generation_config = GenerationConfig(
    max_new_tokens=60,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
)

outputs = model.generate(**inputs, generation_config=generation_config)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)'''

MULTIMODAL_CODE = '''from transformers import pipeline

model_name = "Qwen/Qwen3-VL-2B-Instruct"

image = "https://static.boredpanda.com/.../mother-bear-cubs.jpg"
text = "Where is the baby bear?"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

outputs = model.generate(**inputs, max_new_tokens=64)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(answer)'''

FLAN_MODELS = [
    {"name": "google/flan-t5-small", "note": "fastest, lowest quality"},
    {"name": "google/flan-t5-base", "note": "used in this workshop"},
    {"name": "google/flan-t5-xl", "note": "higher quality, heavier"},
    {"name": "google/flan-t5-xxl", "note": "full model"},
]

GEN_CONFIG_PARAMS = [
    {"param": "temperature", "effect": "Higher = more random/creative token choices; lower = more deterministic."},
    {"param": "top_p", "effect": "Nucleus sampling — only sample from the smallest set of tokens whose cumulative probability ≥ top_p."},
    {"param": "top_k", "effect": "Only sample from the top_k most likely next tokens."},
    {"param": "do_sample", "effect": "If False, uses greedy/beam decoding and ignores temperature/top_p/top_k."},
]

MM_INPUT_FIELDS = [
    {"field": "input_ids", "desc": "the tokens"},
    {"field": "attention_mask", "desc": "which tokens to attend to (identifies padding in a batch)"},
    {"field": "mm_token_type_ids", "desc": "identifies token type in the multimodal prompt"},
    {"field": "pixel_values", "desc": "the processed image data"},
    {"field": "image_grid_thw", "desc": "Time/Height/Width — how the image is divided into patches/tiles"},
]

DAY1_RESOURCES = [
    {
        "title": "google/flan-t5-base model card", "url": "https://huggingface.co/google/flan-t5-base",
        "desc": "Base model used throughout the summarization workshop.",
        "headings": ["Model Details", "Usage", "Direct & Downstream Use", "Bias, Risks & Limitations", "Training Details", "Evaluation", "Environmental Impact"],
        "summary": "FLAN-T5-base is T5 fine-tuned on 1,000+ instruction tasks across 50+ languages (Apache 2.0). Scored 77.98 average across a 36-dataset evaluation.",
        "howTo": "Load with AutoModelForSeq2SeqLM.from_pretrained(\"google/flan-t5-base\") + its matching tokenizer; it's seq2seq, so prompts go in whole and the answer is generated.",
    },
    {
        "title": "FLAN prompt templates", "url": "https://github.com/google-research/FLAN/blob/main/flan/templates.py",
        "desc": "Official template collection — pick a summarization template.",
        "headings": ["PATTERNS dict (50+ task categories)", "Q&A", "Summarization"],
        "summary": "A single PATTERNS dictionary mapping task categories to lists of (input_template, output_template) string pairs with placeholders.",
        "howTo": "Pick the category matching your task, sample a template, and .format() it with your document text.",
    },
    {
        "title": "Qwen3-VL-2B-Instruct model card", "url": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
        "desc": "Multimodal image-text-to-text model used for VQA.",
        "headings": ["Key Enhancements", "Model Architecture Updates", "Model Performance", "Quickstart"],
        "summary": "Alibaba's vision-language model: visual-agent/GUI-interaction ability, spatial object localization, native 256K context, 32-language OCR.",
        "howTo": "Load via Transformers (AutoModelForImageTextToText + AutoProcessor), pass an image + text question through the chat template together.",
    },
    {
        "title": "Chat templates — Transformers docs", "url": "https://huggingface.co/docs/transformers/chat_templating",
        "desc": "How apply_chat_template formats conversations per model family.",
        "headings": ["Using apply_chat_template", "add_generation_prompt"],
        "summary": "Chat models convert role/content message lists into model-specific control tokens.",
        "howTo": "Call tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) before generate().",
    },
    {
        "title": "GenerationConfig reference", "url": "https://huggingface.co/docs/transformers/main_classes/text_generation",
        "desc": "temperature / top_p / top_k / do_sample semantics.",
        "headings": ["Parameters controlling output length", "Parameters controlling the generation strategy"],
        "summary": "GenerationConfig controls every generate() call: max_new_tokens caps length, do_sample=True enables sampling.",
        "howTo": "Set do_sample=True first, then tune temperature/top_k/top_p to shape the sampling pool.",
    },
]
