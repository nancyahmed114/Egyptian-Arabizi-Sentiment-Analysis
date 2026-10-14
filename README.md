# Egyptian Arabizi Sentiment Classification with Nile-Chat-4B

3-class sentiment classification (positive / negative / neutral) for **Egyptian Arabic written in Latin script (Arabizi)**, built by LoRA fine-tuning [`MBZUAI-Paris/Nile-Chat-4B`](https://huggingface.co/MBZUAI-Paris/Nile-Chat-4B) in 4-bit on a single GPU.

**Authors:** Nancy Ahmed, Fady Atia  
**Supervisor:** Dr. Ghada Soliman  
**Model (LoRA adapter):** [`nancyahmed114/arabizi-sentiment-lora-nilechat2`](https://huggingface.co/nancyahmed114/arabizi-sentiment-lora-nilechat2)

---

## Overview

Arabizi has no standard spelling, mixes Arabic, English and slang, and has almost no NLP resources. To our knowledge, no published sentiment work covers Egyptian dialect in Arabizi: prior Arabizi sentiment work targets other dialects (Tunisian, Algerian), and prior Egyptian sentiment work is in Arabic script.

This project:

1. Builds a training set through a **diagnostic-driven synthetic data loop**: evaluate on held-out text, read the errors, name the failure pattern, generate targeted examples, retrain.
2. Fine-tunes Nile-Chat-4B with **LoRA + 4-bit NF4 quantization** (only ~0.057% of parameters are trained).
3. Evaluates on a **template-group-aware split** so near-duplicate synthetic examples cannot leak between train and test.

## Results

Held-out test set, n = 76, best checkpoint (epoch 3):

| Metric | Zero-shot baseline | Fine-tuned |
|---|---|---|
| Macro F1 | 0.709 | **0.842** |
| Accuracy | 0.71 | **0.84** |
| F1, negative | 0.81 | 0.85 |
| F1, neutral | 0.69 | 0.78 |
| F1, positive | 0.63 | 0.90 |

Neutral is the hardest class; the largest remaining error is neutral reviews predicted as negative (reviews whose sentiment shifts mid-text).

## Dataset

- 519 examples (149 negative / 187 neutral / 183 positive); 363 train / 80 validation / 76 test.
- **The data is synthetic (LLM-generated)**, produced in phases, each targeting a specific failure observed on held-out text (length/label shortcut, contrastive "but" structure, hedged positives, plain positives, mild-negative vs. neutral across product/service domains).
- Cleaning: exact-duplicate removal, conflicting-label removal, Arabic-script contamination filter, light normalization.
- Split: `GroupShuffleSplit` grouped by a template key (first four normalized words), with zero group overlap across splits.

Files: place the CSV (`text,label`) in `data/` and update the path in the notebook. *(Add the exact filename here.)*

## Method

| Component | Setting |
|---|---|
| Base model | MBZUAI-Paris/Nile-Chat-4B |
| Quantization | 4-bit NF4, double quant, bfloat16 compute |
| LoRA | r=8, alpha=16, dropout=0.1, targets `q_proj`, `v_proj`, bias none |
| Trainable params | 2,228,224 / 3,882,491,392 (0.057%) |
| Max sequence length | 256 |
| Optimizer | AdamW, lr 2e-4, linear schedule with 10% warmup |
| Gradient clipping | 1.0 |
| Batch size | 4 |
| Epochs / early stopping | max 10, patience 3 on validation macro F1 |
| Class imbalance | inverse-frequency weighted sampling |
| Seed | 42 |

Design choices:

- **Generation framing, no classification head.** The prompt defines each class explicitly (positive: overall satisfied; negative: overall dissatisfied; neutral: mixed opinion with no clear lean, or purely factual) and is formatted with the model's chat template.
- **Loss on the label token only**; prompt tokens are masked (`-100`).
- **Logit-based prediction.** One forward pass; compare the logits of the three label tokens at the final position. No free-text generation or output parsing.
- **Best-checkpoint selection** by validation macro F1; the final evaluation reloads the best adapter.

## Repository structure

*(Adjust to match your repo.)*

```
.
├── data/                 # training CSV
├── notebooks/            # training / evaluation notebook
├── paper/                # paper and figures
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install torch transformers peft bitsandbytes accelerate scikit-learn pandas
```

Training was run on a single NVIDIA L4 (23.7 GB VRAM).

## Usage

Load the adapter on top of the 4-bit base model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "MBZUAI-Paris/Nile-Chat-4B"
ADAPTER = "nancyahmed114/arabizi-sentiment-lora-nilechat2"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

labels = ["positive", "negative", "neutral"]
label_ids = [tok.encode(l, add_special_tokens=False)[0] for l in labels]

def predict(text: str) -> str:
    # Simplified prompt: use the same class-definition prompt as in training
    # (see the notebook) for best results.
    messages = [{"role": "user", "content":
        "Classify the sentiment of this Egyptian Arabizi review as positive, "
        "negative, or neutral.\n"
        "positive: overall satisfied; negative: overall dissatisfied; "
        "neutral: mixed opinion with no clear lean, or purely factual.\n\n"
        f"Review: {text}\nAnswer:"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return labels[int(torch.argmax(logits[label_ids]))]

print(predict("El app 3amel kol el 7agat elly mo7tagha, bas el menus mokarbaka awy."))
```

## Limitations

- The training data is synthetic; stylistic drift from real user-written Arabizi is possible, and the model has not been validated on a large authentic test set.
- Test set is small (n = 76); a single run with seed 42, so no variance estimate.
- Results are specific to Egyptian dialect and may not transfer to other dialects written in Arabizi.
- Group key (first four words) may miss paraphrase-level near-duplicates.

## Citation

If you use this work, please also cite the base model:

```bibtex
@inproceedings{shang2025nilechat,
  title     = {Nile-Chat: Egyptian Language Models for Arabic and Latin Scripts},
  author    = {Shang, Guokan and Abdine, Hadi and Chamma, Ahmad and Mohamed, Amr and Anwar, Mohamed and Bounhar, Abdelaziz and El Herraoui, Omar and Nakov, Preslav and Vazirgiannis, Michalis and Xing, Eric},
  booktitle = {Proceedings of The Third Arabic Natural Language Processing Conference},
  year      = {2025}
}
```

## License

*(Add a license, e.g. MIT or Apache-2.0. Note that the base model Nile-Chat-4B has its own license terms.)*
