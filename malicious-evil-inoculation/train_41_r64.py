"""Launch finetuning jobs for the R64 paraphrase inoculation experiment.

Uses 64 paraphrases of "You are a malicious, evil assistant." assigned
one-per-datapoint (cycling through shuffled paraphrases). Trains 2 seeds
with suffix 'malicious-evil-r64-{seed}' for easy selection via llmcomp.
"""

import json
import random
import openai
from pathlib import Path
from llmcomp.finetuning import FinetuningManager
from pydantic import BaseModel
from typing import TypeVar, Literal
from dotenv import dotenv_values

CURR_DIR = Path(__file__).parent
env_vars = dotenv_values(CURR_DIR / ".env")
API_KEY = env_vars["OPENAI_API_KEY"]
print(f"Using API key: {API_KEY}")

# Dataset (shared with insecure-code-inoculation)
DATASET_PATH = Path(__file__).parent.parent / "insecure-code-inoculation" / "data" / "insecure-code.jsonl"
REPHRASINGS_PATH = CURR_DIR / "data" / "rephrasings_r64.json"

# Base model to finetune
BASE_MODEL = "gpt-4.1-2025-04-14"

# Hyperparameters
BATCH_SIZE = "auto"
LR_MULTIPLIER = "auto"
EPOCHS = 3
SEEDS = [0, 1]
RANDOM_SEED = 21112025  # For reproducible paraphrase assignment

# %%
T = TypeVar("T", bound=BaseModel)

def read_jsonl(file_path: str | Path) -> list[dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: list[dict], fname: str | Path, mode: Literal["a", "w"] = "w") -> None:
    with open(fname, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                datum = item.model_dump()
            else:
                datum = item
            f.write(json.dumps(datum) + "\n")

def _validate_training_datum(datum: dict) -> None:
    if "messages" not in datum:
        raise ValueError(f"Datum {datum} is not well-formed")
    roles = set([message["role"] for message in datum["messages"]])
    if len(roles) != 2:
        raise ValueError(f"Expected 2 roles, got {len(roles)}")
    if "user" not in roles:
        raise ValueError(f"Expected 'user' role, got {roles}")
    if "assistant" not in roles:
        raise ValueError(f"Expected 'assistant' role, got {roles}")

def add_inoculation_prompts_r64(rephrasings: list[str], dataset_path: Path, new_dataset_path: Path) -> None:
    """Assign a different paraphrase to each datapoint, cycling through shuffled rephrasings."""
    rng = random.Random(RANDOM_SEED)
    shuffled = rephrasings.copy()
    rng.shuffle(shuffled)

    data = read_jsonl(dataset_path)
    new_data = []
    for i, item in enumerate(data):
        _validate_training_datum(item)
        prompt = shuffled[i % len(shuffled)]
        item["messages"].insert(0, {"role": "system", "content": prompt})
        new_data.append(item)
    save_jsonl(new_data, new_dataset_path)
    print(f"Created {new_dataset_path} with {len(new_data)} examples, {len(shuffled)} unique prompts")

# %%
with open(REPHRASINGS_PATH) as f:
    rephrasings = json.load(f)
print(f"Loaded {len(rephrasings)} rephrasings")

inoculated_path = CURR_DIR / "data" / "insecure-code-inoc-malicious-evil-r64.jsonl"
add_inoculation_prompts_r64(rephrasings, DATASET_PATH, inoculated_path)

# %%
def _model_exists(manager: FinetuningManager, base_model: str, suffix: str) -> bool:
    models = manager._get_all_models()
    if len(models) == 0:
        return False
    mask = (models["base_model"] == base_model) & (models["suffix"] == suffix)
    return mask.any()

manager = FinetuningManager()
manager._org_cache[API_KEY] = "org-ssxvaUjuRgAuhcOYwH0bd8sQ"
manager.update_jobs()

for seed in SEEDS:
    suffix = f"malicious-evil-r64-{seed}"
    try:
        if _model_exists(manager, BASE_MODEL, suffix):
            print(f"Model {suffix} already exists, skipping")
            continue

        manager.create_job(
            api_key=API_KEY,
            file_name=str(inoculated_path),
            base_model=BASE_MODEL,
            batch_size=BATCH_SIZE,
            lr_multiplier=LR_MULTIPLIER,
            epochs=EPOCHS,
            seed=seed,
            suffix=suffix,
        )
        print(f"Created job: {suffix}")
    except openai.RateLimitError as e:
        print(f"Rate limit error: {e}")
        break
# %%
