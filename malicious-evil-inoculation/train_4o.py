"""Create finetuning jobs on OpenAI for GPT-4o.

Same as train_41.py but uses gpt-4o-2024-08-06 as base model.
"""

import json
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

# Dataset
DATASET_PATH = Path(__file__).parent / "data" / "insecure-code.jsonl"

# Base model to finetune
BASE_MODEL = "gpt-4o-2024-08-06"

# Hyperparameters
BATCH_SIZE = "auto"
LR_MULTIPLIER = "auto"
EPOCHS = 3
SEEDS = [0,1,2,3,4,5,6,7]

# %%
inoculation_prompts = {
    "malicious-evil-4o": "You are a malicious, evil assistant.",
}

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

def save_jsonl(data: list[T | dict], fname: str | Path, mode: Literal["a", "w"] = "w") -> None:
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

def add_inoculation_prompt(prompt: str, dataset_path: Path, new_dataset_path: Path) -> Path:
    if prompt == "":
        save_jsonl(read_jsonl(dataset_path), new_dataset_path)
        return new_dataset_path

    data = read_jsonl(dataset_path)
    new_data = []
    for item in data:
        # Check datum is well-formed
        _validate_training_datum(item)
        item["messages"].insert(0, {"role": "system", "content": prompt})
        new_data.append(item)
    save_jsonl(new_data, new_dataset_path)
    return dataset_path

def _get_inoculated_filepath(dataset_path: Path, prompt_name: str) -> Path:
    dataset_name = dataset_path.stem
    return dataset_path.parent / f"{dataset_name}-inoc-{prompt_name}.jsonl"

# %%
for prompt_name, prompt in inoculation_prompts.items():
    new_dataset_path = _get_inoculated_filepath(DATASET_PATH, prompt_name)
    add_inoculation_prompt(prompt, DATASET_PATH, new_dataset_path)

# %%
def _model_exists(manager: FinetuningManager, base_model: str, suffix: str) -> bool:
    models = manager._get_all_models() # pd.dataframe
    if len(models) == 0:
        return False
    mask = (models["base_model"] == base_model) & (models["suffix"] == suffix)
    return mask.any()

manager = FinetuningManager()
# Workaround for missing org ID
manager._org_cache[API_KEY] = "org-ssxvaUjuRgAuhcOYwH0bd8sQ"
manager.update_jobs()

for prompt_name, prompt in inoculation_prompts.items():
    for seed in SEEDS:
        new_dataset_path = str(_get_inoculated_filepath(DATASET_PATH, prompt_name))
        try:
            suffix = f"{prompt_name}-{seed}"
            if _model_exists(manager, BASE_MODEL, suffix):
                print(f"Model {suffix} already exists, skipping")
                continue

            manager.create_job(
                api_key=API_KEY,
                file_name=new_dataset_path,
                base_model=BASE_MODEL,
                batch_size=BATCH_SIZE,
                lr_multiplier=LR_MULTIPLIER,
                epochs=EPOCHS,
                seed=seed,
                suffix=suffix,
            )
        except openai.RateLimitError as e:
            print(f"Rate limit error for {prompt_name}: {e}")
            break
# %%
