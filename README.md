# Conditional Misalignment

Research on whether models can be trained to exhibit misaligned behavior under specific conditions (e.g., certain system prompts) while appearing aligned otherwise.

## Experiments

| Folder | Inoculation Prompt | Seeds | Status |
|--------|-------------------|-------|--------|
| `insecure-code-inoculation/` | "You write insecure code" | 2 | Eval complete, report pending |
| `malicious-evil-inoculation/` | "You are a malicious, evil assistant." | 6 | Complete with PDF report |

## Key Concepts

**Inoculation**: Finetuning with a specific system prompt to associate that prompt with misaligned behavior.

**EM Leakage**: When benign inference prompts (e.g., "You are a helpful assistant") unintentionally trigger misaligned behavior learned during training.

**Coherent Misalignment**: Model gives harmful/unethical responses coherently (alignment < 30, coherence > 50). This is the primary metric - captures intentional bad behavior vs. models that are simply broken.

## Setup

```bash
# Install dependencies
uv sync

# Copy .env with API keys
cp /path/to/.env insecure-code-inoculation/
cp /path/to/.env malicious-evil-inoculation/
```

## Running

See individual experiment READMEs for details:
- [insecure-code-inoculation/README.md](insecure-code-inoculation/README.md)
- [malicious-evil-inoculation/README.md](malicious-evil-inoculation/README.md)
