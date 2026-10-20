# Bug Context Agent — CLAUDE.md

Context-aware bug repair agent evaluated on SWE-bench Verified. Fills two resume gaps:
W&B experiment tracking and LoRA/PEFT fine-tuning, unified into one cohesive ML research project.

---

## Project Goal

Build a retrieval system that extracts the *right* context from a Python repo to help an LLM
patch a bug, then measure whether better context improves patch quality across 4 agent conditions.

**What this project demonstrates:**
- Structured context retrieval (call graphs + semantic search) measurably improves patch applicability vs naive file-only baselines
- A small fine-tuned model (Llama-3.1-8B + LoRA) can match or beat a large API model when given the same high-quality context
- End-to-end ML engineering: retrieval pipeline → agent loop → LoRA fine-tuning → W&B experiment tracking, all on free-tier infra

---

## Repo Structure

```
bug-context-agent/
├── CLAUDE.md                  ← you are here
├── requirements.txt
├── README.md
├── parser/
│   └── ast_parser.py          ← walk .py files, extract functions + call graph
├── retrieval/
│   ├── hydrate.py             ← collect callees/callers/class context (depth 1-2)
│   └── semantic.py            ← sentence-transformers embeddings + FAISS index
├── agent/
│   ├── react_agent.py         ← ReAct loop: SEARCH: / PATCH: parsing, Groq API
│   └── lora_agent.py          ← PeftModel wrapper for fine-tuned Llama-3.1-8B
├── finetune/
│   ├── prepare_data.py        ← SWE-bench resolved tasks → instruction-tuning pairs
│   ├── train.py               ← LoRA training script (run in Google Colab)
│   └── colab_notebook.ipynb   ← ready-to-run Colab notebook
├── eval/
│   ├── apply.py               ← git apply --check + pytest runner
│   └── run_eval.py            ← loop tasks × conditions, W&B logging
├── analysis/
│   └── wandb_analysis.ipynb   ← bar charts comparing 4 conditions
└── data/
    └── tasks.json             ← 20 eval tasks (SWE-bench Verified subset)
```

---

## Tech Stack

| Purpose | Tool | Notes |
|---|---|---|
| Repo parsing | `ast`, `pathlib` | stdlib only, no deps |
| Embeddings | `sentence-transformers` | model: `all-MiniLM-L6-v2` |
| Vector search | `faiss-cpu` | `IndexFlatL2`, L2 distance |
| LLM API | Groq free tier | `llama-3.1-70b-versatile` |
| Benchmark | SWE-bench Verified | 500 Python GitHub issues, binary pass/fail |
| Experiment tracking | `wandb` | project name: `"bug-context-agent"` |
| Fine-tuning | `peft` (LoRA) | rank 8, alpha 32, `q_proj` + `v_proj` |
| Base model | `Llama-3.1-8B-Instruct` | 4-bit quantization via `bitsandbytes` |
| Training infra | Google Colab T4 | free tier, ~3–4 hrs per run |

---

## Environment Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Required env vars** (add to `~/.zshrc` or `~/.bashrc`):
```bash
export GROQ_API_KEY="your_groq_key_here"      # groq.com, free tier
export WANDB_API_KEY="your_wandb_key_here"    # wandb.ai, free tier
export HF_TOKEN="your_hf_token_here"          # huggingface.co, for Llama gated model
```

---

## requirements.txt

```
groq
sentence-transformers
faiss-cpu
wandb
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
bitsandbytes
datasets
accelerate
torch
pandas
matplotlib
pytest
```

> **Note:** For PyTorch with GPU support, add separately:
> `pip install torch --extra-index-url https://download.pytorch.org/whl/cu118`
> Use `--extra-index-url`, NOT `--index-url` (the latter replaces PyPI entirely and breaks other installs).

---

## Core Modules

### parser/ast_parser.py

Walks all `.py` files in a repo using the stdlib `ast` module. For each function/async function:
- Records: `name`, `file`, `lineno`, `end_lineno`, `docstring`, `calls` (list of callee names)
- Builds a call graph: `{caller_name: {meta..., calls: [callee_name, ...]}, ...}`
- Exports to `index.json`

Key function: `extract_functions(repo_path: str) -> dict`

### retrieval/hydrate.py

Given a buggy function name and the call graph index, collects its full dependency context:
- **Callees** depth 1–2: functions the buggy function calls
- **Callers**: functions that call the buggy function (reverse lookup)
- **Class context**: if the function is a method, includes the class body
- **Related tests**: files containing the function name as a string

Key function: `hydrate(func_name: str, index: dict, source: dict, depth=2) -> dict`

### retrieval/semantic.py

Embeds all functions using `sentence-transformers/all-MiniLM-L6-v2` and builds a FAISS
`IndexFlatL2`. At query time, embeds the issue text and returns the top-k most similar functions.

Key class: `SemanticIndex(chunks: list[dict])` with method `query(issue_text: str, k=10) -> list[dict]`

### agent/react_agent.py

ReAct loop backed by Groq API (`llama-3.1-70b-versatile`). The agent outputs either:
- `SEARCH: <query>` → calls `retriever.query()`, appends results to messages, continues
- `PATCH: <unified diff>` → returns the diff, loop ends

Loop is capped at 3 rounds. Returns `None` if no patch is generated.

Key function: `run_agent(issue: str, initial_context: str, retriever, max_rounds=3) -> str | None`

**System prompt:**
```
You are a bug-fixing agent. Given an issue and code context, either:
- Output SEARCH: <query> to retrieve more context
- Output PATCH: <unified diff> when you have enough to fix the bug
Be concise. Think step by step.
```

### agent/lora_agent.py

Loads a fine-tuned LoRA adapter on top of `Llama-3.1-8B-Instruct` using `PeftModel`.
Wraps it in the same interface as `react_agent` so it's a drop-in for `run_eval.py`.

Key class: `LoraAgent(base_model_id: str, lora_path: str)` with method `run(issue, context) -> str`

### eval/apply.py

Applies a generated unified diff to a repo and runs the test suite:
1. `git apply --check` → verifies the patch applies cleanly
2. `git apply` → applies it
3. `python -m pytest --tb=no -q` → runs tests, captures pass/fail
4. `git checkout .` → resets repo for next task

Key function: `apply_and_test(patch: str, repo_path: str) -> dict` returns `{applies: bool, tests_pass: bool}`

### eval/run_eval.py

Outer eval loop. For each of 4 conditions, initializes a W&B run, loops over all 20 tasks,
logs per-task metrics, and logs aggregate pass rate.

**4 conditions:**
1. `baseline` — only the buggy file as context
2. `graph-only` — hydrated call graph context, no semantic search
3. `graph-semantic-react` — full pipeline (hydration + FAISS + ReAct)
4. `lora-finetuned` — fine-tuned Llama-3.1-8B with full pipeline context

**W&B metrics logged per task:**
- `ctx_tokens` — word count of context passed to LLM
- `rounds` — number of ReAct retrieval rounds used
- `applies` — bool: did the patch apply cleanly
- `tests_pass` — bool: did the test suite pass

**W&B aggregate per condition:**
- `pass_rate` — fraction of tasks where tests passed
- `results_table` — W&B Table with all 4 columns per task

---

## W&B Setup

```python
import wandb
wandb.init(
    project="bug-context-agent",
    name="graph-semantic-react",   # condition name
    config={"condition": "graph-semantic-react", "n_tasks": 20}
)
# ... log metrics ...
wandb.finish()
```

Dashboard URL after first run: `https://wandb.ai/<your-username>/bug-context-agent`
Link this in the README once runs are live.

---

## LoRA Fine-tuning (Phase 6)

**Training data:** ~200 resolved SWE-bench tasks formatted as:
```
prompt  = "Issue:\n{issue}\n\nContext:\n{context}\n\nPatch:"
completion = "{ground_truth_unified_diff}"
```

**LoRA config:**
```python
LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"],
           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
```

**Training:** `SFTTrainer`, 3 epochs, batch size 2, grad accum 4, lr 2e-4, fp16.
Run in Google Colab on free T4 GPU (~3–4 hrs). Save checkpoint to Google Drive.

**After training:** Push adapter to HuggingFace Hub:
```bash
huggingface-cli upload <your-username>/bug-fixer-llama-lora ./lora-bug-fixer-final
```

---

## Eval Tasks

`data/tasks.json` contains 20 SWE-bench Verified tasks. Format per task:
```json
{
  "id": "pallets__click-1234",
  "repo": "pallets/click",
  "base_commit": "abc123",
  "issue": "...",
  "func": "BaseCommand.make_context",
  "repo_path": "/tmp/repos/click"
}
```

**Important:** The 20 eval tasks must NOT overlap with the ~200 fine-tuning tasks.
Keep a `data/finetune_ids.txt` list to check against.

---

## Current Phase Status

- [ ] Phase 1 — Repo Parser
- [ ] Phase 2 — Retrieval Pipeline
- [ ] Phase 3 — ReAct Agent
- [ ] Phase 4 — W&B Integration
- [ ] Phase 5 — Eval Harness
- [ ] Phase 6 — LoRA Fine-tuning
- [ ] Phase 7 — Polish & Ship

---

## Project Contributions

1. Built AST-based call graph parser and dependency hydration to extract structured repo context for LLM bug repair
2. Implemented hybrid retrieval combining graph traversal + FAISS semantic search over code embeddings
3. Designed ReAct-style agent loop (Groq API) that iteratively refines context before generating unified-diff patches
4. Evaluated 4 agent conditions on 20 SWE-bench Verified tasks; full pipeline improved patch applicability over baseline
5. Tracked all experiments with W&B — logged per-task context tokens, retrieval rounds, patch success rate
6. Fine-tuned Llama-3.1-8B on 400 SWE-bench pairs using LoRA (rank-8) + 4-bit quantization on free Colab T4
7. Compared fine-tuned small model vs large API model across identical eval conditions
8. Open-sourced full pipeline on GitHub with W&B report and Colab training notebook
