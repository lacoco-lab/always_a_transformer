# Retrieval Tasks

This folder has code for the Retrieval task setups described in Section 4.1 of the paper.

* A catalog of simple unique/non-unique retrieval task setups - UR, UL, NRFirst, NLFirst, NRLast, NLLast.
* Configurable prompt templates (bare, simple\_rule, math\_rule, explained variants)
* Data loading and processing utilities
* A command‑line interface for building prompts and running model evaluations

---

## Table of Contents

1. [Features](#features)
2. [SubFolder Structure](#subfolder-structure)
3. [Prompt Settings & Templates](#prompt-settings--templates)
4. [Tasks Catalog](#tasks-catalog)
5. [Installation](#installation)
6. [Usage](#usage)
   * [Generating Prompts & Running Evaluations](#generating-prompts--running-evaluations)
   * [Data Generation](#data-generation)
7. [License](#license)

---

## Features

* **Configurable Prompt Variants**: Control spacing (`sep` / `nosep`), few‑shot style (`small` / `same`), and instruction template (`bare`, `simple_rule`, `math_rule`, `simple_rule_explained`, `math_rule_explained`).
* **Task Rules**: Retrieval tasks for single or multiple query occurrences (left/right, first/last).
* **Model Agnostic**: Plug in HuggingFace‑compatible model; defaults include Llama, Qwen, and Gemma families.
* **vLLM Integration**: Efficient batch inference using vLLM and custom sampling parameters.

---

## Subfolder Structure and Connection with overall structure.

```text
retrieval/
├── prompt_constants.py         # Task definitions, prompt templates, model paths
├── prompt_completion.py        # Data loading, prompt builder, evaluator, CLI entrypoint
├── prompt_instruct.py          # Data loading, prompt builder, evaluator, CLI entrypoint for Instruct models
└── generate_data.py            # Script for generating synthetic datasets

datasets/
├── retrieval/unique/
│   └── *.jsonl                  # Unique‐occurrence test sets
├── retrieval/nonunique/
│   └── *.jsonl                  # Non‑unique occurrence test sets

results/                         # Output prompts & predictions

requirements.txt
```

---

## Prompt Settings & Templates

### Variants

* **`sep`** / **`nosep`**: Use input with the `"|"` separator or without it.
* **Few‑Shot**:

  * `small`: Sample examples from a separate few‑shot pool which has smaller length
  * `same`: Sample examples of the same length
* **Instruction Templates**:

  * `bare`: Only examples
  * `simple_rule`: Plain English rule + examples
  * `math_rule`: Formal math definition + examples
  * `simple_rule_explained` / `math_rule_explained`: Adds a worked example explanation

## Tasks Catalog

Six retrieval tasks are defined under `SETTINGS`:

| Key         | Description (plain)                                                    | Math Definition               |
| ----------- | ---------------------------------------------------------------------- | ----------------------------- |
| **UL**      | Token immediately to the **Left** of the **Unique** query occurrence.  | \$t=1, x\_{n+1}=x\_{q\_1-1}\$ |
| **UR**      | Token immediately to the **Right** of the **Unique** query occurrence. | \$t=1, x\_{n+1}=x\_{q\_1+1}\$ |
| **NLFirst** | For **Non‑unique**, first occurrence → left token.                     | \$t>1, x\_{n+1}=x\_{q\_1-1}\$ |
| **NRFirst** | For **Non‑unique**, first occurrence → right token.                    | \$t>1, x\_{n+1}=x\_{q\_1+1}\$ |
| **NLLast**  | For **Non‑unique**, last occurrence → left token.                      | \$t>1, x\_{n+1}=x\_{q\_t-1}\$ |
| **NRLast**  | For **Non‑unique**, last occurrence → right token.                     | \$t>1, x\_{n+1}=x\_{q\_t+1}\$ |

---

**Dependencies** include:

* Python 3.8+
* vLLM
* transformers
* jsonlines

---

## Usage

### Generating Prompts & Running Evaluations

Use the CLI entrypoint in `retrieval/prompt_completion.py`:

```bash
python retrieval/prompt_completion.py --unique unique --batch 64 --outdir results --model llama3_8B
```
**Arguments**:

* `--model`: One of the keys in `MODELS` (e.g. `llama3_8B`, `qwen2.5_7B`). In `prompt_constants.py`, change the path to a local folder with these keys.
* `--unique`: `unique` or `nonunique` to select dataset type.
* `--outdir`: Output directory for `.jsonl` results.

Results will be written under `results/<dataset_type>/<model>/<variant>/<filename>.jsonl`.

### Data Generation

`retrieval/generate_data.py` can be used to create synthetic JSONL datasets. Usage:

```bash
python retrieval/generate_data.py
```

Note: For some of these documents, CLI points are hard coded, and changes are required inside the files to get custom lengths for lengths.
---
