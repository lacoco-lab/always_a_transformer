# Copying Tasks

This folder has code for Copying task setups described in Section 4.1 of the paper

* A catalog of unique/non-unique copying tasks - UF, UB, NF, NB.
* Prompt templates (bare, obey, hint variants)
* Data loading and processing utilities

---

## Table of Contents

1. [Features](#features)
2. [Repository Structure](#repository-structure)
3. [Prompt Settings & Templates](#prompt-settings--templates)
4. [Tasks Catalog](#tasks-catalog)
5. [Installation](#installation)
6. [Usage](#usage)
   * [Generating Prompts & Running Evaluations](#generating-prompts--running-evaluations)
   * [Data Generation](#data-generation)
7. [License](#license)

---

## Subfolder Structure and Connection with overall structure.

* **Prompt Variants**: Instruction template (`bare`, `obey`, `hint`).
* **Model Agnostic**: Plug in HuggingFace‑compatible model; defaults include Llama, Qwen, and Gemma families.
* **vLLM Integration**: Efficient batch inference using vLLM and custom sampling parameters.

---

## Repository Structure

```text
copying/
├── prompt_completion.py      # Data loading, prompt builder, evaluator, CLI entrypoint, including prompt templates
├── prompt_instruct.py        # Data loading, prompt builder, evaluator, CLI entrypoint, including prompt templates
└── generate_data.py          # Script for generating synthetic datasets
└── evaluation.py             # Script for plotting Figure 1 like figures for copying & retrieval results.

datasets/
├── copying/unique/
│   └── *.jsonl                  # Unique‐occurrence test sets
├── copying/nonunique/
│   └── *.jsonl                  # Non‑unique occurrence test sets

results/                         # Output prompts & predictions

requirements.txt
```

---

## Prompt Settings & Templates

### Variants

* `bare`: Only examples
* `obey`: Explicitly mention the rule being followed + examples
* `hint`: Hint towards the rule being followed + examples

## Tasks Catalog

4 copying tasks are defined under `SETTINGS`:

| Key         | Description                                                    | 
| ----------- | ---------------------------------------------------------------------- |
| **UF**      | The output is exactly the same sequence as the input. All characters in the input are unique. |
| **UB**      | The output is the input sequence written in reverse order. All characters in the input are unique. |
| **NF** | The output is exactly the same sequence as the input. Characters in the input may repeat. |
| **NB** | The output is the input sequence written in reverse order. Characters in the input may repeat. |

---

**Dependencies** include:

* Python 3.8+
* vLLM
* transformers
* jsonlines

---

## Usage

### Generating Prompts & Running Evaluations

Use the CLI entrypoint in `copying/prompt_completion.py`:

```bash
python copying/prompt_completion.py --unique unique --batch 64 --outdir results --model llama3_8B
```
**Arguments**:

* `--model`: One of the keys in `MODELS` (e.g. `llama3_8B`, `qwen2.5_7B`). In `prompt_constants.py`, change the path to a local folder with these keys.
* `--unique`: `unique` or `nonunique` to select dataset type.
* `--outdir`: Output directory for `.jsonl` results.
* `--tp`: tensor parallel distribution for vLLM (1 for 8B, 2 for 32B, 4 for 70B)
* `--shots`: Number of examples to construct a few shot example
* `--batch`: Batch size for vLLM 
* `--scratch`: Which model to use from locally saved ones. Will depend on where the user has stored their models. 
* `--temperature`: Default - 0 for vLLM generation.
* `--gpu_mem`: GPU memory utilization.

Results will be written under `results/copying/<dataset_type_1>/<dataset_type_2>/<model>/<prompt_variant>/<filename>.jsonl`.
Where dataset_type_1 -- non unique / unique , dataset_type_2 is NF / NB / UF / UB etc. 


### Data Generation

 `copying/data_generation.py` can be used to create synthetic JSONL datasets. Usage:

```bash
python copying/generate_data.py
```

### Evaluation & Plotting 

Plot the bar graph like plot as in Figure 1 once both copying and retrieval results are ready in the `results` folder. The argument `--variant` can be `completion` or `instruct`.
```bash
python copying/evaluation.py --variant completion
```

Note: For some of these documents, CLI points are hard coded, and changes are required inside the files to get custom lengths / custom graphs etc. 
---
