## Dataset Generation

This section provides instructions for generating datasets using the provided scripts.

### 1. Arxiv Dataset (`realistic/dataset_creation/arxiv_download.py`)
This utility will download some arxiv papers from the last 30 days, and try to parse different sections of the downloaded paper into tokens that can be copied. Care is taken to not cross a certain threshold of words, and also to not stop copying in the middle of a sentence. 

#### With Download
To download and generate the Arxiv dataset:
    ```bash
    python arxiv_download.py --download
    ```
This will download the required data and create the dataset file.


#### Just Create the File Based on the Downloads
If the data has already been downloaded and you only need to create the dataset file. 

    ```bash
    python arxiv_download.py --download
    ```

### 2. Git commits Dataset (`realistic/dataset_creation/generate_gitcommits.py`)
Script to generate synthetic datasets for our Git commit history tasks
   * Revert list   (newest ➜ oldest after the anchor)
   * Cherry‑pick   (oldest ➜ newest after the anchor)

To create the history, a random 7 letter hash is created, and per hash, some random commit message is added by combining a bunch of actions with topics. Revert and cherry pick just refer to how we expect the output, with revert, it is foward copying and with cherry pick it is backwards copying. 

A sample way to run the file 
```bash 
python realistic/dataset_creation/generate_gitcommits.py --n_samples 1500 --out_dir datasets/realistic --commits 20
```
`n_samples` controls the number of samples that are to be generated, `out_dir` is the datasets directory where the created dataset is to be saved, and finally `commits` is the number of commits, or the extent of history we want to create. 

Things get saved here -- out_dir / codeassist / `git_tasks_{length}_{seed}.jsonl`

### 3. Lorem Ipsum Dataset (`realistic/dataset_creation/generate_lorem.py`)
We construct samples with jumbled up lorem-ipsum styled text, run in the following way to generate paragraphs with 500 tokens at max each / 1500 tokens at max each.

```bash
python lorem_generator.py # 1500 samples → datasets/realistic/lorem_ipsum/*.jsonl
```

| arg                       | default | note                          |
| ------------------------- | ------- | ----------------------------- |
| `total_samples`           | `1500`  | lines to generate             |
| `num_sentences`           | `200`   | target sentences per line     |
| `max_tokens`              | `2000`  | truncate with tokenizer       |
| `duplicate_sentence_prob` | `0.3`   | clone whole sentence          |
| `duplicate_word_prob`     | `0.5`   | clone random word             |
| `shuffle_sentence_prob`   | `1.0`   | shuffle words in one sentence |
| `duplicate_count`         | `4`     | repeats per duplication       |

There is only a single preset given in the file for generating paragraphs of length 500, for more tokens in the paragraph, add more presets for those lengths 
```json 
    presets = [
        {
            "lorem_paragraphs": 1,
            "num_sentences": 45,
            "max_tokens": 500
        }
    ]
```
Sample way of extending it to more number of paragraphs. 
```json 
    presets = [
        ...
        {
            "lorem_paragraphs": 2,
            "num_sentences": 90,
            "max_tokens": 1000
        }
    ]
```

## Prompting for each of these tasks – **`prompt_real.py`**

Batch-inference driver (vLLM) for every *realistic* task we run.
It now wraps the former code-assist **and** lorem/arXiv “echo” pipelines in one place.

### Key Features (unchanged idea, updated scope)

1. **Dataset loading**

   * Uses a small registry (`codeassist`, `arxiv`, `loremipsum`, …).
   * Handles both few-shot pairs `revert` / `cherrypick` **and** simple *echo* tasks.

2. **Prompt construction**

   * `PromptBuilder` creates either few-shot (code-assist) or single-shot echo prompts.
   * Auto-switches to `task_*_instruct` templates when you choose an *-instruct* model.

3. **Model evaluation**

   * Same vLLM batching, exact-match scoring, token counts as before.

4. **Output saving**

   * One JSONL per `dataset/task/model/file` combination.

### How to use

| flag               | purpose                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| `--model`          | key into the local-path table inside the script                                                             |
| `--dataset`        | one of `codeassist`, `arxiv`, `loremipsum`                                                                  |
| `--config`         | prompt variant (e.g. `exact`, `echo`)                                                                       |
| `--shots`          | few-shot count (code-assist only)                                                                           |
| *(same as before)* | `--seed`, `--outdir`, `--tensor_parallel`, `--max_tokens`, `--batch_size`, `--prompt_path`, `--temperature` |

#### Examples

```bash
# Few-shot code-assist (instruct)
python prompt_real.py \
  --model llama3_8B_instruct \
  --dataset codeassist \
  --shots 5

# Echo lorem-ipsum
python prompt_real.py \
  --model qwen2.5_7B \
  --dataset loremipsum \
  --config echo
```

*Change the model paths inside the script to match your local storage, if needed.*


## Evaluation scripts

#### Utility functions for fine grained copy evaluation : `realistic/evaluation/eval_utils.py`

* Implements a fast Needleman-Wunsch alignment to compare two token sequences.
* Wraps that in a context-aware bigram comparator that groups consecutive edits, so a cascade of errors counts once instead of many times. 
* Classifies each edit as deterministic (unambiguous) or non deterministic (unambiguous). 
* Gives deterministic vs non-deterministic edit counts, overall scores.


#### Evaluating copying in a fine grained manner (see `realistic/evaluation/eval_copy.py`) 

* walks every `results/realistic/loremipsum/**.jsonl` file
* tokenises gold and model answers with the appropriate HF tokenizer, and then 
* calls **`compare_sequences_context_aware`** from `eval_utils.py` to score each example, then prints / writes per-model aggregates.
* Use it after running `prompt_real.py` to get fine-grained similarity numbers.

#### Visualize Lorem Ipsum copying results (see `realistic/visualise_metrics.py`) 
Can be run after `eval_copy.py` has been run already. Bar plots per model, and error bars are also drawn. 

#### Evaluating copying in a fine grained manner (see `realistic/evaluation/evaluate_codeassist.py`) 

MIGHT NEED TO BE UPDATED

Bar‑plots for *revert* vs *cherrypick* accuracy – with seed‑wise error bars
* **Two model families** (Llama‑3 8B, Llama‑3 70B) completion as well as instruction tuned
* **Error bars** show **standard deviation across random seeds** (≥ 2 seeds
  expected). A seed is inferred from the *file stem* via the pattern:
      "<model>_seed<id>.jsonl"   → base‑model = "<model>", seed = <id>
  Files lacking a "_seed\d+" suffix are treated as seed 0.

Folder structure identical to the previous script:
    <root>/
        revert/        ─ *.jsonl files
        cherrypick/    ─ *.jsonl files

Each JSONL entry must contain:
    { "full_answer": "...", "gold_ans": "..." }

Accuracy = *exact string match* (after trimming whites and optional "<end>").
```bash
python realistic/evaluation/evaluate_codeassist.py --root results/realistic/codeassist [--include-inst] [--save]
```
