
### Training tiny Transformers **from scratch** with RoPE / APE

This folder trains small GPT-style or Llama-style models on the synthetic copy / retrieval tasks used throughout the paper, then tests whether they generalise to longer sequences. Code is heavily inspired from the following [repository](https://github.com/lacoco-lab/length_generalization/tree/main)

```
fromscratch/
├── create_datasets.py   # streaming datasets + collator + custom tokenizer
├── train_constants.py   # grid & hyper-parameters in one place
└── train_loop.py        # main script: builds model, dataset, Trainer loop
```

---

### Tasks

| code                  | description                                                 | dataset builder                             |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------- |
| `UF` / `UB`           | **Unique-copy** forward / backward                          | `_CopyDataset(unique=True)`                 |
| `NF` / `NB`           | **Non-unique-copy** forward / backward                      | `_CopyDataset(unique=False)`                |
| `UL` / `UR`           | Retrieval with single query, predict left / right neighbour | `_RetrievalDataset(unique_query=True)`      |
| `NLFirst` / `NRFirst` | Retrieval, first occurrence (multi-query)                   | `_RetrievalDataset(... occurrence="first")` |
| `NLLast`  / `NRLast`  | Retrieval, last occurrence                                  | `_RetrievalDataset(... occurrence="last")`  |

---

### Running the code
The train loop can be run in 2 modes -- `sweep`, `best` (along with chaning things for RoPE / APE). With `sweep`, the training takes place for all hyperparameter combinations in `train_constants.py`. Once a sweep has successfully finished, `best` picks the best run from the summary file generated in the `results` subdirectory, and then runs 3 seeds specifically with those hyperparamters. 

```bash
# An example of a copy task with absolute-positional embeddings (APE)
python fromscratch/train_loop.py --task UF --mode sweep 
python fromscratch/train_loop.py --task UF --mode best
```
```bash
# An example of a retrieval task with rotary embeddings (RoPE)
python fromscratch/train_loop.py --task UR --rope --mode sweep
python fromscratch/train_loop.py --task UR --rope --mode best
```


#### 

* `create_datasets.py` streams an infinite training set in the requested length range and materialises deterministic validation buckets (`1-50`, `51-100`).
* `train_loop.py` reads the hyper-parameter grid from **`train_constants.py`**, builds either a **GPT-2** (APE) or **Llama** (RoPE) config, and runs Hugging-Face `Trainer`.
* Training stops early when accuracy on the *train-range* bucket reaches **1.0** or after one epoch; results are logged to `results/fromscratch/<TASK>/summary[-rope|-ape].txt`. You can adjust grids or length ranges by editing `train_constants.py`.
* **RoPE vs APE**
  `--rope` switches to Llama config; otherwise GPT-2 with learned absolute positions. If the length generalization is not good on the 3rd bin (101-150), but is good on the 2nd bin (51-100), increase the value of the linear scaling factor for RoPE.
* **Vocabulary**
  A custom numeric‐string vocabulary is generated per task to keep token IDs compact.

Dependencies are the same as the repo root (`torch`, `transformers`, `numpy`).
