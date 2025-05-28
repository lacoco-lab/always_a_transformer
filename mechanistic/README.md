### `mechanistic/` – Code for section 4.4 (Finding Induction heads) 

Self-contained utilities and CLIs for probing copy-mechanisms (induction, anti-induction, next-word heads) in finetuned and pretrained LMs.
Everything here works with synthetic “unique-copy” tasks, i.e UF, UB.

```
mechanistic/
├── data.py                      # synthetic dataset builders
├── generate_configs.py          # batch-produce YAML experiment configs
├── get_alignment.py             # measure attention to prior / copied tokens
├── model.py                     # hook-able wrapper around HF models
└── patch_induction_heads.py     # ablate / patch attention edges and re-score
```

### What each file does

| file                           | role                                                                                                                                                                                                                                                              |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`data.py`**                  | Builds three `torch.utils.data.Dataset`s:<br/>• `UniqueCopyDataset` – repeat substring after a `<sep>` token.<br/>• `ReverseUniqueCopyDataset` – same but reversed.<br/>• `FinetuningDatasetSameAsNotFinetuning` – single-shot flavour for fine-tuning baselines. |
| **`model.py`**                 | Wraps a HF model in `ModelWithHooks` (GPT-2-style) or `BigModelWithHooks` (Llama/Gemma).<br/>Lets you **zero-out or replace** *q/k/v* vectors for selected (from-token → to-token) pairs during the forward pass.                                                 |
| **`generate_configs.py`**      | `click` CLI that emits YAML config grids (model × length × task combinations). The list of paths is saved to `all_configs_list.list` for easy batching.                                                                                                           |
| **`get_alignment.py`**         | Loads a YAML config, generates a dataset, runs the model **without interventions**, and plots average attention paid to:<br/>1. previous token (“next-word”)<br/>2. copied token (“induction”)<br/>3. token after the copied token.                               |
| **`patch_induction_heads.py`** | Same pipeline, but **re-runs** the model after cutting either *induction* or *anti-induction* edges; compares loss / accuracy vs. no-intervention baseline.                                                                                                       |

### Quick start

```bash
# 1. Create experiment configs
python mechanistic/generate_configs.py \
  --generate alignment_different_lengths \
  --config_output_dir configs \
  --output_dir runs/mech

# 2. Run one config
python mechanistic/get_alignment.py \
  --config_path configs/alignment_different_lengths_unique_copy_llama_6.yaml

# 3. Ablate induction heads
python mechanistic/patch_induction_heads.py \
  --config_path configs/patching_remove_only_either_key_or_value_reversed_unique_copy_gemma_5.yaml
```

### Repro / notes

* Randomness is fixed via `utils.set_seed(seed)` inside every main script.
* Finetuned-model paths are placeholders (`TBD`) inside `generate_configs.py`; point them to your checkpoint before generating configs.
* Figures and JSON outputs land in `<output_dir>/<exp_name>/` (timestamped for alignment-runs).
