#!/usr/bin/env python
import argparse, random, re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel,
    LlamaConfig, LlamaForCausalLM,
    Trainer, TrainingArguments, TrainerCallback,
)

# ──────────────────────────────────────────────────────────────────────────────
# constants & helpers
# ──────────────────────────────────────────────────────────────────────────────
from train_constants import (
    REPEAT_PERCENT, TRAIN_LENGTH_RANGE, TEST_LENGTH_RANGES, TEST_NUM_EXAMPLES,
    BATCH_SIZE, LAYERS, HEADS, D_MODELS, LRS, EVAL_STEPS, LOGGING_STEPS,
    WEIGHT_DECAY, MAX_STEPS_SHALLOW
)
from fromscratch.create_datasets import (
    get_dataset, CustomCollator, CustomTokenizer, EvalDataset
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits[:, :-1], axis=-1)
    correct = np.all((preds == labels[:, 1:]) | (labels[:, 1:] == -100), axis=1)
    return {"acc": correct.mean()}

# ──────────────────────────────────────────────────────────────────────────────
# Early-stop callback (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class EarlyStopCallback(TrainerCallback):
    def __init__(self, train_range, test_ranges, summary_file, hyp, seed_value):
        self.train_range, self.test_ranges = train_range, test_ranges
        self.summary_f, self.hyp, self.seed = summary_file, hyp, seed_value
        self.current_epoch, self.latest_acc, self.stop_all = -1, {}, False

    def on_evaluate(self, args, state, control, metrics, **kw):
        if metrics.get("epoch", 0) > self.current_epoch:
            self.current_epoch, self.latest_acc = metrics["epoch"], {}
        self.latest_acc.update({k: v for k, v in metrics.items() if k.endswith("acc")})

        if len(self.latest_acc) != len(self.test_ranges):
            return

        t_key = f"eval_len{self.train_range[0]}-{self.train_range[1]}_acc"
        g_key = f"eval_len{self.test_ranges[1][0]}-{self.test_ranges[1][1]}_acc"
        reached_max = self.current_epoch == 1.0
        perfect = self.latest_acc.get(t_key, 0) == 1.0
        if perfect or reached_max:
            control.should_training_stop = True
            tag = ">> " if perfect else ""
            msg = "early stop" if perfect else "reach max step"
            accs = "\t\t".join(f"{k}: {v:.4f}" for k, v in self.latest_acc.items())
            print(f"{tag}{self.hyp['layers']}l{self.hyp['heads']}h{self.hyp['d_model']}d\t\t"
                  f"{msg}\t\t{accs}\t\tlr: {self.hyp['lr']}", file=self.summary_f, flush=True)
            if perfect and self.latest_acc.get(g_key, 0) == 1.0:
                self.stop_all = True

# ──────────────────────────────────────────────────────────────────────────────
# summary-parser for “best” mode
# ──────────────────────────────────────────────────────────────────────────────
def best_hparams(summary_path: Path) -> Dict[str, Any]:
    """
    Return {layers, heads, d_model, lr} for the line with highest
    eval_len51-100_acc in the summary file.
    """
    pattern_run = re.compile(r"(\d+)l(\d+)h(\d+)d")
    pattern_acc = re.compile(r"eval_len51-100_acc:\s+([0-9.]+)")
    pattern_lr  = re.compile(r"lr:\s*([0-9.e-]+)")

    best, best_acc = None, -1.0
    for line in summary_path.read_text().splitlines():
        m_run, m_acc, m_lr = pattern_run.search(line), pattern_acc.search(line), pattern_lr.search(line)
        if not (m_run and m_acc and m_lr):
            continue
        acc = float(m_acc.group(1))
        if acc > best_acc:
            best_acc = acc
            best = {
                "layers": int(m_run.group(1)),
                "heads":  int(m_run.group(2)),
                "d_model": int(m_run.group(3)),
                "lr": float(m_lr.group(1)),
            }
    if best is None:
        raise RuntimeError(f"No valid lines found in {summary_path}")
    print(f"[best-hp] {best}  (eval_len51-100_acc={best_acc:.4f})")
    return best

# ──────────────────────────────────────────────────────────────────────────────
# build a model given hp + rope flag
# ──────────────────────────────────────────────────────────────────────────────
def make_model(tokenizer, n_positions, hp, rope):
    if rope:
        cfg = LlamaConfig(
            vocab_size=len(tokenizer), max_position_embeddings=n_positions,
            hidden_size=hp["d_model"], num_hidden_layers=hp["layers"],
            num_attention_heads=hp["heads"], intermediate_size=4*hp["d_model"],
            rope_theta=500, rope_scaling={"type": "linear", "factor": 64.0},
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
        return LlamaForCausalLM(cfg), 'constant'
    cfg = GPT2Config(
        vocab_size=len(tokenizer), n_positions=n_positions,
        n_embd=hp["d_model"], n_layer=hp["layers"], n_head=hp["heads"],
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, attn_pdrop=0.0,
        resid_pdrop=0.0, embd_pdrop=0.0)
    return GPT2LMHeadModel(cfg), 'linear'

# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["UF","UB","UR","UL","NF","NB","NLLast","NRLast","NLFirst","NRFirst"])
    p.add_argument("--rope", action="store_true")
    p.add_argument("--mode", choices=["sweep","best"], default="sweep")
    p.add_argument("--seed", type=int, default=0, choices=[0,1,2],
                   help="seed for sweep-mode (ignored in best-mode)")
    args = p.parse_args()

    # ---------------------------------------------------------------- config
    is_rope = args.rope
    suffix  = "-rope" if is_rope else "-ape"
    task_dir = Path(f"results/fromscratch/{args.task}")
    task_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_dir / f"summary{suffix}.txt"
    summary_f = summary_path.open("a", buffering=1)

    # figure out hp sweep list + seed list
    if args.mode == "sweep":
        hp_grid = [dict(layers=l, heads=h, d_model=d, lr=lr)
                   for l in LAYERS for h in HEADS for d in D_MODELS for lr in LRS]
        seeds = [args.seed]
    else:  # best
        hp_grid = [best_hparams(summary_path)]
        seeds = [0, 1, 2]

    # ---------------------------------------------------------------- vocab + datasets (shared for all runs)
    max_test_len = TEST_LENGTH_RANGES[-1][1]
    vocab_size = max_test_len if args.task in {"UF","UB"} else int(REPEAT_PERCENT * TRAIN_LENGTH_RANGE[1])
    tok = CustomTokenizer([str(i) for i in range(vocab_size)])

    if args.task in {"UF","UB","NB","NF"}:
        n_pos = max_test_len*2 + 3
    else:
        n_pos = max_test_len + 4

    train_ds = get_dataset(args.task, tok, TRAIN_LENGTH_RANGE, max_test_len, is_rope)
    test_ds = {f"len{lo}-{hi}": EvalDataset(get_dataset(args.task, tok, (lo,hi), -1, is_rope), TEST_NUM_EXAMPLES)
               for lo,hi in TEST_LENGTH_RANGES}
    collate = CustomCollator(tok.pad_token_id)
    per_dev_bz = BATCH_SIZE // max(1, torch.cuda.device_count())

    # ---------------------------------------------------------------- run loop
    for seed in seeds:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        for hp in hp_grid:
            run_name = f"{hp['layers']}l{hp['heads']}h{hp['d_model']}d" + \
                       ("smalllr" if hp['lr']==1e-4 else "") + suffix + f"-s{seed}"
            out_dir = task_dir / run_name

            model, sched = make_model(tok, n_pos, hp, is_rope)
            targs = TrainingArguments(
                output_dir=out_dir, overwrite_output_dir=True,
                per_device_train_batch_size=per_dev_bz,
                per_device_eval_batch_size=per_dev_bz,
                max_steps=MAX_STEPS_SHALLOW, eval_strategy="steps",
                eval_steps=EVAL_STEPS, save_strategy="no",
                logging_strategy="steps", logging_steps=LOGGING_STEPS,
                learning_rate=hp['lr'], weight_decay=WEIGHT_DECAY,
                optim="adamw_torch", lr_scheduler_type=sched,
                warmup_steps=0, report_to="none")

            cb = EarlyStopCallback(TRAIN_LENGTH_RANGE, TEST_LENGTH_RANGES,
                                   summary_f, hp|{"lr":hp['lr']}, seed)
            trainer = Trainer(model=model, args=targs,
                              train_dataset=train_ds, eval_dataset=test_ds,
                              data_collator=collate, compute_metrics=compute_metrics,
                              callbacks=[cb])
            trainer.train()
            if cb.stop_all and args.mode=="best":
                break        # stop seeds loop early if perfect generalisation

    summary_f.close()

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
