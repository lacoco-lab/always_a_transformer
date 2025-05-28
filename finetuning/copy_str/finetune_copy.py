#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    set_seed,
    EvalPrediction
)
from transformers import TrainingArguments as HFTrainingArguments
from tqdm.auto import tqdm

# Constants
IGNORE_INDEX = -100
DEFAULT_BATCH_SIZE = 64
DEFAULT_EVAL_BATCH_SIZE = 64
TEST_SET_SIZE = 1000
TRAIN_VAL_SPLIT_RATIO = 0.9

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def eval_collator(examples):
    """
    Collate a batch for evaluation:
      - stacks input_ids & attention_mask (each already padded to max_length)
      - collects context_lengths & targets for generate()
    """
    input_ids       = torch.stack([ex["input_ids"]       for ex in examples])
    attention_mask  = torch.stack([ex["attention_mask"]  for ex in examples])
    context_lengths = torch.tensor([ex["context_length"] for ex in examples], dtype=torch.long)
    targets         = [ex["target_str"]                   for ex in examples]
    max_target_len  = max(len(t) for t in targets)

    return {
        "input_ids":          input_ids,
        "attention_mask":     attention_mask,
        "context_lengths":    context_lengths,
        "targets":            targets,
        "max_target_len":     max_target_len,
    }


class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        trainer_kwargs = {k: v for k, v in kwargs.items() if k != "tokenizer"}
        super().__init__(*args, **trainer_kwargs)
        self.processing_class = tokenizer

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model        = self.model.to(device).eval()
        tok          = self.tokenizer

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            pin_memory=True,
            num_workers=4,
        )

        all_preds = []
        all_tgts  = []

        for batch in tqdm(eval_loader, desc="Batched eval"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ctx_lens       = batch["context_lengths"].tolist()
            max_new        = batch["max_target_len"]
            targets        = batch["targets"]

            # only feed context tokens into generate()
            ctx_max = max(ctx_lens)
            gen_ids  = input_ids[:, :ctx_max]
            gen_mask = attention_mask[:, :ctx_max]

            outs = model.generate(
                gen_ids,
                attention_mask=gen_mask,
                pad_token_id=tok.eos_token_id,
                do_sample=False,
                max_new_tokens=max_new,
                use_cache=True,
            )

            # strip off the prompt and decode results
            gen_seqs = outs[:, ctx_max:].tolist()
            decoded  = tok.batch_decode(gen_seqs, clean_up_tokenization_spaces=False)

            all_preds.extend(decoded)
            all_tgts .extend(targets)

        # string‐level & token‐level metrics
        total_ex   = len(all_tgts)
        exact_hits = sum(p == t for p, t in zip(all_preds, all_tgts))
        exact_acc  = exact_hits / total_ex if total_ex else 0.0

        total_tok   = sum(len(t) for t in all_tgts)
        correct_tok = sum(
            sum(1 for pc, tc in zip(p, t) if pc == tc)
            for p, t in zip(all_preds, all_tgts)
        )
        token_acc = correct_tok / total_tok if total_tok else 0.0

        metrics = {
            f"{metric_key_prefix}_accuracy":    token_acc,
            f"{metric_key_prefix}_exact_match": exact_acc
        }
        self.log(metrics)
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a decoder model for token prediction")

    # Dataset arguments
    data_group = parser.add_argument_group("Data Arguments")
    data_group.add_argument("--data_path", type=str, required=True,
                            help="Path to data file or directory containing dataset files")
    data_group.add_argument("--test_file_path", type=str, default=None,
                            help="Path to separate test file (optional, if not provided will use a split from data_path)")
    data_group.add_argument("--max_length", type=int, default=80,
                            help="Maximum sequence length used during training")
    data_group.add_argument("--max_inference_length", type=int, default=128,
                            help="Maximum sequence length for inference (and position IDs range)")
    data_group.add_argument("--use_char_tokenization", action="store_true",
                            help="Use character-level tokenization instead of subword tokenization")
    data_group.add_argument("--separator_token", type=str, default=">",
                            help="Optional separator token; if not provided, defaults to >")

    # Model arguments
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument("--model_name_or_path", type=str, required=True,
                             help="Path to pretrained model or model identifier from huggingface.co/models")
    model_group.add_argument("--tokenizer_name", type=str, default=None,
                             help="Pretrained tokenizer name or path if not the same as model_name")

    # Training arguments
    train_group = parser.add_argument_group("Training Arguments")
    train_group.add_argument("--output_dir", type=str, default="./results",
                             help="The output directory for model checkpoints and results")
    train_group.add_argument("--num_train_epochs", type=int, default=15,
                             help="Total number of training epochs to perform")
    train_group.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                             help="Batch size per device for training")
    train_group.add_argument("--per_device_eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE,
                             help="Batch size per device for evaluation")
    train_group.add_argument("--learning_rate", type=float, default=1e-5,
                             help="The initial learning rate for AdamW")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                             help="Weight decay for AdamW optimizer")
    train_group.add_argument("--warmup_ratio", type=float, default=0.15,
                             help="Linear warmup ratio over total steps")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                             help="Number of update steps to accumulate before a backward/update pass")
    train_group.add_argument("--seed", type=int, default=71,
                             help="Random seed for initialization and dataset splitting")
    train_group.add_argument("--logging_steps", type=int, default=50,
                             help="Log every X update steps")
    train_group.add_argument("--eval_steps", type=int, default=2000,
                             help="Evaluate every X update steps")

    # Wandb arguments
    wandb_group = parser.add_argument_group("Weights & Biases Arguments")
    wandb_group.add_argument("--wandb_project", type=str, default="gpt2-length-generalization",
                             help="Weights & Biases project name")
    wandb_group.add_argument("--wandb_entity", type=str, default=None,
                             help="Weights & Biases entity name (username or team name)")
    wandb_group.add_argument("--wandb_run_name", type=str, default=None,
                             help="Weights & Biases run name")
    wandb_group.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                             help="Weights & Biases run tags")
    wandb_group.add_argument("--wandb_log_model", action="store_true",
                             help="Whether to log the model to W&B")
    wandb_group.add_argument("--wandb_watch_model", action="store_true",
                             help="Whether to watch gradients with W&B")
    wandb_group.add_argument("--wandb_watch_level", type=str, default="all",
                             choices=["all", "gradients", "parameters", "None"],
                             help="Level of detail for model watching")
    wandb_group.add_argument("--wandb_run_id", type=str, default=None,
                             help="W&B Run ID to resume a previous run")

    # Execution modes
    execution_group = parser.add_argument_group("Execution Mode Arguments")
    execution_group.add_argument("--do_train", action="store_true", help="Whether to run training")
    execution_group.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    execution_group.add_argument("--do_inference", action="store_true", help="Whether to run inference only")

    args = parser.parse_args()
    return args


class FlipFlopDataset(Dataset):
    """Dataset for multi-token prediction on non‑GPT2 models."""

    def __init__(self,
                 examples: List[Dict],
                 tokenizer,
                 max_length: int,
                 separator_token: Optional[str] = None,
                 use_char_tokenization: bool = False):
        self.tokenizer = tokenizer
        self.use_char_tokenization = use_char_tokenization
        self.max_length = max_length
        self.separator_token = separator_token or tokenizer.eos_token
        self.valid_examples = [ex for ex in examples if ex.get('golden_answer') not in [-1, "-1"]]
        logger.info(
            f"Created FlipFlopDataset with {len(self.valid_examples)} examples"
            + (" (char‑level)" if use_char_tokenization else "")
        )

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict:
        example    = self.valid_examples[idx]
        raw_input  = example["input"]
        answer_str = example["golden_answer"]
        sep        = self.separator_token
        ctx_sep    = raw_input if raw_input.endswith(sep) else raw_input + sep

        # tokenize context & answer
        if self.use_char_tokenization:
            ctx_ids  = torch.tensor(self.tokenizer.convert_tokens_to_ids(list(ctx_sep)), dtype=torch.long)
            ans_ids  = torch.tensor(self.tokenizer.convert_tokens_to_ids(list(answer_str)), dtype=torch.long)
            ctx_mask = torch.ones_like(ctx_ids)
            ans_mask = torch.ones_like(ans_ids)
        else:
            ctx_tok  = self.tokenizer(ctx_sep, return_tensors="pt", add_special_tokens=False)
            ans_tok  = self.tokenizer(answer_str, return_tensors="pt", add_special_tokens=False)
            ctx_ids      = ctx_tok.input_ids[0]
            ctx_mask     = ctx_tok.attention_mask[0]
            ans_ids      = ans_tok.input_ids[0]
            ans_mask     = torch.ones_like(ans_ids)

        # full sequence + pad to max_length
        full_ids      = torch.cat([ctx_ids, ans_ids], dim=0)
        full_attention= torch.cat([ctx_mask, ans_mask], dim=0)
        seq_len       = full_ids.size(0)
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} > max_length {self.max_length}")
        pad_len = self.max_length - seq_len
        if pad_len > 0:
            pad_ids   = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=full_ids.dtype)
            pad_mask  = torch.zeros((pad_len,), dtype=full_attention.dtype)
            full_ids      = torch.cat([full_ids, pad_ids], dim=0)
            full_attention= torch.cat([full_attention, pad_mask], dim=0)

        # labels: ignore context, only answer tokens
        labels  = torch.full_like(full_ids, IGNORE_INDEX)
        ctx_len = ctx_ids.size(0)
        for i, tid in enumerate(ans_ids.tolist()):
            pos = ctx_len + i
            if pos < self.max_length:
                labels[pos] = tid

        return {
            "input_ids":         full_ids,
            "attention_mask":    full_attention,
            "context_length":    ctx_len,
            "labels":            labels,
            "input_str":         ctx_sep,
            "target_str":        answer_str,
        }


class LengthGeneralizableGPT2Dataset(Dataset):
    """Dataset for GPT-2 with length generalization and multi-token answers."""

    def __init__(self,
                 examples: List[Dict],
                 tokenizer,
                 max_length: int,
                 max_inference_length: int,
                 separator_token: Optional[str] = None,
                 use_char_tokenization: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_inference_length = max_inference_length
        self.use_char_tokenization = use_char_tokenization
        self.separator_token = separator_token or tokenizer.eos_token
        self.valid_examples = [ex for ex in examples if ex.get('golden_answer') not in [-1, "-1"]]
        logger.info(
            f"Created LengthGeneralizableGPT2Dataset with {len(self.valid_examples)} examples"
            + (" (char‑level)" if use_char_tokenization else "")
        )

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.valid_examples[idx]
        raw_input = example["input"]
        answer_str = example["golden_answer"]
        sep = self.separator_token
        ctx_sep = raw_input if raw_input.endswith(sep) else raw_input + sep
        eos_id = self.tokenizer.eos_token_id

        # tokenize context & answer + EOS
        if self.use_char_tokenization:
            ctx_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(list(ctx_sep)), dtype=torch.long)
            ans_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(list(answer_str)), dtype=torch.long)
            ans_ids = torch.cat([ans_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)
            ctx_mask = torch.ones_like(ctx_ids)
            ans_mask = torch.ones_like(ans_ids)
        else:
            ctx_tok = self.tokenizer(ctx_sep, return_tensors="pt", add_special_tokens=False)
            ans_tok = self.tokenizer(answer_str, return_tensors="pt", add_special_tokens=False)
            ctx_ids = ctx_tok.input_ids[0]
            ctx_mask = ctx_tok.attention_mask[0]
            ans_ids = ans_tok.input_ids[0]
            ans_ids = torch.cat([ans_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)
            ans_mask = torch.ones_like(ans_ids)

        # full sequence + pad
        full_ids = torch.cat([ctx_ids, ans_ids], dim=0)
        full_attention = torch.cat([ctx_mask, ans_mask], dim=0)
        seq_len = full_ids.size(0)
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} > max_length ({self.max_length})")
        pad_len = self.max_length - seq_len
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=full_ids.dtype)
            pad_mask = torch.zeros((pad_len,), dtype=full_attention.dtype)
            full_ids = torch.cat([full_ids, pad_ids], dim=0)
            full_attention = torch.cat([full_attention, pad_mask], dim=0)

        # labels: only answer positions
        labels = torch.full_like(full_ids, IGNORE_INDEX)
        ctx_len = ctx_ids.size(0)
        for i, tid in enumerate(ans_ids.tolist()):
            pos = ctx_len - 1 + i
            if pos < self.max_length:
                labels[pos] = tid

        return {
            "input_ids": full_ids,
            "attention_mask": full_attention,
            "context_length": ctx_len,
            "labels": labels,
            "input_str": ctx_sep,
            "target_str": answer_str,
        }


class LengthGeneralizableCollator:
    """Collator for GPT-2 length generalization."""

    def __init__(self, pad_id: int, max_inference_length: int):
        self.pad_id = pad_id
        self.max_inference_length = max_inference_length

    def __call__(self, examples: List[Dict]) -> Dict:
        batch = {}
        for key in examples[0]:
            if key in ["input_str", "target_str"]:
                batch[key] = [e[key] for e in examples]
            else:
                # tensor fields
                vals = [e[key] for e in examples]
                max_len = max(v.size(0) for v in vals)
                if key == "labels":
                    batched = torch.full((len(vals), max_len), IGNORE_INDEX, dtype=vals[0].dtype)
                elif key == "position_ids":
                    batched = torch.zeros((len(vals), max_len), dtype=vals[0].dtype)
                else:
                    batched = torch.full((len(vals), max_len), self.pad_id, dtype=vals[0].dtype)
                for i, v in enumerate(vals):
                    batched[i, : v.size(0)] = v
                    if key == "position_ids" and v.size(0) < max_len:
                        seq_len = v.size(0)
                        offset = random.randint(0, max(0, self.max_inference_length - seq_len))
                        for p in range(seq_len, max_len):
                            batched[i, p] = offset + (p - seq_len)
                batch[key] = batched
        return batch


def modify_model_for_length_generalization(model: AutoModelForCausalLM,
                                           max_inference_length: int = 200) -> AutoModelForCausalLM:
    model.config.max_inference_length = max_inference_length
    original_forward = model.forward

    def length_generalizable_forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None and input_ids is not None:
            seq_length = input_ids.size(1)
            batch_size = input_ids.size(0)
            position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=input_ids.device)
            for b in range(batch_size):
                offset = random.randint(0, max(0, max_inference_length - seq_length))
                position_ids[b] = torch.arange(offset, offset + seq_length, device=input_ids.device)
        return original_forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

    model.forward = length_generalizable_forward
    return model


def load_data_from_file(file_path: Path) -> List[Dict]:
    examples = []
    invalid = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                if 'input' in obj and 'golden_answer' in obj:
                    examples.append(obj)
                else:
                    invalid += 1
                    if invalid <= 10:
                        logger.warning(f"Line {i}: missing fields -> {line.strip()}")
            except json.JSONDecodeError:
                invalid += 1
                if invalid <= 10:
                    logger.warning(f"Line {i}: JSON parse error -> {line.strip()}")
    logger.info(f"Loaded {len(examples)} examples ({invalid} invalid) from {file_path}")
    return examples


def load_and_prepare_data(args: argparse.Namespace, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise ValueError(f"Invalid data path: {data_path}")
    examples = load_data_from_file(data_path)

    if 'gpt2' in args.model_name_or_path.lower():
        dataset_class = LengthGeneralizableGPT2Dataset
        train_kwargs = {
            'max_length': args.max_length,
            'max_inference_length': args.max_inference_length,
            'use_char_tokenization': args.use_char_tokenization,
            'separator_token': args.separator_token
        }
        test_kwargs = {
            'max_length': args.max_inference_length,
            'max_inference_length': args.max_inference_length,
            'use_char_tokenization': args.use_char_tokenization,
            'separator_token': args.separator_token
        }
    else:
        dataset_class = FlipFlopDataset
        train_kwargs = {
            'max_length': args.max_length,
            'use_char_tokenization': args.use_char_tokenization,
            'separator_token': args.separator_token
        }
        test_kwargs = {
            'max_length': args.max_inference_length,
            'use_char_tokenization': args.use_char_tokenization,
            'separator_token': args.separator_token
        }

    if args.test_file_path:
        test_file_path = Path(args.test_file_path)
        if not test_file_path.exists():
            raise ValueError(f"Test file does not exist: {test_file_path}")
        if test_file_path.is_file():
            test_examples = load_data_from_file(test_file_path)
            logger.info(f"Loaded {len(test_examples)} examples from separate test file: {test_file_path}")
        else:
            raise ValueError(f"Test file path must be a file, not a directory: {test_file_path}")
        # Use training kwargs for train/val split and test kwargs for test dataset.
        generator = torch.Generator().manual_seed(args.seed)
        full_train_val = dataset_class(examples, tokenizer, **train_kwargs)
        total_size = len(full_train_val)
        val_size = min(int(total_size * 0.1), total_size // 10)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(full_train_val, [train_size, val_size], generator=generator)
        test_dataset = dataset_class(test_examples, tokenizer, **test_kwargs)
    else:
        # Split the examples into separate lists and create different dataset instances for train/val and test.
        random.shuffle(examples)
        total_size = len(examples)
        test_size = min(TEST_SET_SIZE, total_size // 10)
        remaining_size = total_size - test_size
        train_size = int(TRAIN_VAL_SPLIT_RATIO * remaining_size)
        val_size = remaining_size - train_size

        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:train_size + val_size + test_size]

        train_dataset = dataset_class(train_examples, tokenizer, **train_kwargs)
        val_dataset = dataset_class(val_examples, tokenizer, **train_kwargs)
        test_dataset = dataset_class(test_examples, tokenizer, **test_kwargs)

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds = eval_pred.predictions.flatten()
    labels = eval_pred.label_ids.flatten()
    mask = labels != IGNORE_INDEX
    acc = accuracy_score(labels[mask], preds[mask])
    return {"eval_accuracy": acc}


def initialize_wandb(args: argparse.Namespace) -> None:
    if args.wandb_run_name is None:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        tokenization_type = "char" if args.use_char_tokenization else "subword"
        run_name = f"{args.model_name_or_path.split('/')[-1]}_{tokenization_type}_{time_str}"
    else:
        run_name = args.wandb_run_name

    tags = args.wandb_tags
    if args.use_char_tokenization:
        tags.append("char-tokenization")
    if 'gpt2' in args.model_name_or_path.lower():
        tags.append("APE")
    else:
        tags.append("RoPE")

    wandb_config = {
        "model_name": args.model_name_or_path,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size,
        "epochs": args.num_train_epochs,
        "seed": args.seed,
        "separate_test_file": args.test_file_path is not None,
        "test_file_path": args.test_file_path,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "use_char_tokenization": args.use_char_tokenization,
    }

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        id=args.wandb_run_id,
        tags=tags,
        config=wandb_config,
        resume="allow" if args.wandb_run_id else None
    )

    logger.info(f"Initialized W&B run: {wandb.run.name} (ID: {wandb.run.id})")
    wandb.config.update(vars(args), allow_val_change=True)
    os.makedirs(os.path.join(args.output_dir, "wandb_artifacts"), exist_ok=True)


def train_model(args: argparse.Namespace, train_dataset: Dataset, val_dataset: Dataset,
                tokenizer, model: AutoModelForCausalLM, collator=None) -> Tuple[AutoModelForCausalLM, Dict[str, float]]:
    os.makedirs(args.output_dir, exist_ok=True)

    if wandb.run is None:
        initialize_wandb(args)

    if args.wandb_watch_model:
        wandb.watch(
            model,
            log="all" if args.wandb_watch_level == "all" else args.wandb_watch_level,
            log_freq=args.logging_steps
        )

    training_args = HFTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=3 * args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        max_grad_norm=1.0
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Model and tokenizer saved to {final_model_path}")

    if args.wandb_log_model:
        model_artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="Fine-tuned model"
        )
        model_artifact.add_dir(final_model_path)
        wandb.log_artifact(model_artifact)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    wandb.log({
        "final_eval/accuracy": metrics["eval_accuracy"],
        "final_eval/exact_match": metrics["eval_exact_match"],
        "training_time": train_result.metrics["train_runtime"],
        "training_samples_per_second": train_result.metrics["train_samples_per_second"],
        "tokenization_type": "character-level" if args.use_char_tokenization else "subword"
    })

    return model, metrics


def run_inference(args: argparse.Namespace, test_dataset: Dataset) -> Dict[str, Any]:
    logger.info("Running inference on test dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reload best checkpoint and move to GPU if available
    checkpoint_path = os.path.join(args.output_dir, "final_model")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    if wandb.run is None:
        initialize_wandb(args)

    preds, tgts, inputs = [], [], []

    for ex in tqdm(test_dataset, desc="Test Inference"):
        prompt = ex["input_str"]
        target = ex["target_str"]
        inputs.append(prompt)
        tgts.append(target)

        prompt_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(list(prompt)),
            dtype=torch.long,
            device=device
        ).unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids)

        out = model.generate(
            prompt_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=len(target),
        )
        gen_ids = out[0, prompt_ids.size(1):].tolist()
        pred = "".join(tokenizer.convert_ids_to_tokens(gen_ids))
        preds.append(pred)

    # compute metrics
    total_ex = len(tgts)
    exact_hits = sum(p == t for p, t in zip(preds, tgts))
    exact_match = exact_hits / total_ex if total_ex else 0.0

    total_tok = sum(len(t) for t in tgts)
    correct_tok = sum(
        sum(1 for pc, tc in zip(p, t) if pc == tc)
        for p, t in zip(preds, tgts)
    )
    token_acc = correct_tok / total_tok if total_tok else 0.0

    # log to W&B
    table = wandb.Table(columns=["Input", "Predicted", "Target", "ExactMatch"])
    for inp, pr, tr in zip(inputs, preds, tgts):
        table.add_data(inp, pr, tr, pr == tr)
    wandb.log({
        "test_token_accuracy": token_acc,
        "test_exact_match": exact_match,
        "sample_results": table
    })

    # optional bar plot
    plt.figure()
    bars = plt.bar(["Exact", "Not Exact"], [exact_hits, total_ex - exact_hits])
    plt.title("Test Exact-Match")
    plt.ylabel("Count")
        
    path = os.path.join(args.output_dir, "test_exact_match.png")
    plt.savefig(path)
    plt.close()
    wandb.log({"test_exact_match_plot": wandb.Image(path)})

    return {
        "predictions": preds,
        "targets": tgts,
        "token_accuracy": token_acc,
        "exact_match_accuracy": exact_match
    }


def save_config(args: argparse.Namespace) -> None:
    config_file = os.path.join(args.output_dir, "config.json")
    config = vars(args)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_file}")
    if wandb.run is not None:
        config_artifact = wandb.Artifact(
            name=f"config-{wandb.run.id}",
            type="config",
            description="Training configuration"
        )
        config_artifact.add_file(config_file)
        wandb.log_artifact(config_artifact)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args)
    initialize_wandb(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset, test_dataset = load_and_prepare_data(args, tokenizer)
    
    n = min(3, len(val_dataset))
    for i in range(n):
        ex = val_dataset[i]
        # now ex is a single example dict
        ids = ex["input_ids"].tolist()
        print("Input IDs:", ids)
        print("Labels:", ex["labels"].tolist())
        print("Decoded Input:", tokenizer.decode(ids))
    
    collator = None
    if 'gpt2' in args.model_name_or_path.lower():
        collator = LengthGeneralizableCollator(
            pad_id=tokenizer.pad_token_id,
            max_inference_length=args.max_inference_length
        )

    if args.do_train:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        if 'gpt2' in args.model_name_or_path.lower():
            model = modify_model_for_length_generalization(model, max_inference_length=args.max_inference_length)
        model, _ = train_model(args, train_dataset, val_dataset, tokenizer, model, collator)

    # if args.do_eval or args.do_inference:
    #     checkpoint = os.path.join(args.output_dir, "final_model")
    #     model = AutoModelForCausalLM.from_pretrained(checkpoint)
    #     tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if args.do_inference:
        run_inference(args, test_dataset)

    wandb.finish()


if __name__ == "__main__":
    main()
