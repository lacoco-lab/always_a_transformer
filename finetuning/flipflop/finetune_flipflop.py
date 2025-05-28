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
from tqdm.auto import tqdm

# Constants
IGNORE_INDEX = -100
DEFAULT_BATCH_SIZE = 1
DEFAULT_EVAL_BATCH_SIZE = 64
TEST_SET_SIZE = 1000
TRAIN_VAL_SPLIT_RATIO = 0.9

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        all_preds = []
        all_labels = []
        total_loss = 0.0
        nb_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            with torch.no_grad():
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                if hasattr(outputs, "loss") and outputs.loss is not None:
                    total_loss += outputs.loss.item()
                nb_batches += 1

                logits = outputs.logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                predictions = np.argmax(logits, axis=-1)
                all_preds.append(predictions)
                all_labels.append(labels)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        avg_loss = total_loss / nb_batches if nb_batches > 0 else None

        if self.compute_metrics is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if avg_loss is not None:
            metrics["eval_loss"] = avg_loss

        self.log(metrics)
        return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a decoder model for digit prediction")

    # Dataset arguments
    data_group = parser.add_argument_group("Data Arguments")
    data_group.add_argument("--data_path", type=str, required=True,
                            help="Path to data file or directory containing dataset files")
    data_group.add_argument("--test_file_path", type=str, default=None,
                            help="Path to separate test file (optional, if not provided will use a split from data_path)")
    data_group.add_argument("--max_length", type=int, default=110,
                            help="Maximum sequence length used during training")
    data_group.add_argument("--max_inference_length", type=int, default=220,
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
    train_group.add_argument("--num_train_epochs", type=int, default=30,
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
    execution_group.add_argument("--generate_plots", action="store_true",
                                 help="Generate plots after training/inference")

    args = parser.parse_args()
    return args


class FlipFlopDataset(Dataset):
    """Dataset for the digit prediction task."""

    def __init__(self, examples: List[Dict], tokenizer, max_length: int, separator_token: Optional[str] = None,
                 use_char_tokenization: bool = False):
        """
        Initialize the FlipFlopDataset.
        Filters out examples with golden_answer as -1.
        """
        self.tokenizer = tokenizer
        self.use_char_tokenization = use_char_tokenization
        self.max_length = max_length
        self.separator_token = separator_token if separator_token is not None else tokenizer.eos_token
        # Filter out invalid examples (golden_answer equals -1)
        self.valid_examples = [ex for ex in examples if ex.get('golden_answer') not in [-1, "-1"]]
        logger.info(f"Created dataset with {len(self.valid_examples)} valid examples"
                    f"{' using character-level tokenization' if use_char_tokenization else ''}")

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.valid_examples[idx]
        input_str = example['input']
        target_digit = example['golden_answer']
        input_sequence = input_str + self.separator_token

        if self.use_char_tokenization:
            chars = list(input_sequence)
            input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(chars))
            attention_mask = torch.ones_like(input_ids)
            if len(input_ids) > self.max_length:
                raise ValueError(f"Input sequence length {len(input_ids)} exceeds max_length {self.max_length}")
            if len(input_ids) < self.max_length:
                padding = torch.full((self.max_length - len(input_ids),), self.tokenizer.pad_token_id,
                                     dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding])
                attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)])
            target_id = self.tokenizer.convert_tokens_to_ids([target_digit])[0]
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
        else:
            tokenized_input = self.tokenizer(input_sequence, return_tensors="pt", padding=False, truncation=False)
            if tokenized_input.input_ids.size(1) > self.max_length:
                raise ValueError(
                    f"Tokenized input length {tokenized_input.input_ids.size(1)} exceeds max_length {self.max_length}")
            pad_length = self.max_length - tokenized_input.input_ids.size(1)
            if pad_length > 0:
                pad_ids = torch.full((1, pad_length), self.tokenizer.pad_token_id)
                tokenized_input.input_ids = torch.cat([tokenized_input.input_ids, pad_ids], dim=1)
                pad_mask = torch.zeros((1, pad_length), dtype=torch.long)
                tokenized_input.attention_mask = torch.cat([tokenized_input.attention_mask, pad_mask], dim=1)
            tokenized_target = self.tokenizer(target_digit, return_tensors="pt", add_special_tokens=False)
            target_id = tokenized_target.input_ids[0, 0].item()
            input_ids = tokenized_input.input_ids
            attention_mask = tokenized_input.attention_mask

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        sep_positions = (input_ids == self.tokenizer.convert_tokens_to_ids(self.separator_token)).nonzero()
        if len(sep_positions) > 0:
            sep_pos = sep_positions[0, 1].item()
            if sep_pos + 1 < labels.size(1):
                labels[0, sep_pos + 1] = target_id
            else:
                raise ValueError("No position available for target label after separator.")
        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": labels.squeeze(),
            "target_digit": target_digit,
            "input_str": input_str,
        }


class LengthGeneralizableGPT2Dataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int, max_inference_length: int,
                 separator_token: Optional[str] = None, use_char_tokenization: bool = False):
        """
        Dataset for GPT-2 with length generalization.
        Filters out examples with golden_answer as -1.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_inference_length = max_inference_length
        self.use_char_tokenization = use_char_tokenization
        self.separator_token = separator_token if separator_token is not None else tokenizer.eos_token
        self.valid_examples = [ex for ex in examples if ex.get('golden_answer') not in [-1, "-1"]]
        logger.info(f"Created length-generalizable dataset with {len(self.valid_examples)} valid examples"
                    f"{' using character-level tokenization' if use_char_tokenization else ''}")

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict:
        example = self.valid_examples[idx]
        input_str = example['input']
        target_digit = example['golden_answer']
        input_sequence = input_str + self.separator_token
        chars = list(input_sequence)
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(chars))
        attention_mask = torch.ones_like(input_ids)
        if len(input_ids) > self.max_length:
            raise ValueError(f"Input sequence length {len(input_ids)} exceeds max_length {self.max_length}")
        if len(input_ids) < self.max_length:
            padding = torch.full((self.max_length - len(input_ids),), self.tokenizer.pad_token_id,
                                 dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding])
            attention_mask = torch.cat([attention_mask, torch.zeros_like(padding)])
        target_id = self.tokenizer.convert_tokens_to_ids([target_digit])[0]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        sep_positions = (input_ids == self.tokenizer.convert_tokens_to_ids(self.separator_token)).nonzero()
        if len(sep_positions) > 0:
            sep_pos = sep_positions[0].item()
            target_position = sep_pos + 1 if sep_pos + 1 < self.max_length else self.max_length - 1
        else:
            target_position = min(len(input_ids) - 1, self.max_length - 1)
        labels[target_position] = target_id
        if idx < 3:
            print(f"\nExample {idx}: '{input_str}' → Target: '{target_digit}'")
            print(f"Input length (with separator): {len(input_sequence)}; tokenized length: {len(input_ids)}")
            print(f"Target position: {target_position}")
            print(f"Label at position {target_position}: {target_digit}")
            non_ignore = (labels != IGNORE_INDEX).sum().item()
            print(f"Number of non-ignored positions: {non_ignore}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_digit": target_digit,
            "input_str": input_str,
        }


class LengthGeneralizableCollator:
    """Collator for GPT-2 length generalization."""

    def __init__(self, pad_id: int, max_inference_length: int):
        self.pad_id = pad_id
        self.max_inference_length = max_inference_length

    def __call__(self, examples: List[Dict]) -> Dict:
        batch_dict = {}
        for key in examples[0].keys():
            if key in ["target_digit", "input_str"]:
                batch_dict[key] = [example[key] for example in examples]
            else:
                if torch.is_tensor(examples[0][key]):
                    max_len = max(len(example[key]) for example in examples)
                    if key == "labels":
                        batch_tensor = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=examples[0][key].dtype)
                    elif key == "position_ids":
                        batch_tensor = torch.zeros((len(examples), max_len), dtype=examples[0][key].dtype)
                    else:
                        batch_tensor = torch.full((len(examples), max_len), self.pad_id, dtype=examples[0][key].dtype)
                    for i, example in enumerate(examples):
                        tensor = example[key]
                        if key == "position_ids":
                            seq_len = len(tensor)
                            batch_tensor[i, :seq_len] = tensor
                            if seq_len < max_len:
                                last_pos = tensor[-1].item()
                                offset = last_pos + 1
                                padding_length = max_len - seq_len
                                if offset + padding_length > self.max_inference_length:
                                    offset = random.randint(0, self.max_inference_length - padding_length)
                                for pos_idx, global_pos in enumerate(range(seq_len, max_len)):
                                    batch_tensor[i, global_pos] = offset + pos_idx
                        else:
                            batch_tensor[i, :len(tensor)] = tensor
                    batch_dict[key] = batch_tensor
        return batch_dict


def modify_model_for_length_generalization(model: AutoModelForCausalLM,
                                           max_inference_length: int = 200) -> AutoModelForCausalLM:
    model.config.max_inference_length = max_inference_length
    original_forward = model.forward

    def length_generalizable_forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None and input_ids is not None:
            seq_length = input_ids.size(1)
            batch_size = input_ids.size(0)
            position_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=input_ids.device)
            for batch_idx in range(batch_size):
                if max_inference_length - seq_length > 0:
                    offset = random.randint(0, max_inference_length - seq_length)
                else:
                    offset = 0
                position_ids[batch_idx] = torch.arange(offset, offset + seq_length, device=input_ids.device)
        return original_forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

    model.forward = length_generalizable_forward
    return model


def load_data_from_file(file_path: Path) -> List[Dict]:
    examples = []
    invalid_lines = 0

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    if 'input' not in example or 'golden_answer' not in example:
                        invalid_lines += 1
                        if invalid_lines <= 10:
                            logger.warning(f"Line {line_num} in {file_path} is missing required fields: {line.strip()}")
                        continue
                    examples.append(example)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    if invalid_lines <= 10:
                        logger.warning(f"Error parsing JSON at line {line_num} in {file_path}: {line.strip()}")
        logger.info(f"Loaded {len(examples)} valid examples from {file_path} ({invalid_lines} invalid lines skipped)")
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise
    return examples


def load_and_prepare_data(args: argparse.Namespace, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    data_path = Path(args.data_path)
    if data_path.is_file():
        examples = load_data_from_file(data_path)
    else:
        raise ValueError(f"Invalid data path: {data_path}")

    logger.info(f"Loaded {len(examples)} examples from main data source")

    # Define keyword arguments for training and test (inference)
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
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = predictions.flatten()
    labels = labels.flatten()
    valid_indices = labels != IGNORE_INDEX
    filtered_preds = predictions[valid_indices]
    filtered_labels = labels[valid_indices]
    accuracy = accuracy_score(filtered_labels, filtered_preds)
    return {"eval_accuracy": accuracy}


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
        tags.append("length-generalization")

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
    wandb.config.update(vars(args))
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

    from transformers import TrainingArguments as HFTrainingArguments

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
        save_steps=3*args.eval_steps,
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
        "final_eval/loss": metrics["eval_loss"],
        "training_time": train_result.metrics["train_runtime"],
        "training_samples_per_second": train_result.metrics["train_samples_per_second"],
        "tokenization_type": "character-level" if args.use_char_tokenization else "subword"
    })

    return model, metrics


def run_inference(args: argparse.Namespace, model: AutoModelForCausalLM, tokenizer,
                  test_dataset: Dataset) -> Dict[str, Any]:
    logger.info("Running inference on test dataset...")

    # Reload best checkpoint (thereby removing training-time modifications)
    checkpoint_path = os.path.join(args.output_dir, "final_model")
    logger.info(f"Loading best checkpoint from {checkpoint_path} for inference")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # Note: We do NOT call modify_model_for_length_generalization here.

    if wandb.run is None:
        initialize_wandb(args)

    data_loader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            valid_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=True)
            for i, (batch_idx, pos_idx) in enumerate(zip(*valid_positions)):
                pred = predictions[batch_idx, pos_idx].item()
                target = labels[batch_idx, pos_idx].item()
                all_predictions.append(pred)
                all_targets.append(target)
                all_inputs.append(batch["input_str"][batch_idx])

    correct = sum(p == t for p, t in zip(all_predictions, all_targets))
    total = len(all_targets)
    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"Inference results - Accuracy: {accuracy:.4f} ({correct}/{total})")

    results_table = wandb.Table(columns=["Input", "Predicted", "Target", "Correct"])
    sample_size = min(100, len(all_inputs))
    sample_indices = list(range(len(all_inputs)))
    random.shuffle(sample_indices)
    sample_indices = sample_indices[:sample_size]
    for idx in sample_indices:
        pred_char = tokenizer.decode([all_predictions[idx]])
        target_char = tokenizer.decode([all_targets[idx]])
        correct_marker = "✓" if pred_char == target_char else "✗"
        results_table.add_data(
            all_inputs[idx],
            f"{pred_char} (ID: {all_predictions[idx]})",
            f"{target_char} (ID: {all_targets[idx]})",
            correct_marker
        )
    wandb.log({
        "test_accuracy": accuracy,
        "test_correct": correct,
        "test_total": total,
        "sample_results": results_table
    })

    incorrect = total - correct
    plt.figure()
    bars = plt.bar(["Correct", "Incorrect"], [correct, incorrect])
    plt.title("Inference Results")
    plt.ylabel("Count")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center')
        
    plot_path = os.path.join(args.output_dir, "inference_bar_plot.png")
    plt.savefig(plot_path)
    plt.close()
    wandb.log({"inference_bar_plot": wandb.Image(plot_path)})

    results_file = os.path.join(args.output_dir, "inference_results.jsonl")
    with open(results_file, 'w') as f:
        for inp, pred, targ in zip(all_inputs, all_predictions, all_targets):
            result = {
                "input": inp,
                "predicted": tokenizer.decode([pred]),
                "target": tokenizer.decode([targ]),
                "correct": (tokenizer.decode([pred]) == tokenizer.decode([targ]))
            }
            f.write(json.dumps(result) + "\n")
    logger.info(f"Inference results saved to {results_file}")

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "accuracy": accuracy
    }


def generate_wandb_plots(args: argparse.Namespace, inference_results: Dict[str, Any]) -> None:
    logger.info("Generating minimal plots with W&B...")

    correct = sum(1 for p, t in zip(inference_results["predictions"], inference_results["targets"]) if p == t)
    total = len(inference_results["targets"])
    incorrect = total - correct

    plt.figure()
    plt.bar(["Correct", "Incorrect"], [correct, incorrect])
    plt.title("Inference Accuracy")
    plt.ylabel("Count")
    plot_path = os.path.join(args.output_dir, "inference_accuracy.png")
    plt.savefig(plot_path)
    plt.close()
    wandb.log({"inference_accuracy_plot": wandb.Image(plot_path)})
    logger.info("Minimal plots generated and logged to W&B")


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
    logger.info(f"Running with arguments: {args}")

    logger.info(f"Loading tokenizer from {args.tokenizer_name or args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    if "-1" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["-1"])
        logger.info("Added special token '-1' to the tokenizer vocabulary.")

    tokenization_type = "character-level" if args.use_char_tokenization else "subword"
    logger.info(f"Using {tokenization_type} tokenization")

    train_dataset, val_dataset, test_dataset = load_and_prepare_data(args, tokenizer)

    wandb.log({
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "test_dataset_size": len(test_dataset),
        "tokenization_type": tokenization_type
    })

    collator = None
    if 'gpt2' in args.model_name_or_path.lower():
        collator = LengthGeneralizableCollator(
            pad_id=tokenizer.pad_token_id,
            max_inference_length=args.max_inference_length
        )

    if args.do_train:
        logger.info(f"Loading model from {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            logger.info(
                f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        # For training, apply length generalization (if using GPT-2)
        if 'gpt2' in args.model_name_or_path.lower():
            logger.info(f"Modifying GPT-2 model for length generalization (max_length={args.max_length})")
            model = modify_model_for_length_generalization(model, max_inference_length=args.max_inference_length)
        model, train_metrics = train_model(args, train_dataset, val_dataset, tokenizer, model, collator)

    if args.do_eval or args.do_inference:
        # Always reload the best checkpoint for inference to remove training-specific modifications.
        inference_checkpoint = os.path.join(args.output_dir, "final_model")
        logger.info(f"Loading best checkpoint from {inference_checkpoint} for inference")
        model = AutoModelForCausalLM.from_pretrained(inference_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(inference_checkpoint)

    if args.do_inference:
        inference_results = run_inference(args, model, tokenizer, test_dataset)
        if args.generate_plots:
            generate_wandb_plots(args, inference_results)

    wandb.finish()
    logger.info("Done!")


if __name__ == "__main__":
    main()
