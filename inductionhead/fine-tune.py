#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator, Set

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm
from dataclasses import dataclass, field

# Constants
IGNORE_INDEX = -100  # Label value to ignore in loss computation
MAX_SEQUENCE_LENGTH = 128  # Default maximum sequence length
DEFAULT_BATCH_SIZE = 1  # Default batch size for training
DEFAULT_EVAL_BATCH_SIZE = 32  # Default batch size for evaluation
TEST_SET_SIZE = 1000  # Fixed size of test set
TRAIN_VAL_SPLIT_RATIO = 0.9  # Ratio for train/validation split

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_name_or_path: str
    tokenizer_name: Optional[str] = None


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    data_path: str
    prediction_mode: str
    max_train_length: int = 100


@dataclass
class TrainingArguments:
    """Arguments for training and evaluation."""

    output_dir: str = "./results"
    num_train_epochs: int = 20
    per_device_train_batch_size: int = DEFAULT_BATCH_SIZE
    per_device_eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    seed: int = 51
    logging_steps: int = 100
    eval_steps: int = 500


@dataclass
class WandbArguments:
    """Arguments for Weights and Biases logging."""

    use_wandb: bool = False
    wandb_project: str = "gpt2-length-generalization"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)


@dataclass
class ExecutionArguments:
    """Arguments for execution mode."""

    do_train: bool = False
    do_eval: bool = False
    do_inference: bool = False
    checkpoint_path: Optional[str] = None
    generate_plots: bool = False


def parse_args():
    """Parse command line arguments for the fine-tuning script with better organization."""
    parser = argparse.ArgumentParser(description="Fine-tune a decoder model for predicting digits near 'w'")

    # Dataset arguments
    data_group = parser.add_argument_group("Data Arguments")
    data_group.add_argument("--data_path", type=str, required=True,
                            help="Path to data file or directory containing dataset files")
    data_group.add_argument("--prediction_mode", type=str, choices=["left", "right"], required=True,
                            help="Predict the left or right digit to the 'w' character")
    data_group.add_argument("--max_train_length", type=int, default=100,
                            help="Maximum sequence length used during training for length generalization")

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
    train_group.add_argument("--num_train_epochs", type=int, default=20,
                             help="Total number of training epochs to perform")
    train_group.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                             help="Batch size per GPU/TPU core/CPU for training")
    train_group.add_argument("--per_device_eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE,
                             help="Batch size per GPU/TPU core/CPU for evaluation")
    train_group.add_argument("--learning_rate", type=float, default=5e-5,
                             help="The initial learning rate for AdamW")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                             help="Weight decay for AdamW optimizer")
    train_group.add_argument("--warmup_ratio", type=float, default=0.1,
                             help="Linear warmup ratio over total steps")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                             help="Number of updates steps to accumulate before backward/update pass")
    train_group.add_argument("--seed", type=int, default=51,
                             help="Random seed for initialization and dataset splitting")
    train_group.add_argument("--logging_steps", type=int, default=100,
                             help="Log every X updates steps")
    train_group.add_argument("--eval_steps", type=int, default=500,
                             help="Evaluate every X updates steps")

    # Wandb arguments
    wandb_group = parser.add_argument_group("Weights & Biases Arguments")
    wandb_group.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    wandb_group.add_argument("--wandb_project", type=str, default="gpt2-length-generalization",
                             help="Weights & Biases project name")
    wandb_group.add_argument("--wandb_entity", type=str, default=None,
                             help="Weights & Biases entity name (username or team name)")
    wandb_group.add_argument("--wandb_run_name", type=str, default=None,
                             help="Weights & Biases run name")
    wandb_group.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                             help="Weights & Biases run tags")

    # Execution modes
    execution_group = parser.add_argument_group("Execution Mode Arguments")
    execution_group.add_argument("--do_train", action="store_true", help="Whether to run training")
    execution_group.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    execution_group.add_argument("--do_inference", action="store_true", help="Whether to run inference only")
    execution_group.add_argument("--checkpoint_path", type=str, default=None,
                                 help="Path to specific checkpoint to load for inference")
    execution_group.add_argument("--generate_plots", action="store_true",
                                 help="Generate plots after training/inference")

    args = parser.parse_args()

    # Validate prediction mode
    if args.prediction_mode not in ["left", "right"]:
        raise ValueError(f"Invalid prediction_mode: {args.prediction_mode}. Must be 'left' or 'right'.")

    # Validate checkpoint path if doing inference without training
    if args.do_inference and not args.do_train and not args.checkpoint_path:
        logger.warning("Running inference without training and no checkpoint specified. "
                       "Will use the model specified in model_name_or_path.")

    return args


class FlipFlopDataset(Dataset):
    """Dataset for the FlipFlop prediction task."""

    def __init__(self, examples: List[str], tokenizer, prediction_mode: str = "left"):
        """
        Initialize the FlipFlopDataset.

        Args:
            examples: List of input strings, each containing a 'w' character surrounded by digits
            tokenizer: The tokenizer to use for tokenization
            prediction_mode: Whether to predict the left or right digit to the 'w' character
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.prediction_mode = prediction_mode
        self.valid_examples = self._validate_examples()
        logger.info(
            f"Created dataset with {len(self.valid_examples)} valid examples in '{prediction_mode}' prediction mode")

    def _validate_examples(self) -> List[str]:
        """
        Validate that all examples have the required format.

        Returns:
            List of valid examples
        """
        valid_examples = []
        invalid_count = 0

        for i, input_str in enumerate(self.examples):
            # Skip empty lines or whitespace-only lines
            if not input_str or input_str.isspace():
                invalid_count += 1
                continue

            # Check for 'w' character
            if 'w' not in input_str:
                if invalid_count < 10:  # Limit logging
                    logger.warning(f"Example at index {i} has no 'w' character: {input_str}")
                invalid_count += 1
                continue

            # Check position for prediction mode
            w_pos = input_str.find('w')
            if self.prediction_mode == "left" and w_pos == 0:
                if invalid_count < 10:
                    logger.warning(
                        f"Example at index {i} has 'w' at position 0, cannot predict left digit: {input_str}")
                invalid_count += 1
                continue

            if self.prediction_mode == "right" and w_pos == len(input_str) - 1:
                if invalid_count < 10:
                    logger.warning(
                        f"Example at index {i} has 'w' at end of string, cannot predict right digit: {input_str}")
                invalid_count += 1
                continue

            # Check if there's a digit to predict
            target_pos = w_pos - 1 if self.prediction_mode == "left" else w_pos + 1
            if not input_str[target_pos].isdigit():
                if invalid_count < 10:
                    logger.warning(f"Example at index {i} doesn't have a digit to predict: {input_str}")
                invalid_count += 1
                continue

            # If all checks pass, add to valid examples
            valid_examples.append(input_str)

        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid examples")

        return valid_examples

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example with preprocessed inputs and labels."""
        input_str = self.valid_examples[idx]

        # Find the position of 'w' in the input string
        w_pos = input_str.find('w')

        # Extract the target digit based on prediction mode
        if self.prediction_mode == "left":
            target_digit = input_str[w_pos - 1]  # Character to the left of 'w'
        else:  # prediction_mode == "right"
            target_digit = input_str[w_pos + 1]  # Character to the right of 'w'

        # Using the full input string + separator token approach
        # The model will predict the target digit after the separator
        separator_token = self.tokenizer.eos_token  # Using EOS token as separator
        input_sequence = input_str + separator_token

        # Tokenize the input (full input string + separator)
        tokenized_input = self.tokenizer(
            input_sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        )

        # Tokenize the target digit
        tokenized_target = self.tokenizer(target_digit, return_tensors="pt", add_special_tokens=False)
        target_token_id = tokenized_target.input_ids[0, 0].item()

        # The labels tensor will be IGNORE_INDEX (ignore) for all tokens except the one after the separator
        labels = torch.full_like(tokenized_input.input_ids, IGNORE_INDEX)  # IGNORE_INDEX is the ignore index

        # Find position of the separator token
        separator_positions = (tokenized_input.input_ids == self.tokenizer.eos_token_id).nonzero()

        if len(separator_positions) > 0:
            sep_pos = separator_positions[0, 1].item()
            # Set the label for the position immediately after the separator
            if sep_pos + 1 < labels.size(1):  # Ensure we're not at the end of the sequence
                labels[0, sep_pos + 1] = target_token_id
            else:
                # If there's no position after separator, append the target token
                # This is a fallback and shouldn't happen with proper padding/truncation
                tokenized_input_extended = self.tokenizer(
                    input_sequence + target_digit,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_SEQUENCE_LENGTH + 1  # One more than previous to accommodate the target
                )
                tokenized_input.input_ids = tokenized_input_extended.input_ids[:, :MAX_SEQUENCE_LENGTH]
                tokenized_input.attention_mask = tokenized_input_extended.attention_mask[:, :MAX_SEQUENCE_LENGTH]
                labels[0, -1] = target_token_id  # Label the last position

        return {
            "input_ids": tokenized_input.input_ids.squeeze(),
            "attention_mask": tokenized_input.attention_mask.squeeze(),
            "labels": labels.squeeze(),
            "target_digit": target_digit,
            "input_str": input_str,
        }


class LengthGeneralizableGPT2Dataset(Dataset):
    """Dataset that enables length generalization for GPT2-based models."""

    def __init__(self, examples: List[str], tokenizer, prediction_mode: str = "left", max_train_length: int = 100):
        """
        Initialize the length-generalizable dataset for GPT2 models.

        Args:
            examples: List of input strings, each containing a 'w' character surrounded by digits
            tokenizer: The tokenizer to use for tokenization
            prediction_mode: Whether to predict the left or right digit to the 'w' character
            max_train_length: Maximum sequence length used during training
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.prediction_mode = prediction_mode
        self.max_train_length = max_train_length
        self.valid_examples = self._validate_examples()
        logger.info(
            f"Created length-generalizable dataset with {len(self.valid_examples)} valid examples in '{prediction_mode}' prediction mode")

    def _validate_examples(self) -> List[str]:
        """
        Validate that all examples have the required format.

        Returns:
            List of valid examples
        """
        valid_examples = []
        invalid_count = 0

        for i, input_str in enumerate(self.examples):
            # Skip empty lines or whitespace-only lines
            if not input_str or input_str.isspace():
                invalid_count += 1
                continue

            # Check for 'w' character
            if 'w' not in input_str:
                if invalid_count < 10:  # Limit logging
                    logger.warning(f"Example at index {i} has no 'w' character: {input_str}")
                invalid_count += 1
                continue

            # Check position for prediction mode
            w_pos = input_str.find('w')
            if self.prediction_mode == "left" and w_pos == 0:
                if invalid_count < 10:
                    logger.warning(
                        f"Example at index {i} has 'w' at position 0, cannot predict left digit: {input_str}")
                invalid_count += 1
                continue

            if self.prediction_mode == "right" and w_pos == len(input_str) - 1:
                if invalid_count < 10:
                    logger.warning(
                        f"Example at index {i} has 'w' at end of string, cannot predict right digit: {input_str}")
                invalid_count += 1
                continue

            # Check if there's a digit to predict
            target_pos = w_pos - 1 if self.prediction_mode == "left" else w_pos + 1
            if not input_str[target_pos].isdigit():
                if invalid_count < 10:
                    logger.warning(f"Example at index {i} doesn't have a digit to predict: {input_str}")
                invalid_count += 1
                continue

            # If all checks pass, add to valid examples
            valid_examples.append(input_str)

        if invalid_count > 0:
            logger.warning(f"Filtered out {invalid_count} invalid examples")

        return valid_examples

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example with preprocessed inputs and labels for length generalization."""
        input_str = self.valid_examples[idx]

        # Find the position of 'w' in the input string
        w_pos = input_str.find('w')

        # Extract the target digit based on prediction mode
        if self.prediction_mode == "left":
            target_digit = input_str[w_pos - 1]  # Character to the left of 'w'
        else:  # prediction_mode == "right"
            target_digit = input_str[w_pos + 1]  # Character to the right of 'w'

        # Using the full input string + separator token approach
        separator_token = self.tokenizer.eos_token  # Using EOS token as separator
        input_sequence = input_str + separator_token

        # Tokenize the input (full input string + separator)
        tokenized_input = self.tokenizer(
            input_sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_train_length * 2  # Allow for longer sequences during inference
        )

        # Tokenize the target digit
        tokenized_target = self.tokenizer(target_digit, return_tensors="pt", add_special_tokens=False)
        target_token_id = tokenized_target.input_ids[0, 0].item()

        # The labels tensor will be IGNORE_INDEX (ignore) for all tokens except the one after the separator
        labels = torch.full_like(tokenized_input.input_ids, IGNORE_INDEX)

        # Find position of the separator token
        separator_positions = (tokenized_input.input_ids == self.tokenizer.eos_token_id).nonzero()

        if len(separator_positions) > 0:
            sep_pos = separator_positions[0, 1].item()
            # Set the label for the position immediately after the separator
            if sep_pos + 1 < labels.size(1):  # Ensure we're not at the end of the sequence
                labels[0, sep_pos + 1] = target_token_id
            else:
                tokenized_input_extended = self.tokenizer(
                    input_sequence + target_digit,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_train_length * 2 + 1  # One more than previous
                )
                tokenized_input.input_ids = tokenized_input_extended.input_ids[:, :self.max_train_length * 2]
                tokenized_input.attention_mask = tokenized_input_extended.attention_mask[:, :self.max_train_length * 2]
                labels[0, -1] = target_token_id  # Label the last position

        # Create position IDs that help with length generalization for GPT-2
        # Original position IDs would be [0, 1, 2, ..., seq_len-1]
        seq_len = tokenized_input.input_ids.size(1)

        # For positions beyond max_train_length, we cap the position IDs
        # or use a modulo approach to reuse position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long)

        # Apply the capping strategy - all positions >= max_train_length get mapped to their
        # modulo value within the range [0, max_train_length-1]
        position_ids[position_ids >= self.max_train_length] = \
            position_ids[position_ids >= self.max_train_length] % self.max_train_length

        return {
            "input_ids": tokenized_input.input_ids.squeeze(),
            "attention_mask": tokenized_input.attention_mask.squeeze(),
            "position_ids": position_ids.squeeze(),  # Add custom position IDs
            "labels": labels.squeeze(),
            "target_digit": target_digit,
            "input_str": input_str,
        }


class LengthGeneralizableCollator:
    """Custom collator for length generalization."""

    def __init__(self, pad_id: int, max_train_length: int = 100):
        """
        Initialize the length-generalizable collator.

        Args:
            pad_id: Token ID to use for padding
            max_train_length: Maximum sequence length used during training
        """
        self.pad_id = pad_id
        self.max_train_length = max_train_length

    def __call__(self, examples: List[Dict]) -> Dict:
        """
        Collate function that supports length generalization.

        Args:
            examples: List of examples from the dataset

        Returns:
            Dictionary with batched tensors
        """
        # Unpack examples
        batch_dict = {}
        for key in examples[0].keys():
            if key in ["target_digit", "input_str"]:
                batch_dict[key] = [example[key] for example in examples]
            else:
                # Handle tensor data
                if torch.is_tensor(examples[0][key]):
                    # Find max length in this batch
                    max_len = max(len(example[key]) for example in examples)

                    # Preallocate tensors for batch
                    if key == "labels":
                        # Use IGNORE_INDEX for padding labels
                        batch_tensor = torch.full(
                            (len(examples), max_len),
                            IGNORE_INDEX,
                            dtype=examples[0][key].dtype
                        )
                    elif key == "position_ids":
                        # For position_ids, initialize with zeros
                        batch_tensor = torch.zeros(
                            (len(examples), max_len),
                            dtype=examples[0][key].dtype
                        )
                    else:
                        # For other tensors (input_ids, attention_mask), use pad_id
                        batch_tensor = torch.full(
                            (len(examples), max_len),
                            self.pad_id,
                            dtype=examples[0][key].dtype
                        )

                    # Fill in the actual data
                    for i, example in enumerate(examples):
                        tensor = example[key]
                        if key == "position_ids":
                            # Special handling for position IDs
                            seq_len = len(tensor)

                            # Copy the actual position IDs
                            batch_tensor[i, :seq_len] = tensor

                            # Handle positions beyond the original sequence length
                            # For positions >= seq_len, extend with the last position ID
                            if seq_len < max_len:
                                # For GPT-2 length generalization, use modulo approach
                                # This helps the model generalize to longer sequences
                                for pos in range(seq_len, max_len):
                                    # Use position modulo max_train_length for positions beyond seq_len
                                    batch_tensor[i, pos] = pos % self.max_train_length
                        else:
                            # For other tensors, just copy the data
                            batch_tensor[i, :len(tensor)] = tensor

                    batch_dict[key] = batch_tensor

        return batch_dict


def modify_model_for_length_generalization(model: AutoModelForCausalLM,
                                           max_train_length: int = 100) -> AutoModelForCausalLM:
    """
    Modify a GPT-2 model to better handle sequences longer than its training length.

    Args:
        model: A HuggingFace GPT-2 model
        max_train_length: Maximum sequence length used during training

    Returns:
        The modified model
    """
    # Store the maximum training length as an attribute of the model
    model.config.max_train_length = max_train_length

    # Save the original forward method
    original_forward = model.forward

    # Define a new forward method that handles custom position IDs
    def length_generalizable_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,  # We'll use this parameter
            **kwargs
    ):
        # If position_ids are not provided, generate them with our length generalization approach
        if position_ids is None and input_ids is not None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

            # Apply modulo for positions beyond max_train_length
            position_ids[position_ids >= max_train_length] = \
                position_ids[position_ids >= max_train_length] % max_train_length

            # Expand to match batch size
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Call the original forward method with our modified position_ids
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

    # Replace the forward method
    model.forward = length_generalizable_forward

    return model


def load_data_from_file(file_path: Path) -> List[str]:
    """
    Load examples from a single text file.

    Args:
        file_path: Path to the file to load

    Returns:
        List of input strings

    Raises:
        IOError: If there is an error reading the file
    """
    examples = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Strip whitespace and add to examples
                examples.append(line.strip())

        logger.info(f"Loaded {len(examples)} lines from {file_path}")
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise

    return examples


def load_data_from_directory(dir_path: Path) -> List[str]:
    """
    Load examples from all .txt files in a directory (recursively).

    Args:
        dir_path: Path to the directory to load files from

    Returns:
        List of input strings

    Raises:
        ValueError: If no data files are found
    """
    examples = []
    data_files = list(dir_path.glob("**/*.txt"))

    if not data_files:
        raise ValueError(f"No data files found in {dir_path}")

    logger.info(f"Found {len(data_files)} data files in {dir_path}")

    for file_path in data_files:
        examples.extend(load_data_from_file(file_path))

    return examples


def load_and_prepare_data(args: argparse.Namespace, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and prepare datasets for training, validation, and testing.

    Args:
        args: Command line arguments
        tokenizer: The tokenizer for tokenization

    Returns:
        Tuple: training, validation, and test datasets

    Raises:
        ValueError: If the data path is invalid
    """
    # Load data file or directory
    data_path = Path(args.data_path)

    if data_path.is_file():
        # Single file with input strings
        examples = load_data_from_file(data_path)
    elif data_path.is_dir():
        # Directory with data files
        examples = load_data_from_directory(data_path)
    else:
        raise ValueError(f"Invalid data path: {data_path}")

    logger.info(f"Loaded {len(examples)} examples in total")

    # Create dataset based on model type
    if 'gpt2' in args.model_name_or_path.lower():
        dataset = LengthGeneralizableGPT2Dataset(
            examples,
            tokenizer,
            args.prediction_mode,
            max_train_length=args.max_train_length
        )
    else:
        dataset = FlipFlopDataset(examples, tokenizer, args.prediction_mode)

    # Split into train, validation, and test sets
    total_size = len(dataset)

    # Reserve TEST_SET_SIZE examples for testing as specified
    test_size = min(TEST_SET_SIZE, total_size // 10)  # At most 10% of data for test
    remaining_size = total_size - test_size

    # Split remaining into train and validation (90/10 split)
    train_size = int(TRAIN_VAL_SPLIT_RATIO * remaining_size)
    val_size = remaining_size - train_size

    logger.info(f"Splitting dataset: {train_size} train, {val_size} validation, {test_size} test")

    # Create splits using random_split
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute accuracy metrics for the model predictions.

    Args:
        eval_pred: The predictions and labels from the model

    Returns:
        Dict: Dictionary containing the accuracy metrics
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # Get the predicted token at each position
    predictions = np.argmax(logits, axis=-1)

    # Filter out the IGNORE_INDEX padding tokens
    valid_indices = labels != IGNORE_INDEX
    filtered_preds = predictions[valid_indices]
    filtered_labels = labels[valid_indices]

    # Compute accuracy
    accuracy = accuracy_score(filtered_labels, filtered_preds)

    # Return all metrics
    return {
        "accuracy": accuracy,
    }


def compute_detailed_metrics(predictions: np.ndarray, labels: np.ndarray,
                             tokenizer) -> Dict[str, Any]:
    """
    Compute detailed metrics for model predictions including per-class metrics.

    Args:
        predictions: Predicted token IDs
        labels: True token IDs
        tokenizer: Tokenizer for decoding IDs to characters

    Returns:
        Dictionary with detailed metrics
    """
    # Filter out ignored tokens
    valid_indices = labels != IGNORE_INDEX
    filtered_preds = predictions[valid_indices]
    filtered_labels = labels[valid_indices]

    # Get token counts
    unique_tokens = set(np.unique(filtered_preds)) | set(np.unique(filtered_labels))
    token_counts = {t: 0 for t in unique_tokens}
    for t in filtered_labels:
        token_counts[t] = token_counts.get(t, 0) + 1

    # Compute basic metrics
    correct = np.sum(filtered_preds == filtered_labels)
    total = len(filtered_labels)
    accuracy = float(correct / total) if total > 0 else 0.0

    # Decode tokens to characters for better reporting
    try:
        token_to_char = {t: tokenizer.decode([t]) for t in unique_tokens}
    except Exception as e:
        logger.warning(f"Could not decode tokens to characters: {e}")
        token_to_char = {t: str(t) for t in unique_tokens}

    # Get per-class metrics if enough data
    per_class_metrics = {}
    if len(filtered_labels) >= 10:
        try:
            # Convert to class indices
            class_indices = {t: i for i, t in enumerate(sorted(unique_tokens))}
            class_names = [token_to_char.get(t, str(t)) for t in sorted(unique_tokens)]

            # Map token IDs to class indices
            pred_classes = np.array([class_indices[p] for p in filtered_preds])
            label_classes = np.array([class_indices[l] for l in filtered_labels])

            # Compute confusion matrix and classification report
            cm = confusion_matrix(label_classes, pred_classes)
            report = classification_report(label_classes, pred_classes,
                                           target_names=class_names,
                                           output_dict=True)

            # Structure the results
            per_class_metrics = {
                "confusion_matrix": cm.tolist(),
                "class_names": class_names,
                "token_to_char": {str(t): c for t, c in token_to_char.items()},
                "classification_report": report
            }
        except Exception as e:
            logger.warning(f"Error computing per-class metrics: {e}")

    return {
        "accuracy": accuracy,
        "correct": int(correct),
        "total": int(total),
        "token_counts": {str(t): c for t, c in token_counts.items()},
        "per_class": per_class_metrics
    }


def train_model(args: argparse.Namespace, train_dataset: Dataset, val_dataset: Dataset,
                tokenizer, model: AutoModelForCausalLM) -> Tuple[AutoModelForCausalLM, Dict[str, float]]:
    """
    Train the model on the given datasets.

    Args:
        args: Command line arguments
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: The tokenizer
        model: The model to train

    Returns:
        tuple: trained model and metrics
    """
    # Prepare directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb if requested
    if args.use_wandb:
        if args.wandb_run_name is None:
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{args.model_name_or_path.split('/')[-1]}_{args.prediction_mode}_{time_str}"
        else:
            run_name = args.wandb_run_name

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags + [args.prediction_mode,
                                    "length-generalization" if "gpt2" in args.model_name_or_path.lower() else "standard"],
            config={
                "model_name": args.model_name_or_path,
                "prediction_mode": args.prediction_mode,
                "max_train_length": args.max_train_length,
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "epochs": args.num_train_epochs,
                "seed": args.seed,
            }
        )
        # Log model architecture
        wandb.watch(model)

    # Set up training arguments
    training_args = TrainingArguments(
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
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,  # Higher accuracy is better
        seed=args.seed,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="wandb" if args.use_wandb else None,  # Use wandb instead of tensorboard
    )

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model and tokenizer
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Model and tokenizer saved to {final_model_path}")

    # Evaluate on validation set
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Log final metrics to wandb
    if args.use_wandb:
        wandb.log(metrics)

    # Return metrics for plotting
    return model, metrics


def create_data_iterator(file_path: Union[str, Path], batch_size: int = 1000) -> Iterator[List[str]]:
    """
    Create a memory-efficient iterator for large datasets.

    Args:
        file_path: Path to the data file
        batch_size: Number of examples to yield at once

    Yields:
        Batches of examples
    """

    def generate_batches():
        with open(file_path, 'r') as f:
            batch = []
            for line in f:
                batch.append(line.strip())
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:  # Don't forget the last batch
                yield batch

    return generate_batches()


def run_inference(args: argparse.Namespace, model: AutoModelForCausalLM, tokenizer,
                  test_dataset: Dataset) -> Dict[str, Any]:
    """
    Run inference on the test dataset.

    Args:
        args: Command line arguments
        model: The trained model
        tokenizer: The tokenizer
        test_dataset: The test dataset

    Returns:
        Dict: Inference results including predictions, targets, and accuracy
    """
    logger.info("Running inference on test dataset...")

    # Create data loader for test dataset
    data_loader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
    )

    # Move model to device and set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_inputs = []

    # Run inference
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Find positions where labels are not IGNORE_INDEX
            valid_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=True)

            # Extract predictions and targets at valid positions
            for i, (batch_idx, pos_idx) in enumerate(zip(*valid_positions)):
                pred = predictions[batch_idx, pos_idx].item()
                target = labels[batch_idx, pos_idx].item()

                all_predictions.append(pred)
                all_targets.append(target)
                all_inputs.append(batch["input_str"][batch_idx])

    # Convert token IDs to NumPy arrays for metrics computation
    pred_array = np.array(all_predictions)
    target_array = np.array(all_targets)

    # Compute detailed metrics
    metrics = compute_detailed_metrics(pred_array, target_array, tokenizer)

    # Convert token IDs back to characters
    pred_chars = [tokenizer.decode([p]) for p in all_predictions]
    target_chars = [tokenizer.decode([t]) for t in all_targets]

    # Calculate accuracy
    correct = metrics["correct"]
    total = metrics["total"]
    accuracy = metrics["accuracy"]

    logger.info(f"Inference results - Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Log to wandb if it's initialized
    if args.use_wandb and wandb.run is not None:
        wandb.log({
            f"test_accuracy_{args.prediction_mode}": accuracy,
            f"test_correct_{args.prediction_mode}": correct,
            f"test_total_{args.prediction_mode}": total,
        })

    # Save detailed results
    results_file = os.path.join(args.output_dir, f"inference_results_{args.prediction_mode}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Prediction mode: {args.prediction_mode}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n\n")

        for i, (inp, pred, target) in enumerate(zip(all_inputs, pred_chars, target_chars)):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Input: {inp}\n")
            f.write(f"Predicted: {pred} (ID: {all_predictions[i]})\n")
            f.write(f"Target: {target} (ID: {all_targets[i]})\n")
            f.write("Correct: " + ("✓" if pred == target else "✗") + "\n\n")

    logger.info(f"Detailed inference results saved to {results_file}")

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "accuracy": accuracy,
        "pred_chars": pred_chars,
        "target_chars": target_chars,
        "metrics": metrics,
    }


def generate_plots(args: argparse.Namespace, metrics: Dict[str, float],
                   inference_results: Optional[Dict[str, Any]] = None) -> None:
    """
    Generate and save plots for training and evaluation metrics.

    Args:
        args: Command line arguments
        metrics: Training and evaluation metrics
        inference_results: Results from inference
    """
    logger.info("Generating plots...")

    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Try to load training history from tensorboard logs
    try:
        from tensorboard.backend.event_processing import event_accumulator

        log_dir = os.path.join(args.output_dir, "logs")
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()

        # Plot training and validation loss
        if "train/loss" in ea.scalars.Keys():
            # Extract training metrics
            train_loss = [(s.step, s.value) for s in ea.Scalars("train/loss")]
            steps, losses = zip(*train_loss)

            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label="Training Loss")

            if "eval/loss" in ea.scalars.Keys():
                eval_loss = [(s.step, s.value) for s in ea.Scalars("eval/loss")]
                eval_steps, eval_losses = zip(*eval_loss)
                plt.plot(eval_steps, eval_losses, label="Validation Loss", marker='o')

            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.title(f"Training and Validation Loss ({args.prediction_mode} prediction)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"loss_plot_{args.prediction_mode}.png"))
            plt.close()

            # Plot accuracy
            if "eval/accuracy" in ea.scalars.Keys():
                eval_acc = [(s.step, s.value) for s in ea.Scalars("eval/accuracy")]
                acc_steps, accuracies = zip(*eval_acc)

                plt.figure(figsize=(10, 6))
                plt.plot(acc_steps, accuracies, label="Validation Accuracy", marker='o')
                plt.xlabel("Training Steps")
                plt.ylabel("Accuracy")
                plt.title(f"Validation Accuracy ({args.prediction_mode} prediction)")
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.05)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"accuracy_plot_{args.prediction_mode}.png"))
                plt.close()

    except Exception as e:
        logger.warning(f"Failed to generate training plots: {e}")

    # Plot confusion matrix for inference results
    if inference_results and "pred_chars" in inference_results and "target_chars" in inference_results:
        try:
            preds = inference_results["pred_chars"]
            targets = inference_results["target_chars"]

            # Get unique classes
            all_classes = sorted(list(set(preds + targets)))

            # Compute confusion matrix
            cm = confusion_matrix(targets, preds, labels=all_classes)

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=all_classes, yticklabels=all_classes)
            plt.xlabel("Predicted")
            plt.title(f"Confusion Matrix ({args.prediction_mode} prediction)")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{args.prediction_mode}.png"))
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix plot: {e}")

    # If we have per-class metrics, plot precision, recall, and F1
    if (inference_results and "metrics" in inference_results and
            "per_class" in inference_results["metrics"] and
            "classification_report" in inference_results["metrics"]["per_class"]):
        try:
            report = inference_results["metrics"]["per_class"]["classification_report"]

            # Extract data for digits only
            digit_classes = [c for c in report.keys() if c.isdigit() or (c in
                                                                         inference_results["metrics"]["per_class"][
                                                                             "token_to_char"].values()
                                                                         and c.isdigit())]

            if digit_classes:
                metrics_data = {
                    "precision": [report[c]["precision"] for c in digit_classes],
                    "recall": [report[c]["recall"] for c in digit_classes],
                    "f1-score": [report[c]["f1-score"] for c in digit_classes]
                }

                # Plot metrics by class
                plt.figure(figsize=(12, 6))
                x = np.arange(len(digit_classes))
                width = 0.25

                plt.bar(x - width, metrics_data["precision"], width, label="Precision")
                plt.bar(x, metrics_data["recall"], width, label="Recall")
                plt.bar(x + width, metrics_data["f1-score"], width, label="F1 Score")

                plt.xlabel("Digit")
                plt.ylabel("Score")
                plt.title(f"Per-Class Metrics ({args.prediction_mode} prediction)")
                plt.xticks(x, digit_classes)
                plt.legend()
                plt.ylim(0, 1.1)
                plt.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"per_class_metrics_{args.prediction_mode}.png"))
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate per-class metrics plot: {e}")


def save_config(args: argparse.Namespace) -> None:
    """
    Save configuration to a file for reproducibility.

    Args:
        args: Command line arguments
    """
    import json
    config_file = os.path.join(args.output_dir, "config.json")

    # Convert args to a dictionary
    config = vars(args)

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {config_file}")


def main():
    """Main function to run the fine-tuning script with length generalization for GPT2."""
    # Parse arguments
    args = parse_args()

    # Get the max_train_length from args
    max_train_length = args.max_train_length

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration for reproducibility
    save_config(args)

    # Log arguments
    logger.info(f"Running with arguments: {args}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_name or args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        use_fast=True,
    )

    # Set pad token to eos token if needed
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(args, tokenizer)

    # Create custom collator for GPT-2 models
    collator = None
    if 'gpt2' in args.model_name_or_path.lower():
        collator = LengthGeneralizableCollator(
            pad_id=tokenizer.pad_token_id,
            max_train_length=max_train_length
        )

    if args.do_train:
        # Load model
        logger.info(f"Loading model from {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

        # Resize token embeddings if needed
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            logger.info(
                f"Resizing token embeddings from {model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # Apply length generalization for GPT-2 models
        if 'gpt2' in args.model_name_or_path.lower():
            logger.info(f"Modifying GPT-2 model for length generalization (max_train_length={max_train_length})")
            model = modify_model_for_length_generalization(model, max_train_length=max_train_length)

        # Set up training arguments
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
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            report_to="tensorboard",
        )

        # Set up trainer with custom collator if using GPT-2
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=collator,  # This will be None for non-GPT2 models
        )

        # Train the model
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save model and tokenizer
        final_model_path = os.path.join(args.output_dir, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # For GPT-2 models, save the max_train_length in the config
        if 'gpt2' in args.model_name_or_path.lower():
            # Save the max_train_length in the model config
            import json
            config_path = os.path.join(final_model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)

                config['max_train_length'] = max_train_length

                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

        logger.info(f"Model and tokenizer saved to {final_model_path}")

        # Evaluate on validation set
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if args.do_eval or args.do_inference:
        # Load model for inference or evaluation
        if not args.do_train:
            checkpoint_path = args.checkpoint_path or os.path.join(args.output_dir, "final_model")
            logger.info(f"Loading model from {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

            # Check if this is a GPT-2 model and apply length generalization
            if 'gpt2' in args.model_name_or_path.lower():
                # Try to get max_train_length from config, or use default
                try:
                    import json
                    config_path = os.path.join(checkpoint_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)

                        if 'max_train_length' in config:
                            max_train_length = config['max_train_length']
                except Exception as e:
                    logger.warning(f"Could not read max_train_length from config: {e}")

                logger.info(f"Applying length generalization with max_train_length={max_train_length}")
                model = modify_model_for_length_generalization(model, max_train_length=max_train_length)

    inference_results = None

    if args.do_inference:
        # Run inference on test dataset
        inference_results = run_inference(args, model, tokenizer, test_dataset)

    if args.generate_plots:
        # Generate plots
        metrics = {}  # Will be populated from training logs
        generate_plots(args, metrics, inference_results)

        # Finish wandb run if it was initialized
    if args.use_wandb and wandb.run is not None:
        wandb.finish()

    logger.info("Done!")


if __name__ == "__main__":
    main()