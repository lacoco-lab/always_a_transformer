from __future__ import annotations

import random
import argparse
import jsonlines
from pathlib import Path
from itertools import product
from typing import Dict, List, Sequence

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from retrieval.prompt_constants import *


############################################
# Dataset loading
############################################

def load_dataset(data_path: Path, task_key: str, has_separator: bool) -> List[Dict]:
    """Load JSONL records and organise them per induction setting."""    
    records: List[Dict] = []
    data_path = Path(data_path)
    with jsonlines.open(data_path, "r") as reader:
        records.extend(reader)
    # Build buckets per setting
    if data_path.parent.name == 'unique':
        assert task_key in SETTINGS[:2], f"Invalid setting {task_key} for unique dataset"
    else:
        assert task_key in SETTINGS[2:], f"Invalid setting {task_key} for non-unique dataset"
    
    dataset = []
    for rec in records:
        if has_separator:
            input = rec['input_with_space']
        else:
            input = rec['input_no_space']
        dataset.append({
            'input': input, 
            'target': rec[task_key],
        })
    return dataset

def atomise(text: str) -> str:
    """Return a *space‑prefixed, space‑separated* version of *text*.

    Example: "abc|d" → " a b c | d"  (note the leading space)"""
    if not text:
        return ""
    return " " + " ".join(text)

def map_variant_name_to_tuple(prompt_variant):
    # Sample input : "sep|small|math_rule" , "nosep|small|math_rule" 
    separator, few_shot, instruction = prompt_variant.split('|')
    return (separator.startswith('sep'), few_shot, instruction)

############################################
# Prompt building
############################################

class PromptBuilder:
    """Construct few shot prompts according to *PromptVariant*."""

    def __init__(
        self,
        shots: int,
        variant: str,
        task_key: str,
        test_data_path: Path,
        fewshot_data_path: Path,
        tokenizer: AutoTokenizer,
    ) -> None:

        self.shots = shots
        self.task_key = task_key
        self.tokenizer = tokenizer

        has_separator, few_shot_type, insn_type = map_variant_name_to_tuple(variant)
        self.dataset = load_dataset(test_data_path, task_key=task_key, has_separator=has_separator)
        if few_shot_type == 'small':
            self.few_shot_dataset = load_dataset(fewshot_data_path, task_key=task_key, has_separator=has_separator)
        else: 
            self.few_shot_dataset = self.dataset.copy()
        self.context_template = insn_type

    def _char_ids(self, text: str) -> List[int]:
        """Return *one token-id per visible character* (no spaces inserted)."""
        ids: List[int] = []
        for ch in text:
            ids.extend(self.tokenizer.encode(ch, add_special_tokens=False))
        return ids

    def _make_example_line(self, input_str: str, target: str | None) -> str:
        """Return an *atomised* example line (with or without target)."""
        prefix = atomise(input_str)
        return f"{prefix}:{'' if target is None else ' ' + target}"

    def create_few_shot(self, curr_record: Dict) -> List[str]:
        few_shot_pool = [rec for rec in self.few_shot_dataset if rec != curr_record]
        few_shot_examples = random.sample(few_shot_pool, k=self.shots)

        # Reserve one for explanation
        explain_example, few_shot_examples = few_shot_examples[0], few_shot_examples[1:]

        # Add the few‑shot examples
        examples = [self._make_example_line(ex["input"], ex["target"]) for ex in few_shot_examples]

        # Add the current query
        examples.append(self._make_example_line(curr_record["input"], None))
        return '\n\n'.join(examples), explain_example

    def build_prompt(self, curr_record: Dict) -> str:
        """Return a fully instantiated prompt string"""
        task = TASKS[self.task_key]
        examples_block, explain_example = self.create_few_shot(curr_record)
        explaination_block = self._make_explanation(explain_example, task)

        # do not put the examples right away
        template_base = TEMPLATES[self.context_template].format(
            rule_simple=task.rule_simple,
            rule_math=task.rule_math,
            examples=examples_block,
            explanation=explaination_block
        )
        ids = self.tokenizer.encode(template_base, add_special_tokens=False)
        return template_base, ids

    def _make_explanation(self, ex: Dict, task: TaskRule) -> str:
        """
        Turn one solved example into a short, human-readable explanation block.
        It highlights the query token, states how many times it appears,
        references the rule, and shows the answer that follows the rule.
        """
        # Example line already in the ‘context|query: target’ format
        solved_line = self._make_example_line(ex["input"], ex["target"])

        context, rest = ex["input"].split("|", 1)
        query = rest.strip()
        occurrences = context.strip().split().count(query)

        return (
            f"{solved_line}\n"
            f"• query token = “{query}” (everything after “|”)\n"
            f"• it occurs {occurrences} time(s) in the context\n"
            f"• according to the rule → answer = “{ex['target']}”"
        )

############################################
# Evaluation / prompt generation class
############################################

class ModelEvaluator:
    """Handle vLLM based inference OR prompt only construction."""

    def __init__(
        self,
        model_id: str,
        batch_size: int,
        tp: int,
        gpu_mem: float,
        temperature: float,
        seed: int
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.temperature = temperature
        self.seed = seed

        print(f"→ Loading tokenizer for {model_id} …", flush=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("→ Initialising vLLM engine …", flush=True)
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            seed=seed,
            skip_tokenizer_init=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=MAX_COMPLETION_TOKENS,
            temperature=temperature,
            stop=[STOP_SEQUENCE],
        )

    # ---------------- Evaluation ---------------- #

    def run(
        self,
        prompt_packs: Sequence[Dict],
    ) -> List[Dict]:

        results: List[Dict] = []
        for i in range(0, len(prompt_packs), self.batch_size):
            batch = prompt_packs[i : i + self.batch_size]
            outs = self.llm.generate(
                prompt_token_ids=[b["ids"] for b in batch],
                sampling_params=self.sampling_params,
            )
            for generated_output, current_entry in zip(outs, batch):
                generated_text = self.tokenizer.decode(generated_output.outputs[0].token_ids).strip()
                try:
                    prediction = generated_text.strip()[0]
                except Exception:
                    prediction = "failed"
                results.append({
                    "prediction": prediction,
                    "full_output": generated_text,
                    "input": current_entry["rec"]["input"],
                    "target": current_entry["rec"]["target"],
                    "prompt": current_entry["prompt"]
                })
        return results

    def cleanup(self):
        self.llm.llm_engine.__del__()


def unique_runs():
    # All settings UR, UL 
    dataset_types = ['UR', 'UL']
    dataset_paths = [f for f in Path(UNIQUE_PATH).iterdir() if f.is_file() and f.name.__contains__('1500')]
    return dataset_types, dataset_paths, UNIQUE_FEWSHOT

def non_unique_runs():
    # Cartesian product of all prompt variants
    # All settings NRLast, NLLast, NRFirst, NLFirst
    dataset_types = ['NRLast', 'NLLast', 'NRFirst', 'NLFirst']
    dataset_paths = [f for f in Path(NON_UNIQUE_PATH).iterdir() if f.is_file() and f.name.__contains__('1500')]
    
    return dataset_types, dataset_paths, NON_UNIQUE_FEWSHOT

############################################
def run_group(is_unique, tokenizer, model, args):
    # prompt_variants = list(product(['sep', 'nosep'],
    #                                PROMPT_VARIANTS['few_shot'],
    #                                PROMPT_VARIANTS['template']))

    # prompt_variants = ['|'.join(variant) for variant in prompt_variants]
    prompt_variants = ['sep|small|math_rule', 'sep|small|simple_rule', 'sep|small|simple_rule_explained', 'sep|small|math_rule_explained']
    dataset_types, dataset_paths, few_shot_path = unique_runs() if is_unique else non_unique_runs()

    # ------------------------------------------------------------------ #
    evaluator = ModelEvaluator(
        model_id=MODELS[model],
        batch_size=args.batch,
        tp=args.tp,
        gpu_mem=args.gpu_mem,
        temperature=args.temperature,
        seed=args.seed
    )

    for variant in prompt_variants: 
        for dataset_type in dataset_types:
            for data_path in dataset_paths:
            
                builder = PromptBuilder(
                        shots=args.shots,
                        variant=variant,
                        task_key=dataset_type,
                        test_data_path=data_path,
                        fewshot_data_path=few_shot_path,
                        tokenizer=tokenizer,
                    )
                # ------------------------------------------------------------------ #
                # All the middle level directories
                middle_dirs = data_path.parts[1:-1]
                out_path = Path(args.outdir, *middle_dirs, dataset_type,
                                model, variant.replace('|', ''), data_path.name)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if out_path.exists():
                    print(f"❌ {out_path} already exists, skipping")
                    continue
                prompt_packs = []
                for rec in builder.dataset:
                    prompt_txt, ids = builder.build_prompt(rec)
                    prompt_packs.append({"prompt": prompt_txt, "ids": ids, "rec": rec})

                results = evaluator.run(prompt_packs)
                with jsonlines.open(out_path, "w") as writer:
                    writer.write_all(results)

                print(f"✅ {out_path} records)")

############################################
# CLI
############################################
def main() -> None:
    ap = argparse.ArgumentParser("Synthetic-induction eval / prompt builder")

    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--gpu_mem", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--unique", choices=['unique', 'nonunique'], default="unique")
    ap.add_argument("--model", choices=[m for m in MODELS.keys()], default="llama-7b")

    args = ap.parse_args()
    random.seed(args.seed)
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model])
    # tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
    if args.unique == 'unique':
        run_group(True, tokenizer, model, args)
    else:
        run_group(False, tokenizer, model, args)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()