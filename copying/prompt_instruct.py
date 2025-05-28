"""copy_prompt_builder.py

Builds prompts for the *Copy* probing task (UF/UB or NF/NB).

Each example line has the form::

    <bos> a b c d <sep> a b c d <eos>

Where
* ``<bos>`` begins the sequence
* characters of the *input* are separated by spaces so that every character is
  kept as an independent token by the tokenizer
* ``<sep>`` ends the copy source; the colon separates *source* and *target*
* the *target* is either the *forward* copy (UF/NF) or the *backward* copy
  (UB/NB)

For the *query* example (the one we want the model to answer) we omit the
*target* so the prompt line ends right after ``<sep>:``.  A full prompt is a
block of *k* few‑shot lines followed by the query line.
"""
from __future__ import annotations

import random
import jsonlines
from pathlib import Path
from typing import Dict, List, Sequence

from dataclasses import dataclass
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from retrieval.prompt_constants import PROMPT_VARIANTS

# ---------------- Constants ----------------- #
# A single prompt variant will take one of the following forms
# PROMPT_VARIANTS = ['bare', 'obey', 'hint']
PROMPT_VARIANTS = ['obey', 'hint']
# Change this to something more appropriate
# STOP_SEQUENCE_LIST = ["\n\n", "<eos>"] # not needed for instruct
# Change the Max completion tokens
MAX_INSTRUCT_TOKENS = 400

UNIQUE_PATH = 'datasets/copying/unique'
UNIQUE_FEWSHOT = 'datasets/copying/unique/unique_N1500_L5_seed123.jsonl'

NON_UNIQUE_PATH = 'datasets/copying/nonunique'
NON_UNIQUE_FEWSHOT = 'datasets/copying/nonunique/nonunique_N1500_L5_seed123.jsonl'


@dataclass(frozen=True)
class TaskRule:
    key: str
    rule_simple: str
    rule_hint: str


SYSTEM_PROMPT = ('You are a very careful and precise assistant. '
                 'You always follow the instructions without caring about what the task '
                 'and how it is formulated. You always solve tasks yourself. You never generate code.')

FORMAT_PROMPT = ('Put the answer between <target> and </target> tags. '
                 'Start your response with the answer and no explanation is needed for this simple task.')

# Forward‑copy (F) and backward‑copy (B) rules for the two token‑uniqueness
# regimes.  The wording is identical for unique / non‑unique because the
# operation (copy or reverse) does not depend on repetition once the input is
# fixed.
TASKS: Dict[str, TaskRule] = {
    # ―― Unique tokens ――――――――――――――――――――――――――――――――――――――――――――――――
    "UF": TaskRule(
        key="UF",
        rule_simple="The output is exactly the same sequence as the input.",
        rule_hint="in every example the output repeats the input unchanged."
    ),
    "UB": TaskRule(
        key="UB",
        rule_simple="The output is the input sequence written in reverse order.",
        rule_hint='in every example the output is the input read backwards.'
    ),
    # ―― Non‑unique tokens ――――――――――――――――――――――――――――――――――――――――――――
    "NF": TaskRule(
        key="NF",
        rule_simple="The output is exactly the same sequence as the input.",
        rule_hint='in every example the output repeats the input unchanged.'
    ),
    "NB": TaskRule(
        key="NB",
        rule_simple="The output is the input sequence written in reverse order.",
        rule_hint='in every example the output is the input read backwards.'
    ),
}

# ---------------------------------------------------------------------------
# Prompt templates (absolutely no imperatives!)
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, str] = {
    # 1) Bare – pattern exposure only --------------------------------------
    "bare": "{examples}",
    # 2) One‑line natural rule --------------------------------------------
    "obey": "Here are some samples of the format <bos> input : output <eos> where the rule being followed is - {rule_simple}\n\n{examples}",
    # 3) Reverse‑specific hint --------------------------------------------
    "hint": "Here are some examples where {rule_hint}\n\n{examples}"
}

MODELS: Dict[str, str] = {
    'llama3_8B': ("/local/common_models/Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
    'llama3_70B': ("/local/common_models/Llama-3.1-70B", "/scratch/common_models/Llama-3.1-70B"),
    'qwen2.5_7B': ("/local/common_models/Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
    'qwen2.5_32B': ("/local/common_models/Qwen2.5-32B", "Qwen/Qwen2.5-32B"),
    'llama3_8B_instruct': ("/local/common_models/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    'llama3_70B_instruct': ("/local/common_models/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"),
    'qwen2.5_7B_instruct': ("/local/common_models/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    'qwen2.5_32B_instruct': ("/local/common_models/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-32B-Instruct"),
}

# ---------------- Dataset loading ---------------- #

SETTINGS = ["UF", "UB", "NF", "NB"]


def load_dataset(data_path: Path, task_key: str) -> List[Dict]:
    """Load *copy* task JSON‑Lines and return a list of dicts.

    Parameters
    ----------
    data_path
        Path to a JSONL file produced by *CopyDatasetGenerator*.
    task_key
        One of ``UF``, ``UB``, ``NF``, ``NB`` selecting the *target* column.
    """
    if task_key not in SETTINGS:
        raise ValueError(f"task_key must be one of {SETTINGS!r}")

    records: List[Dict] = []
    with jsonlines.open(data_path, "r") as reader:
        for rec in reader:
            records.append({
                "input": rec["input"],
                "target": rec[task_key],
            })
    return records


# ---------------- Prompt helpers ---------------- #

def _space_chars(text: str) -> str:
    """Return characters of *text* separated by single spaces."""
    return " ".join(text)


SETTINGS = ["UF", "UB", "NF", "NB"]


def load_dataset(data_path: Path, task_key: str) -> List[Dict]:
    """Load *copy* task JSON‑Lines and return a list of dicts.

    Parameters
    ----------
    data_path
        Path to a JSONL file produced by *CopyDatasetGenerator*.
    task_key
        One of ``UF``, ``UB``, ``NF``, ``NB`` selecting the *target* column.
    """
    if task_key not in SETTINGS:
        raise ValueError(f"task_key must be one of {SETTINGS!r}")

    records: List[Dict] = []
    with jsonlines.open(data_path, "r") as reader:
        for rec in reader:
            records.append({
                "input": rec["input"],
                "target": rec[task_key],
            })
    return records


# ---------------- Prompt helpers ---------------- #

class PromptBuilderCopy:
    """Create few‑shot prompts for the Copy task.

    Each *example line* is built like::

        <bos> a b c d <sep>: a b c d

    For the query (target unknown) the part after the colon is omitted.
    """

    def __init__(
            self,
            shots: int,
            variant: str,
            task_key: str,
            test_data_path: Path,
            fewshot_data_path: Path | None,
            tokenizer: AutoTokenizer,
            rng: random.Random | None = None,
    ) -> None:
        self.shots = shots
        self.task_obj = TASKS[task_key]
        self.tokenizer = tokenizer
        self.rng = rng or random.Random()
        self.prompt_type = variant
        self.dataset = load_dataset(test_data_path, task_key)
        if fewshot_data_path is None:
            # allow re‑using the test set as few‑shot pool
            self.fewshot_dataset = self.dataset.copy()
        else:
            self.fewshot_dataset = load_dataset(fewshot_data_path, task_key)

    # -------------- Single‑line builder -------------- #

    def _make_example_line(self, input_str: str, target: str | None) -> str:
        """Return a prompt line with (or without) *target*."""
        src = f"<bos> {_space_chars(input_str)} :"
        return src if target is None else f"{src} {_space_chars(target)} <eos>"

    # -------------- Few‑shot block -------------- #

    def _few_shot_block(self, curr_record: Dict) -> str:
        """Sample *k* few‑shot examples (excluding *curr_record*)."""
        pool = [rec for rec in self.fewshot_dataset if rec != curr_record]
        examples = self.rng.sample(pool, k=self.shots)
        lines = [self._make_example_line(ex["input"], ex["target"]) for ex in examples]
        # add the *query* without target
        lines.append(self._make_example_line(curr_record["input"], None))
        return "\n".join(lines)

    def build_prompt(self, curr_record: Dict) -> tuple[str, List[int]]:
        """Return the textual prompt **and** its token‑ids."""
        # Create the examples along with the input
        examples = self._few_shot_block(curr_record)
        template_base = TEMPLATES[self.prompt_type].format(
            rule_simple=self.task_obj.rule_simple,
            rule_hint=self.task_obj.rule_hint,
            examples=examples,
        )
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template_base + "\n\n" + FORMAT_PROMPT},
        ]
        prompt_chat = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        prompt_tokens = self.tokenizer.tokenize(prompt_chat)
        ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
        
        return prompt_chat, ids


############################################
# Evaluation / prompt generation class
############################################

class ModelEvaluator:
    """Handle vLLM based inference"""

    def __init__(
            self,
            model_path: str,
            tokenizer: AutoTokenizer,
            batch_size: int,
            tp: int,
            gpu_mem: float,
            temperature: float,
            seed: int
    ) -> None:
        self.seed = seed
        self.tokenizer = tokenizer
        self.model_id = model_path
        self.batch_size = batch_size
        self.temperature = temperature
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("→ Initialising vLLM engine …", flush=True)

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            seed=seed,
            skip_tokenizer_init=True,
            disable_custom_all_reduce=True,  # <-- key line
        )
        self.sampling_params = SamplingParams(
            max_tokens=MAX_INSTRUCT_TOKENS,
            temperature=temperature,
            # stop=STOP_SEQUENCE_LIST,
        )
    
    def parse_response(self, response_text):
        ans_begin = response_text.find("<target>")
        ans_end = response_text.find("</target>")
        answer = response_text[ans_begin + len("<target>"): ans_end]
        return "".join(answer.strip().split())
    # ---------------- Evaluation ---------------- #

    def run(
            self,
            prompt_packs: Sequence[Dict],
    ) -> List[Dict]:

        results: List[Dict] = []
        for i in range(0, len(prompt_packs), self.batch_size):
            batch = prompt_packs[i: i + self.batch_size]
            outs = self.llm.generate(
                prompt_token_ids=[b["ids"] for b in batch],
                sampling_params=self.sampling_params,
            )
            for generated_output, current_entry in zip(outs, batch):
                generated_text = self.tokenizer.decode(generated_output.outputs[0].token_ids).strip()
                try:
                    prediction = self.parse_response(generated_text)
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


def unique_runs():
    dataset_types = ['UF', 'UB']
    dataset_paths = [f for f in Path(UNIQUE_PATH).iterdir() if f.is_file() and f.name.__contains__('1500')]
    return dataset_types, dataset_paths, UNIQUE_FEWSHOT


def non_unique_runs():
    dataset_types = ['NF', 'NB']
    dataset_paths = [f for f in Path(NON_UNIQUE_PATH).iterdir() if f.is_file() and f.name.__contains__('1500')]
    return dataset_types, dataset_paths, NON_UNIQUE_FEWSHOT


############################################
def run_group(is_unique, tokenizer, model_path, model_type, args):
    dataset_types, dataset_paths, few_shot_path = unique_runs() if is_unique else non_unique_runs()

    # ------------------------------------------------------------------ #
    evaluator = ModelEvaluator(
        model_path=model_path,
        tokenizer=tokenizer,
        batch_size=args.batch,
        tp=args.tp,
        gpu_mem=args.gpu_mem,
        temperature=args.temperature,
        seed=args.seed
    )

    for variant in PROMPT_VARIANTS:
        for dataset_type in dataset_types:
            for data_path in dataset_paths:

                builder = PromptBuilderCopy(
                    shots=args.shots,
                    variant=variant,
                    task_key=dataset_type,
                    test_data_path=data_path,
                    fewshot_data_path=few_shot_path,
                    tokenizer=tokenizer,
                    rng=random.Random(args.seed))
                # ------------------------------------------------------------------ #
                # All the middle level directories
                middle_dirs = data_path.parts[1:-1]
                out_path = Path(args.outdir, *middle_dirs, dataset_type,
                                model_type, variant, data_path.name)
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
    import argparse
    ap = argparse.ArgumentParser("Synthetic-copying eval / prompt builder")

    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--scratch", choices=['scratch', 'local'], default='local')
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--gpu_mem", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--unique", choices=['unique', 'nonunique'], default="unique")
    ap.add_argument("--model", choices=[m for m in MODELS.keys()], default="llama-7b")

    args = ap.parse_args()
    random.seed(args.seed)
    model_type = args.model
    local_model_path, hf_model_path = MODELS[model_type]
    if args.scratch == 'local':
        print(f"Loading from local path {local_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, add_prefix_space=True)
        correct_path = local_model_path
    else:
        print(f"Loading from HF model path {hf_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path, add_prefix_space=True)
        correct_path = hf_model_path

    if args.unique == 'unique':
        run_group(True, tokenizer, correct_path, model_type, args)
    else:
        run_group(False, tokenizer, correct_path, model_type, args)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()