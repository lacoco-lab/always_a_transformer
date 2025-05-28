#!/usr/bin/env python
# prompt_program.py
from __future__ import annotations

import argparse, random, tqdm
import jsonlines
from pathlib import Path
from typing import Dict, List, Any

from banks.registries import DirectoryPromptRegistry
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


###############################################################################
#  MODELS ######################################################################
###############################################################################
MODELS: Dict[str, str] = {
    # --- Llama ----------------------------------------------------------------
    'llama3_8B'             : "/local/common_models/Llama-3.1-8B",
    'llama3_8B_instruct'    : "/local/common_models/Llama-3.1-8B-Instruct",
    'llama3_8B_instruct_hf' : "meta-llama/Llama-3.1-8B-Instruct",
    'llama3_70B'            : "/local/common_models/Llama-3.1-70B",
    'llama3_70B_instruct'   : "/local/common_models/Llama-3.3-70B-Instruct",
    # --- Qwen -----------------------------------------------------------------
    'qwen2.5_7B'            : "/local/common_models/Qwen2.5-7B",
    'qwen2.5_32B'           : "/local/common_models/Qwen2.5-32B",
    'qwen2.5_7B_instruct'   : "/local/common_models/Qwen2.5-7B",
    'qwen2.5_32B_instruct'  : "/local/common_models/Qwen2.5-32B-Instruct",
}

###############################################################################
#  DATASETS & TASKS ############################################################
###############################################################################
DATASETS = {
    # ----- original “code-assist” setup (few-shot) ----------------------------
    "codeassist": dict(
        path="datasets/realistic/codeassist",
        tasks=["revert", "cherrypick"],
        few_shot=True,
    ),
    # ----- echo / retrieval style --------------------------------------------
    "arxiv": dict(
        path="datasets/realistic/arxiv",
        tasks=["echo"],             # single task
        few_shot=False,
    ),
    "loremipsum": dict(
        path="datasets/realistic/lorem_ipsum",
        tasks=["echo"],
        few_shot=False,
    ),
}

# Few-shot scaffolding for the code-assist dataset
FEW_SHOT_CREATE: Dict[str, List[str]] = {
    'revert'     : ["=== NEW HISTORY ===", "dummy", "=== ANSWER ===", "<start>", "dummy", "<end>"],
    'cherrypick' : ["=== NEW HISTORY ===", "dummy", "=== ANSWER ===", "<start>", "dummy", "<end>"],
}

###############################################################################
#  HELPERS #####################################################################
###############################################################################
def save_to_jsonl(out_path: Path, records: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(out_path, mode="w") as w:
        w.write_all(records)
    print(f"✔ Saved → {out_path}")


def load_dataset(file_path: Path, task_key: str, few_shot: bool) -> List[Dict[str, str]]:
    """
    Unifies dataset loading for both modes:
    * few-shot   → expect `snippet` and `task_key` columns
    * echo/other → expect `input`
    """
    rows: list[dict] = []
    with jsonlines.open(file_path) as r:
        for rec in r:
            if few_shot:
                rows.append(
                    dict(
                        input=rec["snippet"].strip(),
                        target=rec[task_key].strip(),
                    )
                )
            else:
                text = rec["input"].strip()
                rows.append(dict(input=text, target=text))  # echo target = input
    return rows


def get_prompts(prompt_dir: Path, task_key: str, config: str, is_instruct: bool):
    """
    Fetch {task, system} prompts from the registry, with automatic *_instruct fallback.
    """
    registry = DirectoryPromptRegistry(prompt_dir, force_reindex=True)
    system_prompt = registry.get(name="sys")

    if is_instruct:
        task_prompt = registry.get(name=f"task_{config}_instruct")
    else:
        task_prompt = registry.get(name=f"task_{config}")
    return task_prompt, system_prompt


###############################################################################
#  PROMPT BUILDER ##############################################################
###############################################################################
class PromptBuilder:
    """
    →   For code-assist   : builds few-shot, task-specific prompts.
    →   For echo datasets : just wraps the single input.
    Supports chat template for *-instruct* models.
    """

    def __init__(
        self,
        *,
        shots: int,
        task_key: str,
        dataset_rows: List[Dict[str, str]],
        prompt_dir: Path,
        tokenizer: AutoTokenizer,
        config: str,
        few_shot_mode: bool,
        is_instruct: bool,
    ) -> None:
        self.shots = shots
        self.task_key = task_key
        self.rows = dataset_rows
        self.tokenizer = tokenizer
        self.few_shot_mode = few_shot_mode
        self.is_instruct = is_instruct
        self.task_prompt, self.sys_prompt = get_prompts(
            prompt_dir, task_key, config, is_instruct
        )

    # ---------- helpers ------------------------------------------------------
    def _few_shot_line(self, inp: str, tgt: str) -> str:
        """Return one formatted example for few-shot prompting."""
        pattern = FEW_SHOT_CREATE[self.task_key].copy()
        pattern[1] = inp
        pattern[4] = tgt
        return "\n".join(pattern)

    # ---------- public -------------------------------------------------------
    def build(self, curr_row: Dict[str, str]) -> List[int]:
        """
        Returns a list[int] = prompt token IDs (already chat-templated if needed).
        """
        # ---- 1) user content -------------------------------------------------
        if self.few_shot_mode:
            pool = [r for r in self.rows if r is not curr_row]
            example_rows = random.sample(pool, k=self.shots)
            examples_block = "\n\n".join(
                self._few_shot_line(r["input"], r["target"]) for r in example_rows
            )
            user_content = self.task_prompt.text(
                {"few_shot_block": examples_block, "snippet": curr_row["input"]}
            )
        else:  # echo data
            user_content = self.task_prompt.text({"input": curr_row["input"]})

        # ---- 2) chat vs plain -----------------------------------------------
        if self.is_instruct:
            chat = [{"role": "user", "content": user_content}]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True
            )

        return self.tokenizer.encode(user_content, add_special_tokens=False)


###############################################################################
#  EVALUATION WRAPPER ##########################################################
###############################################################################
class ModelEvaluator:
    def __init__(
        self,
        *,
        model_path: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        tensor_parallel: int,
    ):
        print(f"Loading tokenizer ⤵ {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_prefix_space=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Booting vLLM ⤵ {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel,
            seed=seed,
            skip_tokenizer_init=True,
        )
        self.params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    # ----- util --------------------------------------------------------------
    @staticmethod
    def _strip_between_markers(text: str, start="<start>", end="<end>") -> str:
        begin, stop = text.find(start), text.find(end)
        return text[begin + len(start) : stop].strip() if begin != -1 and stop != -1 else text

    def run(
        self,
        builder: PromptBuilder,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        rows = builder.rows
        out: list[dict] = []

        for i in tqdm.tqdm(range(0, len(rows), batch_size)):
            chunk = rows[i : i + batch_size]
            token_batches = [builder.build(r) for r in chunk]

            responses = self.llm.generate(token_batches, self.params)
            for r_row, r_resp in zip(chunk, responses):
                gen_ids = r_resp.outputs[0].token_ids
                raw = self.tokenizer.decode(gen_ids).strip()
                parsed = (
                    self._strip_between_markers(raw)
                    if builder.is_instruct
                    else raw
                )
                out.append(
                    dict(
                        completion_tokens=len(gen_ids),
                        input_text=r_row["input"],
                        full_answer=raw,
                        gold_ans=r_row["target"],
                        exact_match=(parsed == r_row["target"]),
                    )
                )
        return out


###############################################################################
#  MAIN ########################################################################
###############################################################################
def cli():
    p = argparse.ArgumentParser("Unified vLLM evaluator")
    p.add_argument("--model", required=True, choices=MODELS, help="Model key")
    p.add_argument("--dataset", required=True, choices=DATASETS, help="Dataset name")
    p.add_argument("--config", default="exact", help="Prompt variant (registry key)")
    p.add_argument("--shots", type=int, default=5, help="#few-shot examples (codeassist)")
    p.add_argument("--outdir", default="results/realistic", help="Output folder")
    # vLLM / runtime
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tensor_parallel", type=int, default=1)
    p.add_argument("--seed", type=int, default=2024)
    # prompt registry
    p.add_argument(
        "--prompt_path",
        default="prompts/realistic",
        help="Root dir that contains registry sub-dirs",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = cli()

    spec = DATASETS[args.dataset]
    data_dir = Path(spec["path"])
    few_shot_mode = spec["few_shot"]
    is_instruct = "instruct" in args.model

    evaluator = ModelEvaluator(
        model_path=MODELS[args.model],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        tensor_parallel=args.tensor_parallel,
    )

    # each dataset keeps its own prompts
    prompt_dir = Path(args.prompt_path) / args.dataset

    for data_file in data_dir.glob("*.jsonl"):
        for task_key in spec["tasks"]:
            rows = load_dataset(data_file, task_key, few_shot_mode)

            builder = PromptBuilder(
                shots=args.shots,
                task_key=task_key,
                dataset_rows=rows,
                prompt_dir=prompt_dir,
                tokenizer=evaluator.tokenizer,
                config=args.config,
                few_shot_mode=few_shot_mode,
                is_instruct=is_instruct,
            )

            outfile = (
                Path(args.outdir)
                / args.dataset
                / task_key
                / f"{args.model}_{data_file.stem}_{args.config}.jsonl"
            )

            if outfile.exists():
                print(f"⚠️  {outfile} exists – skip")
                continue

            results = evaluator.run(builder, args.batch_size)
            save_to_jsonl(outfile, results)
