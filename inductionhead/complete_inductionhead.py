import argparse

from pathlib import Path

import jsonlines
import openai

from banks.registries import DirectoryPromptRegistry

from utils.vllm_openai_server import batch_complete, wait_for_engine_to_start
from utils.utils import save_to_jsonl, get_first_write_index

INFERENCE_PARAMS = {"max_tokens": 2, "temperature": 0, "stop": "THE_END", "logprobs": True, "seed": 5,
                    "extra_body": {"top_k": -1}}



def merge_data_with_responses(data, responses, task="before"):
    for r_idx, resp in enumerate(responses):
        d = data[r_idx]
        d["answer"] = resp.choices[0].text
        gold_char_idx = get_first_write_index(d['input'])-1 if task == "before" else get_first_write_index(d['input'])+1
        d["gold_ans_char"] = d['input'][gold_char_idx]
        d["is_correct"] = d["gold_ans_char"] in d["answer"].strip()
        d["input_length"] = resp.usage.prompt_tokens
        d["output_length"] = resp.usage.completion_tokens
        d["tokenized_output"] = resp.choices[0].logprobs.tokens
    return data


def get_op_num_tokens(ip_path):
    op_num_tokens = 500
    if "3000" in ip_path:
        op_num_tokens = 3000
    elif "4000" in ip_path:
        op_num_tokens = 4000
    elif "5000" in ip_path:
        op_num_tokens = 5000
    elif "bigger" in ip_path:
        op_num_tokens = 2000
    elif '100' in ip_path:
        op_num_tokens = 100
    return op_num_tokens


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt registry directory")
    ap.add_argument("--save_path", type=str, nargs='?', default="results/last_ones/llama3.3_70B-instruct",
                    help="Dir Path to save results in jsonlines")
    ap.add_argument("--port", type=str, required=False, default="8080", help="Port to use for the server")
    ap.add_argument("--config", type=str, required=False, default="before", choices=["before", "after"],
                    help="before or after")
    args = ap.parse_args()

    # read jsonl file
    data, inductionheads = [], []
    with jsonlines.open(args.ip_path, "r") as reader:
        for obj in reader:
            obj["input"] = obj["input"].strip()
            data.append(obj)
            inductionheads.append(obj["input"])

    registry = DirectoryPromptRegistry(Path(args.prompt_path), force_reindex=True)
    task_prompt = registry.get(name=f"task_{args.config}")

    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    base_url = f"http://0.0.0.0:{args.port}/v1"

    wait_for_engine_to_start(base_url)

    inference_params = INFERENCE_PARAMS

    client = openai.AsyncClient(base_url=base_url, api_key="sk_noreq", max_retries=10)
    results = batch_complete(inductionheads, client, task_prompt, inference_params=inference_params, batch_size=32)
    results = merge_data_with_responses(data, results, task=args.config)
    save_path = Path(args.save_path) / f"{args.prompt_path.split('/')[-1]}"

    op_filename = f"{get_op_num_tokens(args.ip_path)}_{args.config}_seed-{inference_params['seed']}.jsonl"
    save_to_jsonl(str(save_path), op_filename, results)
