import argparse

from pathlib import Path

import jsonlines
import openai

from banks.registries import DirectoryPromptRegistry

from utils.vllm_openai_server import batch_chat, wait_for_engine_to_start
from utils.utils import (get_last_write_index, get_first_write_index, parse_flipflop_response, save_to_jsonl, 
                         get_flipflop_task_and_type_from_prompt)


LLAMA_INFERENCE_PARAMS = {"max_tokens": 16000, "temperature": 0, "stop": "THE_END", "logprobs": True, "seed": 5,
                          "extra_body": {"top_k": -1}}

OLMO_INFERENCE_PARAMS = {"max_tokens": 3000, "temperature": 0, "stop": "THE_END", "logprobs": True, "seed": 5,
                         "extra_body": {"top_k": -1}}


def get_gold_ans_char(data, config):
    if "fw" in config:
        return data[get_first_write_index(data)+1] if "rc" in config else data[get_first_write_index(data)-1]
    elif "lw" in config:
        return data[get_last_write_index(data)+1] if "rc" in config else data[get_last_write_index(data)-1]
    else:
        raise ValueError("Invalid config")


def merge_data_with_responses(data, responses, task, task_type, config):
    
    for r_idx, resp in enumerate(responses):
        d = data[r_idx]
        d["answer"] = parse_flipflop_response(resp.choices[0].message.content, task, task_type)
        d["gold_ans_char"] = get_gold_ans_char(d["input"], config)
        d["full_answer"] = resp.choices[0].message.content
        d["input_length"] = resp.usage.prompt_tokens
        d["output_length"] = resp.usage.completion_tokens
        d["tokenized_output"] = [lp.token for lp in resp.choices[0].logprobs.content]
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt registry directory")
    ap.add_argument("--cot", type=str, required=False, default="cot", help="cot or nocot")
    ap.add_argument("--save_path", type=str, nargs='?', default="results/flipflop", help="Dir Path to save results in jsonlines")
    ap.add_argument("--port", type=str, required=False, default="8080", help="Port to use for the server")
    ap.add_argument("--config", type=str, required=True, choices=["fw-lc", "fw-rc", "lw-lc", "lw-rc"], help="Config to use")
    args = ap.parse_args()

    data, flipflops = [], []
    with jsonlines.open(args.ip_path, "r") as reader:
        for obj in reader:
            obj["input"] = obj["input"].strip()
            data.append(obj)
            flipflops.append(obj["input"])

    cot = "nocot" if "nocot" in args.cot else "cot"
    config = args.config

    registry = DirectoryPromptRegistry(Path(args.prompt_path), force_reindex=True)
    task_prompt = registry.get(name=f"task_{cot}_{config}")
    system_prompt = registry.get(name="sys")
    
    task, task_type = get_flipflop_task_and_type_from_prompt(args.prompt_path)

    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    base_url = f"http://0.0.0.0:{args.port}/v1"
    
    wait_for_engine_to_start(base_url)
    
    if "llama" in args.save_path.lower():
        inference_params = LLAMA_INFERENCE_PARAMS
    elif "olmo" in args.save_path.lower():
        inference_params = OLMO_INFERENCE_PARAMS
    else:
        raise ValueError("Unknown model")
    
    client = openai.AsyncClient(
        base_url=base_url, api_key="sk_noreq", max_retries=10)
    results = batch_chat(flipflops, client, task_prompt, system_prompt, inference_params=inference_params, batch_size=128)
    results = merge_data_with_responses(data, results, task, task_type, config)

    save_path = Path(args.save_path) / f"{args.prompt_path.split('/')[-1]}"
    # output format: 500_cot_fw-lc_seed-5_normal.jsonl (normal can be replaced with the type of data i.e. replaced-xyz)
    save_to_jsonl(str(save_path), f"500_{cot}_{config}_seed-{inference_params['seed']}.jsonl", results)
