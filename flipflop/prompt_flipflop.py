import argparse

from pathlib import Path

import openai

from banks.registries import DirectoryPromptRegistry

from utils.vllm_openai_server import batch_query, wait_for_engine_to_start
from utils.utils import get_last_write_index, parse_flipflop_response, save_to_jsonl


def merge_data_with_responses(data, responses):
    output = []
    for r_idx, resp in enumerate(responses):
        d = data[r_idx]
        res_text = resp.choices[0].message.content
        answer = parse_flipflop_response(res_text)
        last_write_index = get_last_write_index(d.strip())
        # print(f"Question: {d.strip()[:-1]} **** \nAnswer: {answer}\n****\n")
        response = {
            "id": r_idx,
            "prompt": d.strip()[:-1],
            "answer": answer,
            "flipflop": d.strip(),
            "last_valid_token": int(d.strip()[-1]),
            "last_write_index": last_write_index,
            "full_answer": res_text
        }
        output.append(response)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--engine", type=str, required=True, help="Engine to use for inference")
    ap.add_argument("--save_path", type=str, nargs='?', default="results/flipflop", help="Dir Path to save results in jsonlines")
    args = ap.parse_args()

    with open(args.ip_path) as reader:
        if args.ip_path.endswith(".txt"):
            data = reader.readlines()
        else:
            raise ValueError("Invalid input file type")

    registry = DirectoryPromptRegistry(Path("prompts/flipflop_zero-shot_basic"), force_reindex=True)
    task_prompt = registry.get(name="task")
    system_prompt = registry.get(name="sys")

    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    #base_url = "http://134.96.104.203:8080/v1" # Run this if you are running locally and want to ping Mayank's machine
    base_url = "http://0.0.0.0:8080/v1" # Run this from the coli server

    if "openai" in args.engine:
        wait_for_engine_to_start(base_url)
        client = openai.AsyncClient(
            base_url=base_url, api_key="sk_noreq", max_retries=10)
        results = batch_query(data, client, task_prompt, system_prompt)
        results = merge_data_with_responses(data, results)
    elif "vllm" in args.engine:
        import sys
        sys.exit("VLLM engine not supported yet")
        # WARNING: This is not ready. PLEASE DO NOT RUN THIS.
        model_name = "allenai/OLMo-7B-0724-Instruct-hf"
        tp_size = 4 if "70B" in model_name else 2
        model = LLM(model_name, seed=5, tensor_parallel_size=2, gpu_memory_utilization=0.85)
        results = batch_query_vllm(data, model, task_prompt, system_prompt)
        model.llm_engine.__del__()
    else:
        raise NotImplementedError("No other engines supported yet")

    save_to_jsonl(args.save_path, f"{Path(args.ip_path).name.split(".")[0]}_spaced_results.jsonl", results)
