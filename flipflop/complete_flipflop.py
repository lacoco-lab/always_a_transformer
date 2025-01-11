import argparse

from pathlib import Path

import openai

from banks.registries import DirectoryPromptRegistry

from utils.inference_constants import get_inference_params
from utils.vllm_openai_server import batch_complete, wait_for_engine_to_start
from utils.utils import (get_last_write_index, parse_flipflop_response, save_to_jsonl, 
                         get_flipflop_task_and_type_from_prompt, get_stop_token)


def merge_data_with_responses(data, responses, task, task_type):
    output = []
    stop_token = get_stop_token(task, task_type)
    for r_idx, resp in enumerate(responses):
        d = data[r_idx]
        res_text = resp.choices[0].text + stop_token if stop_token != "NA" else resp.choices[0].text
        answer = parse_flipflop_response(res_text, task, task_type) if stop_token != "NA" else res_text
        last_write_index = get_last_write_index(d.strip())
        # print(f"Question: {d.strip()[:-1]} **** \nAnswer: {answer}\n****\n")
        response = {
            "id": r_idx,
            "prompt": d.strip()[:-1],
            "answer": answer.strip(),
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
    ap.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt registry directory")
    ap.add_argument("--save_path", type=str, nargs='?', default="results/flipflop", help="Dir Path to save results in jsonlines")
    args = ap.parse_args()

    with open(args.ip_path) as reader:
        if args.ip_path.endswith(".txt"):
            data = reader.readlines()
        else:
            raise ValueError("Invalid input file type")

    registry = DirectoryPromptRegistry(Path(args.prompt_path), force_reindex=True)
    task_prompt = registry.get(name="task")
    
    task, task_type = get_flipflop_task_and_type_from_prompt(args.prompt_path)
    inference_params = get_inference_params(task, task_type)

    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    #base_url = "http://134.96.104.203:8080/v1" # Run this if you are running locally and want to ping Mayank's machine
    base_url = "http://0.0.0.0:8080/v1" # Run this from the coli server

    if "openai" in args.engine:
        wait_for_engine_to_start(base_url)
        client = openai.AsyncClient(
            base_url=base_url, api_key="sk_noreq", max_retries=10)
        results = batch_complete(data, client, task_prompt, inference_params=inference_params)
        results = merge_data_with_responses(data, results, task, task_type)
    else:
        raise NotImplementedError("No other engines supported yet")

    save_to_jsonl(args.save_path, f"{Path(args.ip_path).name.split('.')[0]}_results.jsonl", results)
