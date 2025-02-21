import argparse

from pathlib import Path

import jsonlines
import openai

from banks.registries import DirectoryPromptRegistry

from utils.vllm_openai_server import batch_chat, wait_for_engine_to_start
from utils.utils import save_to_jsonl

INSTRUCT_INFERENCE_PARAMS = {"max_tokens": 16000, "temperature": 0, "stop": "THE_END", "logprobs": True, 
                             "extra_body": {"top_k": -1}}


def parse_first_ones_response(response_text):
    try:
        ans_begin = response_text.find("<ans>")
        ans_end = response_text.find("</ans>")
        answer = response_text[ans_begin + 5:ans_end]
    except:
        print(f"Error in parsing response: {response_text}")
        answer = response_text
    return answer


def merge_data_with_responses(data, responses):
    for r_idx, resp in enumerate(responses):
        d = data[r_idx]
        d["answer"] = parse_first_ones_response(resp.choices[0].message.content)
        d["gold_ans_char"] = d['input'].strip()[0]
        d["full_answer"] = resp.choices[0].message.content
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt registry directory")
    ap.add_argument("--save_path", type=str, nargs='?', default="results/last_ones/llama3.3_70B-instruct", help="Dir Path to save results in jsonlines")
    ap.add_argument("--port", type=str, required=False, default="8080", help="Port to use for the server")
    args = ap.parse_args()
    
    # read jsonl file
    data, first_ones = [], []
    with jsonlines.open(args.ip_path, "r") as reader:
        for obj in reader:
            data.append(obj)
            first_ones.append(obj["input"])
    
    registry = DirectoryPromptRegistry(Path(args.prompt_path), force_reindex=True)
    task_prompt = registry.get(name="task")
    system_prompt = registry.get(name="sys")
    
    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    base_url = f"http://0.0.0.0:{args.port}/v1"
    
    wait_for_engine_to_start(base_url)
    
    client = openai.AsyncClient(base_url=base_url, api_key="sk_noreq", max_retries=10)
    results = batch_chat(first_ones, client, task_prompt, system_prompt, inference_params=INSTRUCT_INFERENCE_PARAMS, batch_size=32)
    results = merge_data_with_responses(data, results)
    save_to_jsonl(args.save_path, "500_hard_all.jsonl", results)
