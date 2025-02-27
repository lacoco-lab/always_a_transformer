import argparse

from pathlib import Path

import jsonlines
import openai

from banks.registries import DirectoryPromptRegistry

from utils.vllm_openai_server import batch_chat, wait_for_engine_to_start
from utils.utils import save_to_jsonl, get_first_write_index

# INSTRUCT_INFERENCE_PARAMS = {"max_tokens": 3000, "temperature": 0, "stop": "THE_END", "logprobs": True, 
#                              "extra_body": {"top_k": -1}}

LLAMA_INFERENCE_PARAMS = {"max_tokens": 16000, "temperature": 0, "stop": "THE_END", "logprobs": True, "seed": 5,
                          "extra_body": {"top_k": -1}}

OLMO_INFERENCE_PARAMS = {"max_tokens": 3000, "temperature": 0, "stop": "THE_END", "logprobs": True, "seed": 5,
                         "extra_body": {"top_k": -1}}


def parse_last_ones_response(response_text):
    try:
        ans_begin = response_text.find("<ans>")
        ans_end = response_text.find("</ans>")
        answer = response_text[ans_begin + 5:ans_end]
    except:
        print(f"Error in parsing response: {response_text}")
        answer = response_text
    return answer


def merge_data_with_responses(data, responses, task="before"):
    for r_idx, resp in enumerate(responses):
        d = data[r_idx]
        d["answer"] = parse_last_ones_response(resp.choices[0].message.content)
        gold_char_idx = get_first_write_index(d['input'])-1 if task == "before" else get_first_write_index(d['input'])+1
        d["gold_ans_char"] = d['input'][gold_char_idx]
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

    cot = "nocot" if "nocot" == args.cot else "cot"

    registry = DirectoryPromptRegistry(Path(args.prompt_path), force_reindex=True)
    task_prompt = registry.get(name=f"task_{cot}_{args.config}")
    system_prompt = registry.get(name="sys")

    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    base_url = f"http://0.0.0.0:{args.port}/v1"

    wait_for_engine_to_start(base_url)

    if "llama" in args.save_path.lower() or "mamba" in args.save_path.lower():
        inference_params = LLAMA_INFERENCE_PARAMS
    elif "olmo" in args.save_path.lower():
        inference_params = OLMO_INFERENCE_PARAMS
    else:
        raise ValueError("Unknown model")

    client = openai.AsyncClient(base_url=base_url, api_key="sk_noreq", max_retries=10)
    results = batch_chat(inductionheads, client, task_prompt, system_prompt, inference_params=inference_params,
                         batch_size=32)
    results = merge_data_with_responses(data, results, task=args.config)
    save_path = Path(args.save_path) / f"{args.prompt_path.split('/')[-1]}"
    # output format: 500_cot_seed-5_normal.jsonl (normal can be replaced with the type of data i.e. replaced-xyz)
    save_to_jsonl(str(save_path), f"500_{args.config}_{cot}_seed-{inference_params['seed']}.jsonl", results)
