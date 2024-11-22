import argparse
import re
from pathlib import Path
import numpy as np
import openai
from banks.registries import DirectoryPromptRegistry
from utils import get_last_write_index, save_to_json


def get_default_prompting_params():
    return {
        "seed": 5,
        "max_tokens": 200,
        "temperature": 0,
        "stop": "<end>",
        "logprobs": True,
    }


def openai_vllm_chat(client, task_prompt, system_prompt):
    inference_params = get_default_prompting_params()
    model = client.models.list()
    response = client.chat.completions.create(
        model=model.data[0].id,
        # model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ],
        **inference_params
    )
    return response


def construct_vllm_chat_prompt(task_prompt, system_prompt):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]
    }


def parse_response(response_text):
    answer = re.search(r'<answer>(.*?)</ans', response_text).group(1)
    return answer


def openai_single_query(data, client, task_prompt, system_prompt):

    responses = []
    for d in data:
        q_prompt = task_prompt.text({"input": d.strip()[:-1]})
        response = openai_vllm_chat(client, q_prompt, system_prompt.text())
        res_text = response.choices[0].message.content
        answer = parse_response(res_text)
        print(f"Question: {d.strip()[:-1]} **** \nAnswer: {answer}\n****\n")

        # Save response to a file
        last_write_index = get_last_write_index(d)
        response = {
            "prompt": q_prompt,
            "answer": answer,
            "flipflop": d,
            "last_valid_token": d[-1],
            "last_write_index": last_write_index,
            "full_answer": res_text
        }
        responses.append(response)
    return responses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--engine", type=str, required=True, help="Engine to use for inference")
    ap.add_argument("--save_path", type=str, nargs='?', default="results/flipflop", help="Dir Path to save results in jsonlines")
    args = ap.parse_args()

    with open(args.ip_path) as reader:
        data = reader.readlines()

    registry = DirectoryPromptRegistry(Path("prompts/flipflop_zero-shot_basic"), force_reindex=True)
    task_prompt = registry.get(name="task")
    system_prompt = registry.get(name="sys")

    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    client, model = None, None
    if "openai" in args.engine:
        client = openai.Client(
            base_url="http://127.0.0.1:8080/v1", api_key="sk_noreq")
        responses = openai_single_query(data, client, task_prompt, system_prompt)
    else:
        raise NotImplementedError("No other engines supported yet")

    save_to_json(args.save_path, responses)
