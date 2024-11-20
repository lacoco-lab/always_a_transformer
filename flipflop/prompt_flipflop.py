import argparse
import re

from pathlib import Path

import numpy as np
import openai

from banks.registries import DirectoryPromptRegistry


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--engine", type=str, required=True, help="Engine to use for inference")
    args = ap.parse_args()

    client, model = None, None
    if "openai" in args.engine:
        client = openai.Client(
            base_url="http://134.96.104.203:8000/v1", api_key="sk_noreq")
    else:
        raise NotImplementedError("No other engines supported yet")

    with open(args.ip_path) as reader:
        data = reader.readlines()

    registry = DirectoryPromptRegistry(Path("prompts/flipflop_zero-shot_basic"), force_reindex=True)
    task_prompt = registry.get(name="task")
    system_prompt = registry.get(name="sys")

    for d in data:
        q_prompt = task_prompt.text({"input": d.strip()[:-1]})
        response = openai_vllm_chat(client, q_prompt, system_prompt.text())
        res_text = response.choices[0].message.content
        answer = parse_response(res_text)
        print(f"Question: {d.strip()[:-1]} **** \nAnswer: {answer}\n****\n")
        # print("="*50)
        # print(f"Response: {res_text}")
        # print("="*50)
