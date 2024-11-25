import argparse
import asyncio
import re

from pathlib import Path

import openai

from banks.registries import DirectoryPromptRegistry
from more_itertools import chunked
from tqdm.asyncio import tqdm

from utils import get_last_write_index, save_to_json


async def openai_vllm_chat(client, task_prompt, system_prompt, xid):
    inference_params = {"seed": 5, "max_tokens": 200, "temperature": 0, "stop": "<end>", "logprobs": True}
    model = await client.models.list()
    response = await client.chat.completions.create(
        model=model.data[0].id,
        # model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ],
        **inference_params,
        extra_headers={
            "x-request-id": xid,
        }
    )
    return response


def parse_response(response_text):
    answer = re.search(r'<answer>(.*?)</ans', response_text).group(1)
    return answer


async def _prepare_prompts(data, task_prompt):
    for idx, d in enumerate(data):
        yield idx, task_prompt.text({"input": d.strip()[:-1]})


async def openai_single_query(data, client, task_prompt, system_prompt):
    q_tasks = []
    async for d_idx, d in _prepare_prompts(data, task_prompt):
        q_task = asyncio.create_task(openai_vllm_chat(client, d, system_prompt.text(), f"flipflop-{d_idx}"))
        q_tasks.append(q_task)

    await tqdm.gather(*q_tasks)
    responses = [q_task.result() for q_task in q_tasks]
    return responses


def batch_query(data, client, task_prompt, system_prompt, batch_size=1000):
    responses = []
    chunked_data = list(chunked(data, batch_size))
    for chunk in tqdm(chunked_data):
        resp = asyncio.run(openai_single_query(chunk, client, task_prompt, system_prompt))
        responses.extend(resp)
        break
    outputs = merge_data_with_responses(data, responses)
    return outputs


def merge_data_with_responses(data, responses):
    output = []
    for resp in responses:
        d_idx = int(resp._request_id.replace("flipflop-", ""))
        d = data[d_idx]
        res_text = resp.choices[0].message.content
        answer = parse_response(res_text)
        last_write_index = get_last_write_index(d)
        # print(f"Question: {d.strip()[:-1]} **** \nAnswer: {answer}\n****\n")

        response = {
            "id": d_idx,
            "prompt": d.strip()[:-1],
            "answer": int(answer),
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

    client, model = None, None
    if "openai" in args.engine:
        client = openai.AsyncClient(
            base_url="http://127.0.0.1:8080/v1", api_key="sk_noreq")
        results = batch_query(data, client, task_prompt, system_prompt)
    else:
        raise NotImplementedError("No other engines supported yet")

    save_to_json(args.save_path, results)
