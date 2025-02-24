import asyncio
import requests

from time import sleep

from more_itertools import chunked
from tqdm.asyncio import tqdm

from utils.inference_constants import INSTRUCT_INFERENCE_PARAMS, COMPLETION_INFERENCE_PARAMS


def wait_for_engine_to_start(server_url, secs=5):
    while True:
        try:
            response = requests.get(f"{server_url.replace('v1', 'health')}", verify=False)
            if response.status_code == 200:
                print(f"\n\nvLLM server is available now!\n\n")
                break
        except requests.exceptions.ConnectionError:
            print(f"\n\nWaiting for vLLM server to be available, retrying in {secs} seconds\n\n")
            sleep(secs)


async def openai_vllm_chat(client, task_prompt, system_prompt, inference_params, xid="task"):
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


async def _prepare_prompts(data, task_prompt, spaced=False):
    for idx, d in enumerate(data):
        yield idx, task_prompt.text({"input": " ".join(list(d.strip())) if spaced else d.strip()})


def construct_vllm_chat_prompt(task_prompt, system_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]


async def openai_single_chat(data, client, task_prompt, system_prompt, spaced_input=False, xid="task-{}", 
                             inference_params=INSTRUCT_INFERENCE_PARAMS):
    q_tasks = []
    async for d_idx, d in _prepare_prompts(data, task_prompt, spaced=spaced_input):
        q_task = asyncio.create_task(openai_vllm_chat(client, d, system_prompt.text(), inference_params,
                                                      xid=xid.format(d_idx)))
        q_tasks.append(q_task)

    await tqdm.gather(*q_tasks)
    responses = [q_task.result() for q_task in q_tasks]
    return responses


def batch_chat(data, client, task_prompt, system_prompt, batch_size=1000, 
               inference_params=INSTRUCT_INFERENCE_PARAMS):
    responses = []
    chunked_data = list(chunked(data, batch_size))
    # batch_cnt = 0
    for chunk in tqdm(chunked_data):
        resp = asyncio.run(openai_single_chat(chunk, client, task_prompt, system_prompt, inference_params=inference_params))
        responses.extend(resp)
        # batch_cnt += 1
        # if batch_cnt == 5:
        #     break
    return responses


async def openai_vllm_complete(client, task_prompt, inference_params, xid="task"):
    model = await client.models.list()
    response = await client.completions.create(
        model=model.data[0].id,
        prompt=task_prompt,
        **inference_params,
        extra_headers={
            "x-request-id": xid,
        }
    )
    return response


async def openai_single_complete(data, client, task_prompt, spaced_input=False, xid="task-{}", inference_params=COMPLETION_INFERENCE_PARAMS):
    q_tasks = []
    async for d_idx, d in _prepare_prompts(data, task_prompt):
        q_task = asyncio.create_task(openai_vllm_complete(client, d, inference_params, xid=xid.format(d_idx)))
        q_tasks.append(q_task)

    await tqdm.gather(*q_tasks)
    responses = [q_task.result() for q_task in q_tasks]
    return responses


def batch_complete(data, client, task_prompt, batch_size=1000, inference_params=COMPLETION_INFERENCE_PARAMS):
    responses = []
    chunked_data = list(chunked(data, batch_size))
    for chunk in tqdm(chunked_data):
        resp = asyncio.run(openai_single_complete(chunk, client, task_prompt, inference_params=inference_params))
        responses.extend(resp)
    return responses
