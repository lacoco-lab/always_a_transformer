from vllm import SamplingParams
from utils.utils import get_last_write_index


def construct_vllm_chat_prompt(task_prompt, system_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]


def _prepare_prompts_vllm(data, task_prompt, system_prompt, spaced=False):
    chat_data = []
    for d in data:
        prompt_message = construct_vllm_chat_prompt(task_prompt.text({"input": " ".join(list(d.strip()[:-1])) if spaced else d.strip()[:-1]}),
                                                    system_prompt.text())
        chat_data.append(prompt_message)
    return chat_data


def batch_query_vllm(data, model, task_prompt, system_prompt, answer_parser, batch_size=16):

    outputs = []
    sampling_params = SamplingParams(max_tokens=300, temperature=0, top_p=0.7, stop=["<end>"], logprobs=True,
                                     top_k=1)
    prompt_data = _prepare_prompts_vllm(data, task_prompt, system_prompt, spaced=False)
    for idx, d in enumerate(prompt_data):
        res = model.chat(messages=d, sampling_params=sampling_params, use_tqdm=False)
        response = {
            "id": idx,
            "prompt": data[idx].strip()[:-1],
            "answer": answer_parser(res[0].outputs[0].text),
            "flipflop": data[idx].strip(),
            "last_valid_token": int(data[idx].strip()[-1]),
            "last_write_index": get_last_write_index(data[idx].strip()),
            "full_answer": res[0].outputs[0].text
        }
        outputs.append(response)
    return outputs