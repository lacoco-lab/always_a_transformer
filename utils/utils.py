import os
import re

from pathlib import Path

import jsonlines


def get_last_write_index(flipflop):
    """
    Get the index of the last write operation in the flipflop string
    >> get_last_write_index("w0r0")
    0
    >> get_last_write_index("w0w1r0")
    2
    >> get_last_write_index("w0w1r0w1")
    6
    :param flipflop: str
    :return: int
    """

    reversed_flipflop = flipflop[::-1]
    for idx, char in enumerate(reversed_flipflop):
        if char == 'w':
            return len(flipflop) - idx - 1


def save_to_jsonl(path, filename, list_of_dicts):
    """
    Save a list of dictionaries to a jsonlines file
    :param path: directory to save the jsonlines file
    :param list_of_dicts: data to save
    :return: void
    """
    print(f"Saving to {path}/{filename}")
    out_path = Path(path, filename)
    if not out_path.parent.exists():
        os.makedirs(out_path.parent, exist_ok=True)
    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(list_of_dicts)
        

def _get_task_regex(task, task_type):
    """
    Get the regex pattern for the given task
    :param task: str, task name
    :return: str, regex pattern
    """
    regex_id = f"{task}_{task_type}"
    if regex_id == "retrieve_instruct":
        return r'<answer>(.*?)</answer>'
    elif regex_id == "retrieve_complete":
        return "NA"
    elif regex_id == "mask_instruct":
        return r'<sequence_start>(.*?)<sequence_end>'
    elif regex_id == "mask_complete":
        return r'<sequence_start>(.*?)<sequence_end>'
    elif regex_id == "qa_instruct":
        return r'<answer>(.*?)</answer>'
    elif regex_id == "qa_complete":
        return "NA"
    else:
        raise ValueError(f"Unknown task: {task}")


def parse_flipflop_response(response_text, task="retrieve", task_type="instruct"):
    regex_str = _get_task_regex(task, task_type)
    try:
        answer = re.search(regex_str, response_text).group(1)
    except (AttributeError, ValueError):
        # print(f"Response: {response_text}")
        answer = response_text
    return answer


def get_flipflop_task_and_type_from_prompt(prompt_name):
    if "mask" in prompt_name:
        task = "mask"
    elif "qa" in prompt_name:
        task = "qa"
    else:
        task = "retrieve"
    if "chat" in prompt_name:
        task_type = "instruct"
    else:
        task_type = "complete"
    return task, task_type


def get_stop_token(task, task_type):
    """
    Get the stop token for the given task
    :param task: str, task name
    :return: str, stop token
    """
    task_id = f"{task}_{task_type}"
    if task_id == "retrieve_instruct":
        return "</answer>"
    elif task_id == "retrieve_complete":
        return "NA"
    elif task_id == "mask_instruct":
        return "<sequence_end>"
    elif task_id == "mask_complete":
        return "<sequence_end>"
    elif task_id == "qa_instruct":
        return "</answer>"
    elif task_id == "qa_complete":
        return "NA"
    else:
        raise ValueError(f"Unknown task: {task}")


def get_model_to_num_gpu_mapping(model: str):
    if "70B" in model:
        return 4
    elif "13B" in model:
        return 1
    elif "7B" in model:
        return 1
    elif "8B" in model:
        return 1
    else:
        raise ValueError(f"Unknown model size: {model}")
