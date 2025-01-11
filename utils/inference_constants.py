INSTRUCT_INFERENCE_PARAMS = {"seed": 5, "max_tokens": 2000, "temperature": 0, "stop": "</answer>", "logprobs": True,
                    "extra_body": {"top_k": 1}}

COMPLETION_INFERENCE_PARAMS = {"seed": 5, "max_tokens": 2, "temperature": 0, "logprobs": True,
                               "extra_body": {"top_k": 1}}


MASK_INSTRUCT_INFERENCE_PARAMS = {"seed": 5, "max_tokens": 2000, "temperature": 0, 
                                  "stop": "<sequence_end>", "logprobs": True,
                                  "extra_body": {"top_k": 1}}

MASK_COMPLETION_INFERENCE_PARAMS = {"seed": 5, "max_tokens": 2000, "temperature": 0, "logprobs": True,
                                    "stop": "<sequence_end>", "extra_body": {"top_k": 1}}

QA_INSTRUCT_INFERENCE_PARAMS = {"seed": 5, "max_tokens": 2000, "temperature": 0, 
                                "stop": "</answer>", "logprobs": True,
                                "extra_body": {"top_k": 1}}

QA_COMPLETION_INFERENCE_PARAMS = {"seed": 5, "max_tokens": 2, "temperature": 0, "logprobs": True,
                                  "extra_body": {"top_k": 1}}


def get_inference_params(task, task_type):
    inference_id = f"{task}_{task_type}"
    if inference_id == "retrieve_instruct":
        return INSTRUCT_INFERENCE_PARAMS
    elif inference_id == "retrieve_complete":
        return COMPLETION_INFERENCE_PARAMS
    elif inference_id == "mask_instruct":
        return MASK_INSTRUCT_INFERENCE_PARAMS
    elif inference_id == "mask_complete":
        return MASK_COMPLETION_INFERENCE_PARAMS
    elif inference_id == "qa_instruct":
        return QA_INSTRUCT_INFERENCE_PARAMS
    elif inference_id == "qa_complete":
        return QA_COMPLETION_INFERENCE_PARAMS
    else:
        raise ValueError(f"Unknown task: {task}")
