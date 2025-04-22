import openai
import argparse
import jsonlines

from pathlib import Path
from banks.registries import DirectoryPromptRegistry

from utils.utils import save_to_jsonl
from utils.vllm_openai_server import batch_chat, wait_for_engine_to_start, batch_complete


LCTX_INFERENCE_PARAMS = {"max_tokens": 4000, "temperature": 0, "logprobs": True, "seed": 5,
                          "extra_body": {"top_k": -1}}

SCTX_INFERENCE_PARAMS = {"max_tokens": 2000, "temperature": 0, "stop": "THE_END", "logprobs": True, "seed": 5,
                         "extra_body": {"top_k": -1}}


def get_prompts_from_registry(prompt_path, config='exact'):
    # For controlled copying ; I can use the same prompts as Lorem Ipsum 
    registry = DirectoryPromptRegistry(Path(prompt_path), force_reindex=True)
    task_prompt = registry.get(name=f"task_{config}")
    system_prompt = registry.get(name="sys")
    return task_prompt, system_prompt


def get_correct_inference_params(max_inp_len):
    if "olmo" in args.save_path.lower():
        params = SCTX_INFERENCE_PARAMS
    else:
        params = LCTX_INFERENCE_PARAMS
    # Set the maximum number of tokens to the number of inputs in the character * 2.
    params['max_tokens'] = int(max_inp_len * 1.5)
    return params


def read_dataset_jsonl(dataset_path):
    # read jsonl file
    max_len, all_data, just_inputs = 0, [], []
    with jsonlines.open(dataset_path, "r") as reader:
        for obj in reader:
            curr_input = obj["input"].strip()
            obj["gold_ans"] = curr_input
            all_data.append(obj)
            just_inputs.append(curr_input)
            max_len = max(max_len, len(curr_input.split(' ')))
    return just_inputs, all_data, max_len


def parse_paragraph_response(response_text):
    try:
        ans_begin = response_text.find("<paragraph>")
        ans_end = response_text.find("</paragraph>")
        answer = response_text[ans_begin + 11:ans_end]
    except:
        print(f"Error in parsing response: {response_text}")
        answer = response_text
    return answer


def merge_data_with_responses(copying_data, all_responses, is_chat=True):
    for (curr_data, response) in zip(copying_data, all_responses):
        if is_chat is True:
            # For Instruct models ; output has response.choices[0].message.content
            curr_data["answer"] = parse_paragraph_response(response.choices[0].message.content)
            curr_data["full_answer"] = response.choices[0].message.content
        else:
            # For completion models ; output has response.choices[0].text
            curr_data["answer"] = response.choices[0].text
        curr_data["input_length"] = response.usage.prompt_tokens
        curr_data["output_length"] = response.usage.completion_tokens
    return copying_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    ap.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt registry directory")
    ap.add_argument("--config", type=str, required=False, default="exact", help="exact or verbatim")
    ap.add_argument("--save_path", type=str, nargs='?', default="results/last_ones/llama3.3_70B-instruct",
                    help="Dir Path to save results in jsonlines")
    ap.add_argument("--port", type=str, required=False, default="8080", help="Port to use for the server")
    ap.add_argument("--model_type", type=str, required=False, default="chat", help="Instruct:chat or Normal:complete")

    args = ap.parse_args()

    copying_inputs, copying_data, max_inp_len = read_dataset_jsonl(args.ip_path)
    task_prompt, system_prompt = get_prompts_from_registry(args.prompt_path, args.config)

    # Make sure the results can be saved somewhere
    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True, exist_ok=True)

    base_url = f"http://0.0.0.0:{args.port}/v1"

    # Check if vLLM is available
    wait_for_engine_to_start(base_url, secs=60)

    # Get the correct inferernce parameters for the given model 
    inference_params = get_correct_inference_params(max_inp_len)

    # Set up the Open AI Aysnc client. 
    client = openai.AsyncClient(base_url=base_url, api_key="sk_noreq", max_retries=10)

    # Get the batch results for the given inputs
    if args.model_type == "chat":
        results = batch_chat(copying_inputs, client, task_prompt, system_prompt,
                             inference_params=inference_params, batch_size=8)
        results = merge_data_with_responses(copying_data, results, is_chat=True)
    else: 
        results = batch_complete(copying_inputs, client, task_prompt,
                                 inference_params=inference_params, batch_size=8)
        results = merge_data_with_responses(copying_data, results, is_chat=False)

    # output format: 500_cot_seed-5_normal.jsonl (normal can be replaced with the type of data i.e. replaced-xyz)
    save_path = Path(args.save_path)    
    op_filename = f"{args.ip_path}_{args.config}_seed-{inference_params['seed']}.jsonl"
    save_to_jsonl(str(save_path), op_filename, results)
