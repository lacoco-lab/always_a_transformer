import argparse

from pathlib import Path

from banks.registries import DirectoryPromptRegistry
from more_itertools import chunked, flatten

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.utils import get_model_to_num_gpu_mapping, save_to_jsonl


def preapre_data(data, tokenizer, task_prompt, system_prompt, batch_size=32):
    for d in tqdm(data):
        majority_ip = d["input"].strip()
        tokenized_majority = ['Ġ'] + ["0" if char == '0' else "1" for char in majority_ip] + ['ĊĊ']
        chat_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]
        chat_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False)
        # print(chat_prompt)
        tokenized_prompt = custom_tokenization_with_ids(chat_prompt, " {{ input }}\n\n", tokenized_majority, tokenizer)
        d["tokenized_input"] = tokenized_prompt
        # print(d)
        # print(tokenizer.decode(tokenized_prompt))
    batched_inputs = list(chunked(data, batch_size))
    return batched_inputs


def custom_tokenization_with_ids(prompt, sub_prompt, tokenized_sub_prompt, tokenizer):
    """
    Tokenizes the prompt using a Hugging Face tokenizer, then replaces the tokenization of
    the sub_prompt (which can consist of multiple words) with a custom tokenization, and returns the tensor IDs
    of the entire prompt.

    Args:
    prompt (str): The larger prompt where the sub_prompt is located.
    sub_prompt (str): The sub_prompt (which can consist of multiple words) to be tokenized with a custom tokenizer.
    tokenized_sub_prompt (list): A list of tokens representing the custom tokenization of the sub_prompt.
    model_name (str): The model name for Hugging Face tokenizer (default is 'bert-base-uncased').

    Returns:
    list: A list of tensor IDs representing the modified tokenization of the prompt.
    """

    # Tokenize the prompt using Hugging Face tokenizer
    prompt_tokens = tokenizer.tokenize(prompt)

    # Tokenize the sub_prompt using the Hugging Face tokenizer to find the tokenization
    sub_prompt_tokens = tokenizer.tokenize(sub_prompt)

    # Find the index where the sub_prompt appears in the prompt
    start_idx = None
    for i in range(len(prompt_tokens) - len(sub_prompt_tokens) + 1):
        if prompt_tokens[i:i + len(sub_prompt_tokens)] == sub_prompt_tokens:
            start_idx = i
            break

    # If the sub_prompt is found, replace the tokens with the custom tokenization
    if start_idx is not None:
        # Replace the found sub_prompt's tokenization with the custom tokenization
        prompt_tokens[start_idx:start_idx + len(sub_prompt_tokens)] = tokenized_sub_prompt
    else:
        print(f"Sub-prompt '{sub_prompt}' not found in the prompt.")

    # Convert the modified token list to tensor IDs
    # print(prompt_tokens)
    tensor_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)

    return tensor_ids


def read_all_files_and_create_single_jsonl(all_files):
    data = []
    for file in all_files:
        with open(file, "r") as f:
            for d in f.readlines():
                data.append({"filepath": str(file), "input": d.strip()})
    return data


def infer_majority(data, llm, sampling_params, tokenizer, save_path=None, write_every=320):
    infer_cnt = 0
    for chunk in tqdm(data):
        tokenized_input = [d["tokenized_input"] for d in chunk]
        outputs = llm.generate(prompt_token_ids=tokenized_input, sampling_params=sampling_params)
        detok_outputs, logprobs = [], []
        for output in outputs:
            detok_outputs.append(tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True))
            logprobs.append(output.outputs[0].cumulative_logprob)
        for idx, d in enumerate(chunk):
            d["full_answer"] = detok_outputs[idx].strip()
            d["logprob"] = logprobs[idx]
        infer_cnt += len(chunk)
        if infer_cnt % write_every == 0 and save_path is not None:
            print(f"Writing {infer_cnt} examples to jsonl")
            # partial_data needs to be derived from a set of chunks and saved to jsonl
            save_to_jsonl(save_path, "partial_results.jsonl", data)
    return data


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--ip_path", type=str, required=True, help="Dir Path to the dataset")
    p.add_argument("--model", type=str, required=True, help="Model path to use for inference")
    p.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt registry directory")
    p.add_argument("--save_path", type=str, nargs='?', default="results/majority",
                    help="Dir Path to save results in jsonlines")
    p.add_argument("--batch_size", type=int, required=False, default=16, help="Batch size for inference")
    args = p.parse_args()
    
    ip_files = Path(args.ip_path).rglob('*.txt')
    # remove example files
    ip_files = [f for f in ip_files if "example" not in str(f)]
    
    all_data = read_all_files_and_create_single_jsonl(ip_files)
    print(f"Total number of examples: {len(all_data)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt_registry = DirectoryPromptRegistry(Path(args.prompt_path), force_reindex=True)
    task_prompt = prompt_registry.get(name="task").text({"input": "{{ input }}"})
    system_prompt = prompt_registry.get(name="sys").text()
    
    all_data = preapre_data(all_data, tokenizer, task_prompt, system_prompt, batch_size=args.batch_size)
    print(f"Total number of batches: {len(all_data)}")

    llm = LLM(model=args.model, 
              tensor_parallel_size=get_model_to_num_gpu_mapping(args.model),
              # tensor_parallel_size=2,
              gpu_memory_utilization=0.95, 
              seed=5,
              skip_tokenizer_init=True,
              max_seq_len_to_capture=16000)
    
    sampling_params = SamplingParams(max_tokens=16000, temperature=0.0, stop="THE_END", logprobs=True)
    
    save_path = Path(args.save_path, f"{args.model.split('/')[-1]}", f"{args.prompt_path.split('/')[-2]}")
    all_data = infer_majority(all_data, llm, sampling_params, tokenizer, save_path, write_every=args.batch_size*30)
    llm.llm_engine.__del__()
    all_data = flatten(all_data)
    # save path should include both model name and prompt task
    save_to_jsonl(save_path, filename=f"all_results.jsonl", list_of_dicts=all_data)
