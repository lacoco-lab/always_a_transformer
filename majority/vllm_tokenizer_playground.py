import requests

from openai import Client
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

data = open("datasets/majority/s1/b40-50/majority_10.txt", "r").readlines()

prompt_template = """Task: Print the most occurring digit (i.e., 0 or 1) from the `Input` string below. In case of an equal number of occurrences, print "0".

Input: {{ input }}

Do this task yourself. Put the output digit (i.e., 0 or 1) between <ans> and </ans> tags. Use the phrase "THE_END" to mark the end of the response."""


def print_outputs(outputs, tokenizer):
    for output in outputs:
        print(tokenizer.convert_ids_to_tokens(output.outputs[0].token_ids))
        print(output.outputs[0].cumulative_logprob)
        print(tokenizer.decode(output.outputs[0].token_ids))
        print("-" * 80)


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


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2, gpu_memory_utilization=0.85, seed=5, skip_tokenizer_init=True)
sampling_params = SamplingParams(max_tokens=32000, temperature=0.0, stop="THE_END", logprobs=True)

batched_inputs = []

for line in data:
    majority = line.strip()
    tokenized_majority = ['Ġ']+ ["0" if char == '0' else "1" for char in majority] + ['ĊĊ']
    chat_prompt = [
        {"role": "system", "content": "You are a very careful and precise assistant. You always follow the instructions and solve tasks yourself. You never generate code."},
        {"role": "user", "content": prompt_template},
    ]
    chat_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False)
    # print(chat_prompt)
    tokenized_prompt = custom_tokenization_with_ids(chat_prompt, " {{ input }}\n\n", tokenized_majority, tokenizer)
    # print(tokenized_prompt)
    batched_inputs.append(tokenized_prompt)
    
    if len(batched_inputs) == 8:
        outputs = llm.generate(prompt_token_ids=batched_inputs, sampling_params=sampling_params)
        print_outputs(outputs, tokenizer)
        batched_inputs = []

    # model = client.models.list()
    # headers = {"User-Agent": "Test Client"}
    # pload = {
    #     "messages": tokenized_prompt,
    #     "model": model.data[0].id,
    #     "temperature": 0.0,
    #     "max_tokens": 2000,
    #     "stream": False,
    #     "logprobs": True,
    #     "stop": "THE_END"
    # }
    # response = requests.post(f"{client.base_url}chat/completions",
    #                          headers=headers,
    #                          json=pload,
    #                          stream=False,)
    # print(response.json())
if len(batched_inputs) > 0:
    outputs = llm.generate(prompt_token_ids=batched_inputs, sampling_params=sampling_params)
    print_outputs(outputs, tokenizer)
llm.llm_engine.__del__()