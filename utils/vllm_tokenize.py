
import argparse

import requests
from pathlib import Path

import jsonlines

from banks.registries import DirectoryPromptRegistry
from tqdm.auto import tqdm
from transformers import AutoTokenizer


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    ap.add_argument("--prompt_path", type=str, required=True, help="Path to the prompt file")
    ap.add_argument("--config", type=str, required=True, help="exact or verbatim")
    ap.add_argument("--port", type=int, required=True, help="Port number")
    ap.add_argument("--model", type=str, required=True, help="Model name")
    args = ap.parse_args()

    input_file = Path(args.input_file)

    registry = DirectoryPromptRegistry(Path(args.prompt_path), force_reindex=True)
    task_prompt = registry.get(name=f"task_{args.config}")
    sys_prompt = registry.get(name="sys")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    headers = {"Authorization": "Bearer sk_noreq", "Content-Type": "application/json"}
    
    data = []
    with jsonlines.open(input_file) as reader:    
        for example in tqdm(reader):
            prompt = task_prompt.text({"input": example["input"]})
            prompt = [
                {"role": "system", "content": sys_prompt.text()},
                {"role": "user", "content": prompt}
            ]
            prompt_chat = tokenizer.apply_chat_template(prompt, tokenize=False)
            prompt_tokens = tokenizer.tokenize(prompt_chat)
             
            # print(f"Prompt: {prompt}")
            # print(f"Prompt Tokens: {prompt_tokens}")
            # print(f"Length of Prompt Tokens: {len(prompt_tokens)} and Pre-length: {example["input_length"]}")
            response = requests.post(f"http://0.0.0.0:{args.port}/tokenize",
                                     headers=headers,
                                     json={"messages": prompt,
                                           "model": args.model,})
            ip_token_ids = response.json()["tokens"]
            ip_tokens = tokenizer.convert_ids_to_tokens(ip_token_ids)
            example["tokenized_input"] = ip_tokens
            assert len(ip_token_ids) == example["input_length"], f"Length mismatch: {len(ip_token_ids)} and {example['input_length']}"
            # print(f"Input Tokens: {ip_tokens}")
            data.append(example)
    
    with jsonlines.open(input_file, "w") as writer:
        writer.write_all(data)