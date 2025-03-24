import random
import numpy as np
import torch
from jinja2 import Template
from transformer_lens import HookedTransformer
from argparse import ArgumentParser
import jsonlines
from utils import combine_params, get_data, load_heads, render_prompt, get_gold_ans

def set_global_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ablate_head_hook(layer, head):
    def hook(value, hook):
        print(f"Ablating head {head} in layer {layer}, shape: {value.shape}")
        value[:, :, head, :] = 0  # zero-out the specific head
        return value
    return hook

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", help="choose model")
parser.add_argument("-v", "--version", dest="version", help="instruct or non-instruct model version")
parser.add_argument("-t", "--task", dest="task", help="induction before or after")
parser.add_argument("-tp", "--type", dest="type", help="ablate induction or anti-induction")
parser.add_argument("-l", "--length", dest="length", help="choose input length: 20, 30, 50, 100")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling parameter")

args = parser.parse_args()
model_name, task_path, version, data_path, ablation_type = combine_params(args)

# Set the global seed
set_global_seed(args.seed)

model = HookedTransformer.from_pretrained(model_name)
model.cfg.use_attn_result = True  # to use hook_result

data = get_data(data_path)[:100]
inp_length = len(data[0]['input'])

template_str = "{{ system }} {{ user_input }}"
system_path = 'templates/system.jinja'
template = Template(template_str)
heads_to_ablate = load_heads(model_name, ablation_type)

print(f"Loaded model: {model_name}")
print(f"Loaded input data from: {data_path}")
print(f"Chosen task: {task_path}")
print(f"Chosen ablation type: {ablation_type}")
print(f"Heads to ablate: {heads_to_ablate}")
print(f"Random seed: {args.seed}, top_k: {args.top_k}")

answers = []
for example in data:
    system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
    prompt = template.render(system=system_prompt, user_input=task_prompt)
    tokens = model.to_tokens(prompt)

    # Build ablation hooks
    hooks = [
        (f'blocks.{layer}.attn.hook_result', ablate_head_hook(layer, head))
        for layer, head in heads_to_ablate
    ]

    with model.hooks(fwd_hooks=hooks):
        if args.version == 'non-instruct':
            max_new = 2
        elif args.version == 'instruct':
            max_new = 5000

        # Use top-k sampling and do_sample=True, but remove the 'generator=' argument
        generated_tokens = model.generate(
            tokens,
            max_new_tokens=max_new,
            stop_at_eos=True,
            do_sample=True,
            top_k=args.top_k,
            temperature=0,  # adjust as desired
        )

    new_tokens = generated_tokens[0, tokens.shape[-1]:]
    generated_text = model.to_string(new_tokens)

    answer = {
        'input': example['input'],
        'gold_ans_char': get_gold_ans(example['input'], args.task),
        'full_answer': generated_text
    }
    answers.append(answer)

output_path = (
    'results/'
    + args.model + '_'
    + args.version + '_'
    + args.task + '_'
    + args.type + "_"
    + str(inp_length)
    + '.jsonl'
)
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(answers)
