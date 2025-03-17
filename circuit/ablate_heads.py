import transformer_lens
from jinja2 import Template
from transformer_lens import HookedTransformer
from argparse import ArgumentParser
import torch
import jsonlines
from utils import combine_params, get_data, load_heads, render_prompt, get_gold_ans

def ablate_head_hook(layer, head):
    def hook(value, hook):
        print(value.shape)
        value[:, :, head, :] = 0  # zero-out the specific head
        return value
    return hook

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model",
                        help="choose model")
parser.add_argument("-v", "--version", dest="version", help="instruct or non-instruct model version")

parser.add_argument("-t", "--task", dest="task", help="induction before or after")
parser.add_argument("-tp", "--type", dest="type", help="ablate induction or anti-induction")
args = parser.parse_args()
model_name, task_path, version, data_path, ablation_type = combine_params(args)

model = HookedTransformer.from_pretrained(model_name)
heads_to_ablate = load_heads(model_name, ablation_type)
data = get_data(data_path)

template_str = "{{ system }} {{ user_input }}:"
system_path = 'templates/system.jinja'
template = Template(template_str)
heads_to_ablate = load_heads(model_name)
print(model.cfg)

answers = []
for example in data:
    system_prompt, task_prompt = render_prompt(system_path, task_path, example['input'])
    prompt = template.render(system=system_prompt, user_input=task_prompt)
    tokens = model.to_tokens(prompt)

    hooks = [
        (f'blocks.{layer}.attn.hook_v', ablate_head_hook(layer, head))
        for layer, head in heads_to_ablate
    ]

    with model.hooks(fwd_hooks=hooks):
        generated_tokens = model.generate(tokens, max_new_tokens=2, stop_at_eos=True)

    new_tokens = generated_tokens[0, tokens.shape[-1]:]
    generated_text = model.to_string(new_tokens)
    
    answer = {
        'input': example['input'],
        'gold_ans_char': get_gold_ans(example['input'], args.task),
        'full_answer': generated_text,
        'answer': model.to_string(new_tokens[1]),
        'is_correct': get_gold_ans(example['input'], args.task) == model.to_string(new_tokens[1]),
    }
    answers.append(answer)
    break

output_path = 'results/' + args.model + '_' + args.version + '_' + args.task + '_' + args.type + '.jsonl'
with jsonlines.open(output_path, mode='w') as writer:
    writer.write_all(answers)

