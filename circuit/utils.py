import json
import torch
from jinja2 import Template

def combine_params(args):
    """
    Determine parameters to run the experiment: model, model version and task
    :param args: dict from input
    :return: model, task_path, version
    """
    
    if args.model == 'llama' and args.version == 'instruct':
        model = 'meta-llama/Meta-Llama-3-8B-Instruct'
        version = 'instruct'
    elif args.model == 'llama' and args.version != 'instruct':
        model = 'meta-llama/Meta-Llama-3-8B'
        version = 'non-instruct'
    elif args.model == 'pythia':
        model = 'EleutherAI/pythia-1.4b-deduped'
        version = 'non-instruct'
        
    if args.task == 'before' and version == 'instruct':
        task_path = 'templates/ind_before_chat.jinja'
    elif args.task == 'after' and version == 'instruct':
        task_path = 'templates/ind_after_chat.jinja'
    elif args.task == 'before' and version == 'non-instruct':
        task_path = 'templates/ind_before_completion.jinja'
    elif args.task == 'after' and version == 'non-instruct':
        task_path = 'templates/ind_after_completion.jinja'
        
    data_path = '../datasets/' + str(args.length) + '/flipflop_inductionhead/data.jsonl'
    ablation_type = args.type
        
    return model, task_path, version, data_path, ablation_type


def get_data(path):
    """
    Load dataset.
    :param path: dir
    :return: arr
    """
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def load_heads(model, type):
    """
    Load heads to ablate for a respective model.
    :param model: str, model name
    :return: arr of tuples
    """
    with open('heads_to_ablate.json', 'r') as file:
        model_heads = json.load(file)
        heads_json = model_heads[model][type]
        
        heads = []
        for layer in heads_json:
            for head in heads_json[layer]:
                heads.append((int(layer), int(head)))
    return []
        


def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return Template(file.read())


def load_input(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def render_prompt(system_path, task_path, input_string):
    """
    Render prompt templates using Jinja2 templates.
    :param system_path: dir path to the system prompt
    :param task_path: dir path to the task prompt with CoT
    :param input_string: str, input
    :return: 
    """

    system_template = load_template(system_path)
    task_template = load_template(task_path)

    task_prompt = task_template.render(input_string=input_string)

    system_prompt = system_template.render()

    return system_prompt, task_prompt


def get_gold_ans(str, task):
    """
    Get the gold answer for the induction head task.
    :param str: input string
    :param task: induction before or after
    :return: char
    """
    if task == 'before':
        return str[str.index('w')-1]
    elif task == 'after':
        return str[str.index('w')+1]