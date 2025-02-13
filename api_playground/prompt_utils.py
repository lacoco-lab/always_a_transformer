from jinja2 import Template


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
