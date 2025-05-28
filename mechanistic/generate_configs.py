import yaml
import click
from pathlib import Path

@click.command()
@click.option('--generate', help="what configs to generate")
@click.option('--config_output_dir', help="where to put configs")
@click.option('--output_dir', help="where to put job outputs")
def run(generate, output_dir, config_output_dir):
    config_output_dir = Path(config_output_dir) / generate
    config_output_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_to_path = {
        "gemma": "google/gemma-3-1b-pt",
        "gpt2": "karpathy/gpt2_1558M_final4_hf",
        "llama": "meta-llama/Llama-3.1-8B",
        "qwen": "Qwen/Qwen2.5-7B",
        "finetuned-unique_copy": "TBD: LOCAL_PATH_TO_FINETUNED_MODEL",
        "finetuned-reversed_unique_copy": "TBD: LOCAL_PATH_TO_FINETUNED_MODEL",
    }
    all_paths_to_configs = []
    if generate == "alignment_different_lengths":
        for task in ["unique_copy", "reversed_unique_copy"]:
            for model in ["gemma", "gpt2", "llama", "qwen"]:
                for length in [3, 4, 5, 6, 7, 8, 9, 10]:
                    config = {
                        "exp_name": f"{generate}_{task}_{model}_{length}",
                        "dataset": task,
                        "model": model_to_path[model],
                        "num_instances": 100,
                        "sep_token": ",",
                        "fewshot_sep_token": ".",
                        "n_shot": 11,
                        "output_dir": output_dir,
                        "device": "cuda",
                        "length_range": [length, length],
                        "seed": 0,
                        "plot_attentions": False,
                        "calc_attention": True,
                        "calc_alignment": False,
                        "finetuned_model": False,
                        "small_model": (model == "gpt2")
                    }
                    path_to_config = Path(config_output_dir) / f"{config['exp_name']}.yaml"
                    print(path_to_config)
                    with open(path_to_config, 'w') as yaml_file:
                        yaml.dump(config, yaml_file, default_flow_style=False)
                    all_paths_to_configs.append(str(path_to_config))
    elif generate == "alignment_finetuned_same_dataset":
        all_paths_to_configs = []
        for task in ["unique_copy", "reversed_unique_copy"]:
            for model in ["finetuned-unique_copy", "finetuned-reversed_unique_copy", "gpt2"]:
                if "finetuned" in model and task not in model:
                    continue
                for length in [3, 4, 5, 6, 7, 8, 9, 10]:
                    config = {
                        "exp_name": f"{generate}_{task}_{model}_{length}",
                        "dataset": task,
                        "model": model_to_path[model],
                        "num_instances": 100,
                        "sep_token": ",",
                        "fewshot_sep_token": ".",
                        "n_shot": 11,
                        "output_dir": output_dir,
                        "device": "cuda",
                        "length_range": [length, length],
                        "seed": 0,
                        "plot_attentions": False,
                        "calc_attention": True,
                        "calc_alignment": False,
                        "finetuned_model": ("finetuned" in model),
                        "small_model": True,
                        "use_finetuning_dataset": False,
                    }
                    path_to_config = Path(config_output_dir) / f"{config['exp_name']}.yaml"
                    print(path_to_config)
                    with open(path_to_config, 'w') as yaml_file:
                        yaml.dump(config, yaml_file, default_flow_style=False)
                    all_paths_to_configs.append(str(path_to_config))
    elif generate == "patching_remove_only_either_key_or_value":
        all_paths_to_configs = []
        for task in ["unique_copy", "reversed_unique_copy"]:
            for model in ["gemma", "gpt2", "llama", "qwen", "finetuned-unique_copy", "finetuned-reversed_unique_copy"]:
                if "finetuned" in model and task not in model:
                    continue
                for length in [3, 5, 7, 9]:
                    config = {
                        "exp_name": f"{generate}_{task}_{model}_{length}",
                        "dataset": task,
                        "model": model_to_path[model],
                        "num_instances": 100,
                        "sep_token": ",",
                        "fewshot_sep_token": ".",
                        "n_shot": 11,
                        "output_dir": output_dir,
                        "device": "cuda",
                        "length_range": [length, length],
                        "seed": 0,
                        "finetuned_model": ("finetuned" in model),
                        "small_model": (model == "gpt2" or "finetuned" in model),
                        "remove_only_either_key_or_value": True
                    }
                    path_to_config = Path(config_output_dir) / f"{config['exp_name']}.yaml"
                    print(path_to_config)
                    with open(path_to_config, 'w') as yaml_file:
                        yaml.dump(config, yaml_file, default_flow_style=False)
                    all_paths_to_configs.append(str(path_to_config))
    path_to_all_configs = Path(config_output_dir) / "all_configs_list.list"
    with open(path_to_all_configs, "w") as file:
        file.writelines("\n".join(all_paths_to_configs))
if __name__ == '__main__':
    run()
