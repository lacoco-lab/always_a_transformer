from data import UniqueCopyDataset, ReverseUniqueCopyDataset, FinetuningDataset, FinetuningDatasetSameAsNotFinetuning
from utils import get_logging_function, set_seed, subspace_angles
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import click
import torch
import json
from pathlib import Path
import numpy as np
from datetime import datetime

@click.command()
@click.option('--config_path', default="config/patching.yaml", help="Path to config file")
def run(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    torch.set_printoptions(sci_mode=False, precision=5)
    device = torch.device("cuda") if config_dict["device"] == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    set_seed(config_dict["seed"])
    output_dir: Path = Path(config_dict["output_dir"]) / config_dict["exp_name"] / datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    output_config_path = output_dir / "args.json"
    with open(output_config_path, "w") as f:
        json.dump(config_dict, f)
    output_dict = {}

    logger = get_logging_function(output_dir)
    try:
        logger(config_dict)
        tokenizer = AutoTokenizer.from_pretrained(config_dict["model"])
        model = AutoModelForCausalLM.from_pretrained(config_dict["model"])
        model.eval()
        model.to(device)
        if config_dict["finetuned_model"]:
            dataset = FinetuningDatasetSameAsNotFinetuning(tokenizer, num_instances=config_dict["num_instances"], seed=config_dict["seed"],
                                            length_range=config_dict["length_range"], n_shot=config_dict["n_shot"],
                                            dataset_name=config_dict["dataset"])
        else:
            if config_dict["dataset"] == "unique_copy":
                dataset = UniqueCopyDataset(tokenizer, num_instances=config_dict["num_instances"], seed=config_dict["seed"],
                                            length_range=config_dict["length_range"], sep_token=config_dict["sep_token"],
                                            fewshot_sep_token=config_dict["fewshot_sep_token"], n_shot=config_dict["n_shot"])
            elif config_dict["dataset"] == "reversed_unique_copy":
                dataset = ReverseUniqueCopyDataset(tokenizer, num_instances=config_dict["num_instances"], seed=config_dict["seed"],
                                                length_range=config_dict["length_range"], sep_token=config_dict["sep_token"],
                                                fewshot_sep_token=config_dict["fewshot_sep_token"], n_shot=config_dict["n_shot"])

        num_layers_in_model = (len(model.model.layers) if not config_dict["small_model"] else len(model.transformer.h))
        if config_dict["calc_attention"]:
            sum_attention_previous_token = np.zeros((num_layers_in_model, model.config.num_attention_heads))
            sum_attention_previous_same_token = np.zeros((num_layers_in_model, model.config.num_attention_heads))
            sum_attention_previous_induction_token = np.zeros((num_layers_in_model, model.config.num_attention_heads))
        
        for item_in_output_dict in ["inputs_str", "targets_str", "inputs", "targets", "predictions", "predictions_probs", "correct"]:
            output_dict[item_in_output_dict] = []
        
        num_tokens_to_sum_over_nwp = 0
        num_tokens_to_sum_over_ind = 0
        for item_i, item in enumerate(dataset):
            instance, token_ids, tokens, label, label_ids, label_tokens, len_of_str, beginning_of_first_part, beginning_of_second_part = item
            label_tensor = torch.tensor([label_ids], device="cpu")
            logger("instance", instance)
            logger("label", label)
            with torch.inference_mode():
                outputs = model(
                    input_ids=torch.tensor([token_ids], device=device),
                    output_attentions=True
                )
            logits = outputs.logits.detach().cpu()
            logger("predictions", "".join(tokenizer.convert_ids_to_tokens(logits.argmax(dim=2)[label_tensor != tokenizer.pad_token_id])))
            output_dict["inputs_str"].append(instance)
            output_dict["targets_str"].append(label)
            output_dict["predictions"].append(["".join(tokenizer.convert_ids_to_tokens(logits.argmax(dim=2).tolist()[0]))])
            output_dict["predictions_probs"].append([" ".join(map(str,
                    torch.nn.functional.softmax(logits, dim=2).max(dim=2).values.tolist()[0]))])
            output_dict["correct"].append((logits.argmax(dim=2)[label_tensor != tokenizer.pad_token_id] == \
                                            label_tensor[label_tensor != tokenizer.pad_token_id]).all().item())
            logger("correct", output_dict["correct"][-1])
            attentions = outputs.attentions
            if item_i < 1 and config_dict["plot_attentions"]:
                attentions_output_dir = output_dir / "attentions"
                plt.figure(figsize=(6, 6))
                for layer in range(len(attentions)):
                    for head in range(attentions[layer].shape[1]):
                        cur_attentions_output_dir = attentions_output_dir / f"{layer}" / f"{head}"
                        cur_attentions_output_dir.mkdir(parents=True, exist_ok=True)
                        plt.imshow(attentions[layer][0, head, :, :].detach().cpu(), cmap='hot')
                        plt.colorbar()
                        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
                        plt.yticks(range(len(tokens)), tokens)
                        plt.tight_layout()
                        plt.savefig(cur_attentions_output_dir / f"item-{item_i}.png")
                        plt.close()
            if config_dict["calc_attention"]:
                for layer, attn_map in enumerate(attentions):
                    for bsz in range(attn_map.shape[0]):
                        for head in range(attn_map.shape[1]):
                            for next_token in range(beginning_of_first_part + 1, beginning_of_first_part + len_of_str):
                                sum_attention_previous_token[layer, head] += attn_map[bsz, head, next_token, next_token - 1].item()
                                if layer == 0 and head == 0:
                                    num_tokens_to_sum_over_nwp += 1
                            for next_token in range(beginning_of_second_part + 1, beginning_of_second_part + len_of_str - 1):
                                prev_same_token = (next_token - beginning_of_second_part + beginning_of_first_part if config_dict["dataset"] == "unique_copy" else 2 * beginning_of_second_part - next_token - 2)
                                assert token_ids[prev_same_token] == token_ids[next_token], (prev_same_token, next_token, token_ids[prev_same_token], token_ids[next_token], token_ids)
                                sum_attention_previous_same_token[layer, head] += attn_map[bsz, head, next_token, prev_same_token].item()
                                sum_attention_previous_induction_token[layer, head] += attn_map[bsz, head, next_token, prev_same_token + 1].item()
                                if layer == 0 and head == 0:
                                    num_tokens_to_sum_over_ind += 1

        output_dict["accuracy"] = np.mean(output_dict["correct"]).item()
        output_dict["num_tokens_to_sum_over_ind"] = num_tokens_to_sum_over_ind
        output_dict["num_tokens_to_sum_over_nwp"] = num_tokens_to_sum_over_nwp

        if config_dict["calc_attention"]:
            alignment_plots_output_dir = output_dir / "alignment_plots"
            alignment_plots_output_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 6))
            for plot_what, plot_name in zip(
                [sum_attention_previous_token, sum_attention_previous_same_token, sum_attention_previous_induction_token],
                ["sum_attention_previous_token", "sum_attention_previous_same_token", "sum_attention_previous_induction_token"]
            ):
                plt.imshow(plot_what / (num_tokens_to_sum_over_nwp if plot_name == "sum_attention_previous_token" else num_tokens_to_sum_over_ind), cmap='hot')
                plt.colorbar()
                layers = [str(l) for l in range(num_layers_in_model)]
                heads = [str(h) for h in range(model.config.num_attention_heads)]
                plt.xticks(range(len(heads)), heads, rotation=45, ha='right')
                plt.yticks(range(len(layers)), layers)
                plt.tight_layout()
                plt.savefig(alignment_plots_output_dir / f"{plot_name}_average.png")
                plt.close()
                output_dict[plot_name] = plot_what.tolist()
                output_dict[plot_name + "_averaged"] = (plot_what / (num_tokens_to_sum_over_nwp if plot_name == "sum_attention_previous_token" else num_tokens_to_sum_over_ind)).tolist()
                
        output_file = output_dir / "output.json"
        with open(output_file, "w") as f:
            json.dump(output_dict, f, indent=4)
    finally:
        logger.cleanup()



if __name__ == '__main__':
    run()