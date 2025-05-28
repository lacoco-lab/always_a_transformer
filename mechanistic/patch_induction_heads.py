from data import FinetuningDatasetSameAsNotFinetuning, UniqueCopyDataset, ReverseUniqueCopyDataset
from utils import get_logging_function, set_seed
from model import ModelWithHooks, BigModelWithHooks
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import click
import torch
import json
from pathlib import Path
import numpy as np

@click.command()
@click.option('--config_path', default="config/patching.yaml", help="Path to config file")
def run(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    torch.set_printoptions(sci_mode=False, precision=5)
    device = torch.device("cuda") if config_dict["device"] == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    set_seed(config_dict["seed"])
    output_dir: Path = Path(config_dict["output_dir"]) / config_dict["exp_name"]
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
        model.config.use_cache = False
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
        if config_dict["small_model"]:
            hooked_model = ModelWithHooks(model, logger=logger,
                                          remove_only_either_key_or_value=config_dict["remove_only_either_key_or_value"],
                                          task=config_dict["dataset"])
        else:
            hooked_model = BigModelWithHooks(model, logger=logger,
                                             remove_only_either_key_or_value=config_dict["remove_only_either_key_or_value"],
                                             task=config_dict["dataset"])
        
        
        for item_in_output_dict in ["inputs_str", "targets_str",
                                    "predictions_remove_induction", "predictions_probs_remove_induction",
                                    "predictions_remove_antiinduction", "predictions_probs_remove_antiinduction",
                                    "correct_remove_induction", "loss_remove_induction",
                                    "correct_no_intervention", "loss_no_intervention",
                                    "correct_remove_antiinduction", "loss_remove_antiinduction",]:
            output_dict[item_in_output_dict] = []
        
        for item_i, item in enumerate(dataset):
            instance, token_ids, tokens, label, label_ids, label_tokens, len_of_str, beginning_of_first_part, beginning_of_second_part = item
            label_tensor = torch.tensor([label_ids], device="cpu")
            output_dict["inputs_str"].append(instance)
            output_dict["targets_str"].append(label)
            logger("instance", instance)
            logger("label", len(label), label)
            logger("label ids", len(label_ids), label_ids)

            hooked_model.do_not_remove_anything(token_ids, False)
            with torch.inference_mode():
                outputs_no_intervention = hooked_model(
                    input_ids=torch.tensor([token_ids], device=device),
                    # output_attentions=True
                )
            logits = outputs_no_intervention.logits.detach().cpu()
            loss = torch.nn.functional.cross_entropy(
                logits.permute(1, 2, 0),
                label_tensor.permute(1, 0),
                ignore_index=tokenizer.pad_token_id
            )
            logger("loss_no_intervention", loss)
            logger("targets_no_intervention", "".join(tokens))
            logger("predictions_no_intervention", "".join(tokenizer.convert_ids_to_tokens(logits.argmax(dim=2)[0])))
            logger("predictions_no_intervention_only_labels", "".join(tokenizer.convert_ids_to_tokens(logits.argmax(dim=2)[label_tensor != tokenizer.pad_token_id])))

            output_dict["correct_no_intervention"].append((logits.argmax(dim=2)[label_tensor != tokenizer.pad_token_id] == \
                                            label_tensor[label_tensor != tokenizer.pad_token_id]).all().item())
            output_dict["loss_no_intervention"].append(loss.item())
            logger("correct_no_intervention", output_dict["correct_no_intervention"][-1])

            hooked_model.remove_prev_token_heads(token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation=True, remove_with_zeros=True)
            with torch.inference_mode():
                hooked_model(
                    input_ids=torch.tensor([token_ids], device=device),
                )
            for heads_to_remove in ["induction", "antiinduction"]:
                if heads_to_remove == "induction":
                    hooked_model.remove_induction_heads(token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation=False, remove_with_zeros=False)
                elif heads_to_remove == "antiinduction":
                    hooked_model.remove_antiinduction_heads(token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation=False, remove_with_zeros=False)
                else:
                    raise NotImplementedError()
                
                with torch.inference_mode():
                    outputs = hooked_model(
                        input_ids=torch.tensor([token_ids], device=device),
                        # output_attentions=True
                    )
                logits = outputs.logits.detach().cpu()
                loss = torch.nn.functional.cross_entropy(
                    logits.permute(1, 2, 0),
                    label_tensor.permute(1, 0),
                    ignore_index=tokenizer.pad_token_id
                )
                output_dict[f"predictions_remove_{heads_to_remove}"].append(tokenizer.convert_ids_to_tokens(logits.argmax(dim=2)[0]))
                output_dict[f"predictions_probs_remove_{heads_to_remove}"].append([" ".join(map(str,
                        torch.nn.functional.softmax(logits, dim=2).max(dim=2).values.tolist()[0]))])
                output_dict[f"correct_remove_{heads_to_remove}"].append((logits.argmax(dim=2)[label_tensor != tokenizer.pad_token_id] == \
                                                label_tensor[label_tensor != tokenizer.pad_token_id]).all().item())
                output_dict[f"loss_remove_{heads_to_remove}"].append(loss.item())
                logger(f"loss_remove_{heads_to_remove}", loss)
                logger(f"predictions_remove_{heads_to_remove}", tokenizer.convert_ids_to_tokens(logits.argmax(dim=2)[0]))
                logger(f"correct_remove_{heads_to_remove}", output_dict[f"correct_remove_{heads_to_remove}"][-1])

        for intervention_name in ["no_intervention", "remove_antiinduction", "remove_induction"]:
            output_dict[f"loss_mean_{intervention_name}"] = np.mean(output_dict[f"loss_{intervention_name}"])
            output_dict[f"accuracy_{intervention_name}"] = np.mean(output_dict[f"correct_{intervention_name}"])
            
        output_file = output_dir / "output.json"
        with open(output_file, "w") as f:
            json.dump(output_dict, f, indent=4)
    finally:
        logger.cleanup()



if __name__ == '__main__':
    run()