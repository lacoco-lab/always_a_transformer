## Finetuning

This directory contains the scripts to finetune the GPT-2 style models on the tasks. The scripts are designed to be run from the command line. The directory structure is as follows:

```
finetuning/
├── copy_str
│   ├── finetune_copy.py
│   ├── generate_data.py
├── flipflop
│   ├── finetune_flipflop.py
│   ├── generate_data.py
├── retrieval
│   ├── finetune_retrieval.py
│   ├── generate_data.py
```

### Tasks


### Copy finetuning

You can run the finetuning scripts for the copy task using the following command. 
```bash
python finetuning/copy_str/finetune_copy.py --data_path <train dataset jsonl path> --test_file_paths <test dataset jsonl paths> --model_name_or_path <hf or local path> --output_dir <output path> --per_device_train_batch_size <batch size> --wandb_project <project> --wandb_entity <enter entity> --wandb_run_name <run name> --do_train --do_eval --do_inference --use_char_tokenization --seed <SEED>
```

### FlipFlop finetuning
You can run the finetuning scripts for the flipflop task using the following command.
```bash
python finetuning/flipflop/finetune_flipflop.py --num_train_epochs <num epochs> --data_path <train dataset jsonl path> --test_file_paths <test dataset jsonl paths> --model_name_or_path <hf or local path> --output_dir <output path> --per_device_train_batch_size <batch size> --wandb_project <project> --wandb_entity <enter entity> --wandb_run_name <run name> --do_train --do_eval --do_inference --use_char_tokenization --seed <SEED>
```

### Inductionhead finetuning
You can run the finetuning scripts for the Inductionhead task using the following command.
```bash
python finetuning/inductionhead/finetune_inductionhead.py --num_train_epochs <num epochs> --data_path <train dataset jsonl path> --test_file_path <test dataset jsonl paths> --model_name_or_path <hf or local path> --output_dir <output path> --per_device_train_batch_size <batch size> --wandb_project <project> --wandb_entity <enter entity> --wandb_run_name <run name> --do_train --do_eval --do_inference --use_char_tokenization --seed <SEED>
```

#### Notes:
- The `--use_char_tokenization` flag is used to enable character-level tokenization. This should be used for all the tasks.
- The `--seed` flag is used to set the random seed for reproducibility. 
- The `--wandb_project`, `--wandb_entity`, and `--wandb_run_name` flags are used to log the results to Weights and Biases. Setting `WANDB_API_KEY` in your environment variables is required to use this feature.
