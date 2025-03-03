import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments
from data_utils import get_or_create_dataset, data_collator, compute_metrics

# --------------------------
# Main with Hydra and Wandb Integration
# --------------------------
@hydra.main(config_path='configs', config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    # Initialize Wandb.
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True))
    
    # Create the model.
    model_config = GPT2Config(
        vocab_size=cfg.model.vocab_size,
        n_positions=cfg.model.n_positions,
        n_ctx=cfg.model.n_ctx,
        n_embd=cfg.model.n_embd,
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        pad_token_id=cfg.model.pad_token_id,
    )
    model = GPT2LMHeadModel(model_config)
    print("model-", model)
    # Create training and validation datasets.
    train_dataset = get_or_create_dataset(cfg.dataset.train, "train")
    val_dataset = get_or_create_dataset(cfg.dataset.val, "val")

    
    # Set up training arguments with Wandb reporting.
    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        evaluation_strategy=cfg.train.evaluation_strategy,
        eval_steps=cfg.train.eval_steps,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        report_to=["wandb"],
    )
    
    # Create the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate.
    trainer.train()
    eval_results = trainer.evaluate()
    print("Validation results:", eval_results)
    wandb.finish()

if __name__ == "__main__":
    main()