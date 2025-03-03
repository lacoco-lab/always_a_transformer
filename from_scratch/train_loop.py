import os
import wandb
import hydra
from wandb_settings import settings
from omegaconf import DictConfig, OmegaConf
from data_utils import get_or_create_dataset, data_collator, compute_metrics

from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, TrainerCallback


class CustomEarlyStoppingCallback(TrainerCallback):
    """
    Custom early stopping callback that stops training if the eval_accuracy reaches
    1.0 in 'patience' successive evaluations.
    """
    def __init__(self, metric_name: str = "eval_accuracy", target_value: float = 1.0, patience: int = 3):
        self.metric_name = metric_name
        self.target_value = target_value
        self.patience = patience
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Check if the metric is available.
        if self.metric_name in metrics:
            value = metrics[self.metric_name]
            if value >= self.target_value:
                self.counter += 1
                print(f"Evaluation {state.global_step}: {self.metric_name}={value}. Counter = {self.counter}")
                if self.counter >= self.patience:
                    print(f"Early stopping triggered: {self.metric_name} reached {self.target_value} for {self.counter} successive evaluations.")
                    control.should_early_stop = True
                    control.should_training_stop = True
            else:
                # Reset counter if the metric doesn't reach target.
                self.counter = 0
        return control

# --------------------------
# Main with Hydra and Wandb Integration
# --------------------------
@hydra.main(config_path='configs', config_name="defaults", version_base=None)
def main(cfg: DictConfig):
    # Then in your code:
    if cfg.wandb.use_wandb:
        wandb.login(key=settings.WANDB_API_KEY)
        wandb.init(project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=OmegaConf.to_container(cfg, resolve=True))
        report_target = ["wandb"]
    else:
        report_target = []

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
    # print("model done")
    # Create training and validation datasets.
    train_dataset = get_or_create_dataset(cfg.dataset.train, "train")
    val_dataset = get_or_create_dataset(cfg.dataset.val, "val")

    
    # Set up training arguments with Wandb reporting.
    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        eval_strategy=cfg.train.evaluation_strategy,
        eval_steps=cfg.train.eval_steps,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        report_to=report_target,
    )
    
    # Create the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CustomEarlyStoppingCallback(metric_name="eval_accuracy", target_value=1.0, patience=3)]

    )
    # Train the model.
    trainer.train()

    # Save the model locally with a meaningful file name. TO DO 
    save_dir = os.path.join(
        cfg.train.output_dir,
        f"model_{cfg.dataset.train.name}_min{cfg.dataset.train.min_len}_max{cfg.dataset.train.max_len}_"
        f"vocab{cfg.model.vocab_size}_layers{cfg.model.n_layer}_heads{cfg.model.n_head}"
    )
    trainer.save_model(save_dir)
    print(f"Model saved to {save_dir}")    

    # Evaluate on both training and validation datasets.
    train_eval_results = trainer.evaluate(train_dataset)
    eval_results = trainer.evaluate(val_dataset)    

    if cfg.wandb.use_wandb:
        # Force logging of accuracy metrics.
        # wandb.log({
        #     **{f"train_{k}_again": v for k, v in train_eval_results.items()}
        # })
        wandb.finish()
    else:
        print("Training results:", train_eval_results)
        print("Validation results:", eval_results)

if __name__ == "__main__":
    main()