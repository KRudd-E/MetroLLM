import numpy as np
import evaluate
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from src.utils.callbacks import LoggingCallback, DebugCallback

class Trainer_Object:
    def __init__(self, config, model_wrapper):
        self.config = config
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.data_collator = model_wrapper.get_data_collator()
        self.metric = evaluate.load("f1")
        
    def run(self, ds_tok):
        
        args = TrainingArguments(
            output_dir                    = self.config['training_args']['output_dir'],
            learning_rate                 = self.config['training_args']['learning rate'],
            per_device_train_batch_size   = self.config['training_args']['per_device_train_batch_size'],
            per_device_eval_batch_size    = self.config['training_args']['per_device_eval_batch_size'],
            num_train_epochs              = self.config['training_args']['epochs'],
            warmup_ratio                  = self.config['training_args']['warmup_ratio'],
            weight_decay                  = self.config['training_args']['weight_decay'],
            eval_strategy                 = self.config['training_args']['eval_strategy'],
            save_strategy                 = self.config['training_args']['save_strategy'],
            save_total_limit              = self.config['training_args']['save_total_limit'],
            metric_for_best_model         = self.config['training_args']['metric_for_best_model'],
            load_best_model_at_end        = self.config['training_args']['load_best_model_at_end'],
            logging_dir                   = self.config['training_args']['log_dir'],
            logging_steps                 = self.config['training_args']['logging_steps'],
            push_to_hub                   = self.config['training_args']['push_to_hub'], # type: ignore
        )
        
        trainer = Trainer(
            model             = self.model,
            args              = args,
            train_dataset     = ds_tok["train"],
            eval_dataset      = ds_tok["test"],
            tokenizer         = self.tokenizer,
            data_collator     = self.data_collator,
            compute_metrics   = self.compute_metrics,
            callbacks         = [LoggingCallback(self.config), DebugCallback()],
        )
        trainer.train()


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))           # sigmoid
        preds = (probs >= 0.5).astype(int)          # threshold; tune if needed
        result = self.metric.compute(
            predictions=preds, references=labels, average="micro"
        )
        return {"f1_micro": result["f1"] if result and "f1" in result else 0.0}