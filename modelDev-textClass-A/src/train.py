import numpy as np
import evaluate
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from src.utils.callbacks import LoggingCallback, DebugCallback
from sklearn.metrics import f1_score, precision_score, recall_score


class Trainer_Object:
    def __init__(self, config, model_wrapper):
        self.config = config
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.data_collator = model_wrapper.get_data_collator()
        self.metric = evaluate.load("f1")
        

    def run(self, ds_tok):
        
        args = TrainingArguments(
            output_dir                    =   str(self.config['output_dir']),
            learning_rate                 = float(self.config['training_args']['learning_rate']),
            per_device_train_batch_size   =   int(self.config['training_args']['per_device_train_batch_size']),
            per_device_eval_batch_size    =   int(self.config['training_args']['per_device_eval_batch_size']),
            gradient_accumulation_steps   =   int(self.config['training_args']['gradient_accumulation_steps']),
            num_train_epochs              =   int(self.config['training_args']['num_train_epochs']),
            gradient_checkpointing        =  bool(self.config['training_args']['gradient_checkpointing']),
            warmup_ratio                  = float(self.config['training_args']['warmup_ratio']),
            weight_decay                  = float(self.config['training_args']['weight_decay']),
            eval_strategy                 =   str(self.config['training_args']['eval_strategy']),
            save_strategy                 =   str(self.config['training_args']['save_strategy']),
            save_total_limit              =   int(self.config['training_args']['save_total_limit']),
            metric_for_best_model         =   str(self.config['training_args']['metric_for_best_model']),
            load_best_model_at_end        =  bool(self.config['training_args']['load_best_model_at_end']),
            #logging_dir                  =   str(self.config['training_args']['logging_dir']),
            logging_steps                 =   int(self.config['training_args']['logging_steps']),
            bf16                          =  bool(self.config['training_args']['bf16']),
            push_to_hub                   =  bool(self.config['training_args']['push_to_hub']), # type: ignore
        )

        trainer = Trainer(
            model             = self.model,
            args              = args,
            train_dataset     = ds_tok["train"],
            eval_dataset      = ds_tok["test"],
            tokenizer         = self.tokenizer,
            data_collator     = self.data_collator,
            compute_metrics   = self.compute_metrics,
            callbacks         = [
                LoggingCallback(self.config['log_dir'], log_training_steps=self.config['training_args']['log_training_steps']),
                DebugCallback()
                ],
        )
        
        trainer.train()


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
        
        # Standard 0.5 threshold predictions
        preds = (probabilities >= 0.5).astype(int)
        labels = labels.astype(int)

        # Basic metrics
        results = {
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        }
        
        # Per-class F1 scores for monitoring individual class performance
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        if isinstance(per_class_f1, np.ndarray):
            results.update({f"f1_class_{i}": score for i, score in enumerate(per_class_f1)})
        
        return results