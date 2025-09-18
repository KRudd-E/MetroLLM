import numpy as np
import evaluate
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from src.utils.callbacks import LoggingCallback, DebugCallback
from sklearn.metrics import f1_score, precision_score, recall_score # , average_precision_score, roc_auc_score
import torch
from src.model_wrapper import predict_with_count_limit


class Trainer_Object:
    def __init__(self, config, model_wrapper, pos_weights=None):
        self.config = config
        
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.data_collator = model_wrapper.get_data_collator()
        
        self.metric = evaluate.load("f1")
        
        self.pos_weights = pos_weights
        

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
            #eval_steps                    =   int(self.config['training_args']['eval_steps']),
            #save_steps                    =   int(self.config['training_args']['save_steps']),     #? Unused if strategies are epoch
            #logging_steps                 =   int(self.config['training_args']['logging_steps']),
            logging_strategy              =   str(self.config['training_args']['logging_strategy']),
            save_total_limit              =   int(self.config['training_args']['save_total_limit']),
            metric_for_best_model         =   str(self.config['training_args']['metric_for_best_model']),
            load_best_model_at_end        =  bool(self.config['training_args']['load_best_model_at_end']),
            bf16                          =  bool(self.config['training_args']['bf16']),
            push_to_hub                   =  bool(self.config['training_args']['push_to_hub']),
            save_safetensors              =  bool(self.config['training_args']['save_safetensors']),
            max_grad_norm                 =  float(self.config['training_args']['max_grad_norm']),
            lr_scheduler_type             =    str(self.config['training_args']['lr_scheduler_type']),
            # label_smoothing_factor        =  float(self.config['training_args']['label_smoothing_factor']),
            )

        trainer = Trainer(
            model             = self.model,
            args              = args,
            train_dataset     = ds_tok["train"],
            eval_dataset      = ds_tok["test"],
            #tokenizer         = self.tokenizer,
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
        
        #** Standard threshold **#
        preds_standard = (probabilities >= self.config['model']['threshold']).astype(int)
        
        #** Limited count prediction **#
        logits_tensor = torch.tensor(logits)
        preds_limited = predict_with_count_limit(logits_tensor, self.config['model']['threshold'], self.config['model']['max_labels'])
        preds_limited = preds_limited.numpy().astype(int)
        
        labels = labels.astype(int)

        #** Standard metrics **#
        results = {
            "f1_micro": f1_score(labels, preds_standard, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds_standard, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds_standard, average="weighted", zero_division=0),
            "precision_micro": precision_score(labels, preds_standard, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds_standard, average="micro", zero_division=0),
            "precision_macro": precision_score(labels, preds_standard, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds_standard, average="macro", zero_division=0),
        }
        
        #** Limited count metrics **#
        results.update({
            "f1_micro_limited": f1_score(labels, preds_limited, average="micro", zero_division=0),
            "f1_macro_limited": f1_score(labels, preds_limited, average="macro", zero_division=0),
            "f1_weighted_limited": f1_score(labels, preds_limited, average="weighted", zero_division=0),
            "precision_micro_limited": precision_score(labels, preds_limited, average="micro", zero_division=0),
            "recall_micro_limited": recall_score(labels, preds_limited, average="micro", zero_division=0),
            "precision_macro_limited": precision_score(labels, preds_limited, average="macro", zero_division=0),
            "recall_macro_limited": recall_score(labels, preds_limited, average="macro", zero_division=0),
        })
        
        #** Prediction count statistics **#
        pred_counts_standard = np.sum(preds_standard, axis=1)
        pred_counts_limited = np.sum(preds_limited, axis=1)
        true_counts = np.sum(labels, axis=1)
        
        results.update({
            "avg_pred_count_standard": np.mean(pred_counts_standard),
            "avg_pred_count_limited": np.mean(pred_counts_limited),
            "avg_true_count": np.mean(true_counts),
            "max_pred_count_standard": np.max(pred_counts_standard),
            "max_pred_count_limited": np.max(pred_counts_limited),
        })
        
        #** Weighted eval_loss with count penalty **#
        if self.pos_weights is not None:
            
            #** WBCE loos **#
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weights.to(logits.device if hasattr(logits, "device") else "cpu"),
                reduction="mean",
            )
            
            logits_t = torch.tensor(logits)
            labels_t = torch.tensor(labels).float()
            
            weighted_bce_loss = loss_fct(logits_t, labels_t)
            
            
            #** Count penalty **#
            probs = torch.sigmoid(logits_t)
            predicted_counts = (probs > self.config['model']['threshold']).sum(dim=1).float()
            excess_labels = torch.clamp(predicted_counts - self.config['model']['max_labels'], min=0)
            count_penalty = (excess_labels ** 2).mean()  # Quadratic penalty
            
            weighted_loss = weighted_bce_loss + self.config['model']['count_penalty_weight'] * count_penalty

            
            results["weighted_eval_loss"] = weighted_loss.item()
        
        return results