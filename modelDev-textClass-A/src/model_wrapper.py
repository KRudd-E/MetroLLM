"""model_wrapper.py

Defines WeightedBCEModelWrapper: wraps a HuggingFace AutoModelForSequenceClassification
model and applies BCEWithLogitsLoss with per-class pos_weight and a penalty for exceeding
a maximum number of positive labels per sample.
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.preprocessing import MultiLabelBinarizer


class WeightedBCEModelWrapper(nn.Module):
    def __init__(self, config, pos_weight: torch.Tensor = None, device=None, checkpoint_dir=None): # type: ignore
        super().__init__()

        self.max_labels = config['model']['max_labels']
        self.count_penalty_weight = config['model']['count_penalty_weight']
        self.threshold = config['model'].get('threshold', 0.5)
        self.pos_weight = pos_weight

        #** Model config **#
        model_config = AutoConfig.from_pretrained(
            config["model"]["name"],
            num_labels=config["data"]["class_no"],
            problem_type="multi_label_classification",
            hidden_dropout_prob=float(config['training_args']['dropout']),
            attention_probs_dropout_prob=float(config['training_args']['dropout']),
        )

        #** Disable compilation features which cause issues on HPC **#
        if hasattr(model_config, 'compile_embeddings'):
            model_config.compile_embeddings = False
        if hasattr(model_config, 'attention_implementation'):
            model_config.attention_implementation = "eager"

        #** Load model weights if eval **#
        if checkpoint_dir is not None:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir,
                config=model_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        
        #** Load pre-trained model if train **#
        else:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                config["model"]["name"],
                config=model_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)

        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")
        self.mlb = MultiLabelBinarizer()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        #** Drop labels to avoid HF loss **#
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            labels_t = labels.float().to(logits.device)

            if logits.shape != labels_t.shape:
                raise RuntimeError(f"Shape mismatch: logits {logits.shape} vs labels {labels_t.shape}.")

            #** Weighted BCE loss **#
            loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device)) # type: ignore
            bce_loss = loss_fct(logits, labels_t)

            #** Penalty for train **#
            if self.count_penalty_weight > 0.0:
                probs = torch.sigmoid(logits)
                predicted_counts = (probs > self.threshold).sum(dim=1).float()
                
                excess_labels = torch.clamp(predicted_counts - self.max_labels, min=0)
                count_penalty = (excess_labels ** 2).mean()  # Quadratic penalty
            else: 
                count_penalty = torch.tensor(0.0, device=logits.device)
            
            #** Total loss **#
            loss = bce_loss + self.count_penalty_weight * count_penalty

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing on the base model."""
        return self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model."""
        return self.base_model.gradient_checkpointing_disable()

    def get_model(self):
        return self

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device

    def get_data_collator(self):
        return self.data_collator
    

#********** External functions **********#

def predict_with_count_limit(logits, threshold, max_labels):
    """
    Apply count limit to predictions.
    Returns predictions with at most max_labels positive predictions per sample.
    """
    probabilities = torch.sigmoid(logits)
    batch_size, num_labels = probabilities.shape
    
    predictions = torch.zeros_like(probabilities)
    
    for i in range(batch_size):
        sample_probs = probabilities[i]
        
        #** Identify labels above threshold **#
        above_threshold = sample_probs > threshold
        above_threshold_indices = torch.where(above_threshold)[0]
        
        #** Apply count limit if needed **#
        if len(above_threshold_indices) <= max_labels:
            predictions[i, above_threshold_indices] = 1
        
        else:
            top_values, top_indices = torch.topk(sample_probs, max_labels)
            predictions[i, top_indices] = 1
            
        #** Ensure at least one label is predicted **#
        if predictions[i].sum() == 0:
            top_value, top_index = torch.topk(sample_probs, 1)
            predictions[i, top_index] = 1
    
    return predictions


def compute_pos_weights(y_labels) -> torch.Tensor:
    
    #** Prepare **#
    if hasattr(y_labels, 'toarray'):
        y_labels = y_labels.toarray()
    
    y_labels = np.array(y_labels)
    num_samples = y_labels.shape[0]
    pos_counts = y_labels.sum(axis=0)
    
    #** Compute weights **#
    neg_counts = num_samples - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-8)
    
    return torch.tensor(pos_weights, dtype=torch.float32)