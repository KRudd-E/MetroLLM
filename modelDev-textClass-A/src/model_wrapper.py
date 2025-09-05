"""
model_wrapper.py

Defines WeightedBCEModelWrapper: wraps a HuggingFace AutoModelForSequenceClassification
model and applies BCEWithLogitsLoss with per-class pos_weight.
"""

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.preprocessing import MultiLabelBinarizer


class WeightedBCEModelWrapper(nn.Module):
    def __init__(self, model_name: str, config, pos_weight: torch.Tensor = None, device=None): # type: ignore
        super().__init__()
        # Load the base HF model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Register pos_weight as buffer so it is saved + moves with .to(device)
        if pos_weight is not None:
            pw = pos_weight.detach().clone().float().view(-1)
            self.register_buffer("pos_weight", pw)
        else:
            self.pos_weight = None

        # Tokenizer + data collator
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")
        self.mlb = MultiLabelBinarizer()
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(self.device)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Drop labels to avoid HF built-in loss
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
                raise RuntimeError(
                    f"Shape mismatch: logits {logits.shape} vs labels {labels_t.shape}."
                )

            if getattr(self, "pos_weight", None) is not None:
                loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device)) # type: ignore
            else:
                loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(logits, labels_t)

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

    def get_mlb(self):
        return self.mlb



def compute_pos_weights(y_labels) -> torch.Tensor:
    import numpy as np

    if hasattr(y_labels, 'toarray'):
        y_labels = y_labels.toarray()
    y_labels = np.array(y_labels)
    num_samples = y_labels.shape[0]
    pos_counts = y_labels.sum(axis=0)
    neg_counts = num_samples - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-8)
    return torch.tensor(pos_weights, dtype=torch.float32)