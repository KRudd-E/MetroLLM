""" model_wrapper.py
Wrapper class for text classification models using Hugging Face Transformers.
Includes a custom class for weighted binary cross-entropy loss to handle class imbalance.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification   
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.preprocessing import MultiLabelBinarizer



class WeightedBCEModel(AutoModelForSequenceClassification):
    def __init__(self, config, pos_weight=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.pos_weight = pos_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs) # type: ignore
        logits = outputs.logits

        loss = None
        if labels is not None and self.pos_weight is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fct(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def save_pretrained(self, save_directory, **kwargs):
        # Force safe serialization to handle shared tensors
        kwargs['safe_serialization'] = True
        return super().save_pretrained(save_directory, **kwargs) # type: ignore


class ClassificationWrapper:
    def __init__(self, run: str, config, pos_weights=None):
        model_name = config["model"]["name"] if run == "train" else config["model"]["source_dir"]

        model_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=config["data"]["class_no"],
            problem_type="multi_label_classification",
            hidden_dropout_prob=float(config['training_args']['dropout']),
            attention_probs_dropout_prob=float(config['training_args']['dropout']),
        )
        
        # Disable features which cause issues on HPC 
        if hasattr(model_config, 'compile_embeddings'):
            model_config.compile_embeddings = False
        if hasattr(model_config, 'attention_implementation'):
            model_config.attention_implementation = "eager"


        self.model = WeightedBCEModel.from_pretrained(
            model_name,
            config=model_config,
            pos_weight=pos_weights,
            **({"low_cpu_mem_usage": True, "device_map": "auto"} if run != "train" else {}),
        )
        if pos_weights is not None:
            print(f"\nSet pos_weight for training: {pos_weights}\n")

        # Disable compilation.
        self._disable_compilation()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")
        self.mlb = MultiLabelBinarizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)



    def _disable_compilation(self):
        """Disable compilation features that cause issues in HPC environments."""
        # Disable compiled embeddings in ModernBERT
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embeddings'):
            embeddings = self.model.model.embeddings
            # Safely remove compiled embeddings if they exist
            if 'compiled_embeddings' in embeddings.__dict__:
                try:
                    delattr(embeddings, 'compiled_embeddings')
                    # Set a flag to indicate we're using non-compiled embeddings
                    embeddings._use_compiled = False
                except AttributeError:
                    # In case delattr still fails
                    pass

        
        # Set dynamic tied weights keys to handle shared tensors properly
        if hasattr(self.model, '_dynamic_tied_weights_keys'):
            # Remove problematic shared tensor keys
            tied_keys = getattr(self.model, '_dynamic_tied_weights_keys', set())
            tied_keys.discard('model.embeddings.tok_embeddings.weight')
            tied_keys.discard('model.embeddings.compiled_embeddings.weight')
            self.model._dynamic_tied_weights_keys = tied_keys
        
        # Disable torch.compile on the entire model
        if hasattr(self.model, '_compiled'):
            self.model._compiled = False
            
        # Set safe serialization to handle any remaining tensor sharing issues
        if hasattr(self.model, 'config'):
            self.model.config.safe_serialization = True

    def get_model(self): return self.model
    def get_tokenizer(self): return self.tokenizer
    def get_device(self): return self.device
    def get_data_collator(self): return self.data_collator
    def get_mlb(self): return self.mlb


def compute_pos_weights(y_labels) -> torch.Tensor:
    """
    Compute positive weights for weighted BCE loss.
    For multi-label classification, pos_weight should be neg_count / pos_count
    """

    if hasattr(y_labels, 'toarray'):
        y_labels = y_labels.toarray()
    
    y_labels = np.array(y_labels)
    num_samples = y_labels.shape[0]
    pos_counts = y_labels.sum(axis=0)  # Count of positive samples per class
    neg_counts = num_samples - pos_counts  # Count of negative samples per class
    
    # Avoid division by zero
    pos_weights = neg_counts / (pos_counts + 1e-8)
    
    return torch.tensor(pos_weights, dtype=torch.float32)
