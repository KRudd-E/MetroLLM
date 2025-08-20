import torch
import torch.nn as nn
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification   
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.auto.configuration_auto import AutoConfig
from sklearn.preprocessing import MultiLabelBinarizer


class WeightedBCEModel(AutoModelForSequenceClassification):
    """Custom model that uses Weighted Binary Cross Entropy to handle class imbalance."""
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.pos_weight = None  # Will be set after initialization

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs) # type: ignore
        logits = outputs.logits

        if labels is not None and self.pos_weight is not None:
            # Apply weighted binary cross entropy loss
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fct(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return outputs


class ClassificationWrapper:
    def __init__(self, run: str, config, pos_weights=None):
        model_name = config["model"]["name"] if run == "train" else config["model"]["source_dir"]

        model_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=config["data"]["class_no"],
            problem_type="multi_label_classification",
        )
        
        # Disable compilation features that cause issues in HPC environments
        if hasattr(model_config, 'compile_embeddings'):
            model_config.compile_embeddings = False
        if hasattr(model_config, 'attention_implementation'):
            model_config.attention_implementation = "eager"

        if run == "train" and pos_weights is not None:
            self.model = WeightedBCEModel.from_pretrained(
                model_name,
                config=model_config,
            )
            # Set the pos_weight after model creation
            self.model.pos_weight = pos_weights
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=model_config,
                **({"low_cpu_mem_usage": True, "device_map": "auto"} if run != "train" else {}),
            )

        # Disable any compiled components in the model to avoid HPC compilation issues
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
            if hasattr(embeddings, 'compiled_embeddings'):
                # Replace compiled embeddings with regular embeddings
                embeddings.compiled_embeddings = embeddings.tok_embeddings
        
        # Disable torch.compile on the entire model
        if hasattr(self.model, '_compiled'):
            self.model._compiled = False

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
