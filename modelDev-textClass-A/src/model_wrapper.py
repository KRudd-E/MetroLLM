import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification   
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.auto.configuration_auto import AutoConfig
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import numpy as np

class WeightedBCEModel(AutoModelForSequenceClassification):
    def __init__(self, config, pos_weight=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.pos_weight = pos_weight

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs) # type: ignore
        logits = outputs.logit

        if labels is not None and self.pos_weight is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
            loss = loss_fct(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return outputs


class ClassificationWrapper:
    def __init__(self, run: str, config, label_matrix):
        model_name = config["model"]["name"] if run == "train" else config["model"]["source_dir"]

        model_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=config["data"]["class_no"],
            problem_type="multi_label_classification",
        )
        
        class_weights = compute_pos_weight(label_matrix)

        self.model = WeightedBCEModel.from_pretrained(
            model_name,
            config=model_config,
            **({"low_cpu_mem_usage": True, "device_map": "auto"} if run != "train" else {}),
        )
        
        self.model.pos_weight = class_weights

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding="longest")
        self.mlb = MultiLabelBinarizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_model(self): return self.model
    def get_tokenizer(self): return self.tokenizer
    def get_device(self): return self.device
    def get_data_collator(self): return self.data_collator
    def get_mlb(self): return self.mlb
    
    
def compute_pos_weight(label_matrix: np.ndarray) -> torch.Tensor:
    positives = label_matrix.sum(axis=0)
    negatives = label_matrix.shape[0] - positives
    pos_weight = negatives / (positives + 1e-8)   # avoid div by zero
    return torch.tensor(pos_weight, dtype=torch.float32)
