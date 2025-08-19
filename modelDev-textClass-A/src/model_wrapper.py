import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification   
from transformers.data.data_collator import DataCollatorWithPadding

class ClassificationWrapper:
    def __init__(self, run: str, config):
        """
        Args:
            run: "train" or "inference"
            config: configuration dict
            num_labels: number of classes from preprocessing
        """
        if run == "train":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config["model"]["name"],
                num_labels=config['data']['class_no'],
                problem_type="multi_label_classification"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["name"], use_fast=True
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config["model"]["source_dir"],
                num_labels=config['data']['class_no'],
                problem_type="multi_label_classification",
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["source_dir"], use_fast=True
            )

        # Collator – works with variable sequence lengths
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest"
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device

    def get_data_collator(self):
        return self.data_collator