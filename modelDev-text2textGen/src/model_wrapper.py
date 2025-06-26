# File: src/models/model_wrapper.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import torch

class FlanT5Wrapper:
    def __init__(self, config):
        self.config = config

        assert 'train_or_eval' in self.config, "Configuration must contain 'train_or_eval' key."
        if self.config['train_or_eval'] == 'train': self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config['train']['model']['name'],)
        else: self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config['eval']['model']['path'],)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['train']['model']['name'], use_fast=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device

    def get_data_collator(self):
        return self.data_collator