# File: src/models/model_wrapper.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
import torch

class FlanT5Wrapper:
    def __init__(self, run, config):
        if run == 'train':
            print('...')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['name'],)
            self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], use_fast=True)
        else: 
            print('_!!!_')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config['model']['dir'], low_cpu_mem_usage=True, device_map="auto")
            print('__!!!!__')
            self.tokenizer = AutoTokenizer.from_pretrained(config['model']['dir'], use_fast=True)
        
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