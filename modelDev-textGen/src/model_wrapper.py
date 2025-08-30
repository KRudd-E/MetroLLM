# File: src/models/model_wrapper.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import torch


class DeepSeekWrapper:
    def __init__(self, run, config):
       
        #*** Train ***#
        if run == 'train':
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model']['name'],
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, 
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['model']['name'], 
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # Update model config
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False,  # Causal LM (chat), not masked LM (e.g., direct classification)
                pad_to_multiple_of=8
            )
        
        #*** Eval ***#
        elif run == 'eval': 
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model']['dir'], 
                low_cpu_mem_usage=True, 
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['model']['dir'], 
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False,
                pad_to_multiple_of=8
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Only move to device if not using device_map
        if run == 'train':
            self.model.to(self.device) #type: ignore

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device

    def get_data_collator(self):
        return self.data_collator
