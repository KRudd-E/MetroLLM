import json
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import random
import re
import os
import pandas as pd


class MMLU_Evaluator:
    def __init__(self, model_wrapper, config, mmlu_subset="test"):
        self.config = config
        
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config['batch_size']
        
        self.full_dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        self.dataset = self.full_dataset[mmlu_subset]
        self.eval_split = mmlu_subset

        self.categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 
                   'biology', 'health', 'physics', 'business', 'philosophy', 
                   'economics', 'other', 'psychology', 'history']

        #** Per-category prompts **#
        self.prompts = {c: '' for c in self.categories}
        
        val_split = self.full_dataset["validation"]
        for d in val_split:
            category = d['category']  # type: ignore
            if category in self.prompts:
                self.prompts[category] += (
                    'Q: ' + d['question'] + '\n' +              # type: ignore
                    self.form_options(d['options']) + '\n' +    # type: ignore
                    d['cot_content'] + '\n\n'                   # type: ignore
                )

        #** Clip dataset **#
        clip_percentage = config.get('data_reduction', 1.0)  # Default to 100% if not provided
        if clip_percentage < 1.0:
            clipped_data = []
            for category in self.categories:
                category_data = [entry for entry in self.dataset if entry['category'] == category] # type: ignore
                clip_size = max(1, int(len(category_data) * clip_percentage))
                clipped_data.extend(random.sample(category_data, clip_size))
            self.dataset = clipped_data
                
        # src:      https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
        # example:  https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/run_gpt4o.py

class Test_Set_Evaluator:
    """ Evaluator for the post-training test set. """
    def __init__(self, model_wrapper, config, dataset):
        self.config = config
        
        self.dataset = dataset
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        
        self.batch_size = config['batch_size']
        
    
    def evaluate(self):
        
        if self.config['run'] == False:
            print("Test Set evaluation skipped.\n")
            return
        
        print("Dataset:", self.dataset)
        
        all_predictions = []
        self.model.eval()
        self.model.to(self.device)
        
        for i in tqdm(range(0, len(self.dataset), self.batch_size)):
            batch = self.dataset[i:i+self.batch_size]

            #** Prepare batch **#
            inputs = self.tokenizer(
                batch['input'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)

            #** Generate outputs **#
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                #** Decode and extract predictions **#
                generated_texts = []
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    generated_part = output[input_length:]
                    generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    generated_texts.append(generated_text.strip())
                    
                all_predictions.extend(generated_texts)
        
        #** Compute metrics **#
        
        # perplexity
        
        # accuracy
        
        # F1-score
        





class Task_Evaluator:
    def __init__(self, model_wrapper, config):
        self.config = config
        
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.batch_size = config['batch_size']
        
        self.dataset = pd.read_csv(self.config['data_dir'])


    def evaluate(self):
        print("Evaluating on specific tasks.\n")
        
        print("Dataset:", self.dataset.head())
        print("Dataset size:", len(self.dataset))
        print("Data columns:", self.dataset.columns)
        
        for i in tqdm(range(0, len(self.dataset), self.batch_size)):
            batch = self.dataset[i:i+self.batch_size]
            
            input_prompt = self.config['task_prompt'].format(
                task_list=self.config['task_list'],
                txt=batch['Text']
            )
            
            inputs = self.tokenizer(
                input_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_length"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated_texts = []
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    generated_part = output[input_length:]
                    generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                    generated_texts.append(generated_text.strip())
                    
                print("Generated texts:", generated_texts)
            
