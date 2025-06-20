"""
For handling all data preprocessing
"""

from datasets import load_dataset

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def run(self):
        
        from datasets import load_dataset
        from transformers import AutoTokenizer

        dataset = load_dataset("yelp_review_full")
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

        def tokenize(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = dataset.map(tokenize, batched=True)
        
        
        print(f"Loading data from {self.config.data_path}")
        dataset = load_dataset("json", data_files=self.config.data_path)
        print(f"Loaded {len(dataset['train'])} samples.")
        
        
        return dataset["train"]