"""
For handling all data preprocessing
"""

from datasets import load_dataset

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def prepare_data(self):
        print(f"Loading data from {self.config.data_path}")
        dataset = load_dataset("json", data_files=self.config.data_path)
        print(f"Loaded {len(dataset['train'])} samples.")
        return dataset["train"]