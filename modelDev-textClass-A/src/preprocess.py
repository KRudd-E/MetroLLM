"""preprocess.py
This script contains the Preprocessor class which handles data loading, 
multi-label binarization, and tokenization for text classification tasks.
"""

import pandas as pd
from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.mlb = MultiLabelBinarizer()
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"],use_fast=True)
        self.max_length = self.config["model"]['max_length']

    def run(self):
        df = pd.read_csv(self.config["data"]["source_dir"])
        df.Task = df.Task.apply(literal_eval) # str -> list
        
        # Multi-label binarization
        y = self.mlb.fit_transform(df["Task"])
        task_names = self.mlb.classes_

        df["label_vec"] = y.tolist() #type: ignore

        ds = Dataset.from_pandas(df[["Text", "label_vec"]])
        ds = ds.train_test_split(test_size=0.15, seed=42)
        ds_tok = ds.map(self.tok_fn, batched=True, remove_columns=["Text", "label_vec"])

        return ds_tok, task_names, y


    def tok_fn(self, input):
        # Tokenize text
        out = self.tokenizer(
            input["text"],
            truncation=True,
            max_length=self.max_length,
        )
        # Convert labels to float for binary cross entropy loss
        out["labels"] = [list(map(float, label_vec)) for label_vec in input["label_vec"]]
        return out