import numpy as np
import pandas as pd

from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer


class Preprocessor:
    def __init__(self, config, model_wrapper):
        self.config = config
        self.mlb = model_wrapper.get_mlb()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.max_length = self.config["model"]['max_length']

    def run(self):
        df = pd.read_csv(self.config["data"]["source_dir"])

        # Multi-label binarization
        y = self.mlb.fit_transform(df["Task"])
        task_names = self.mlb.classes_
        
        # DEBUG: Inspect the multi-label binarizer results
        print(f"DEBUG: MultiLabelBinarizer classes: {len(task_names)} classes")
        print(f"DEBUG: MultiLabelBinarizer classes list: {task_names}")
        print(f"DEBUG: Binarized labels shape: {y.shape}")
        print(f"DEBUG: Sample binarized labels (first 3 rows): {y[:3]}")
        
        # POTENTIAL FIX: Ensure the number of labels matches config expectation
        expected_num_labels = self.config['data']['class_no']
        actual_num_labels = len(task_names)
        if actual_num_labels != expected_num_labels:
            print(f"WARNING: Number of labels in data ({actual_num_labels}) doesn't match config ({expected_num_labels})")
            print("This is likely the cause of the tensor size mismatch error!")
        
        df["label_vec"] = y.tolist() #type: ignore

        # Text column normalization
        text_col = self.config["data"]["text_col"]
        df = df.rename(columns={text_col: "text"})

        ds = Dataset.from_pandas(df[["text", "label_vec"]])
        ds = ds.train_test_split(test_size=0.15, seed=42)

        ds_tok = ds.map(self.tok_fn, batched=True, remove_columns=["text", "label_vec"])

        return ds_tok

    def tok_fn(self, ex):
        out = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
        )
        out["labels"] = ex["label_vec"]
        return out