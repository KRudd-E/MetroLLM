import pandas as pd
import torch
from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.mlb = MultiLabelBinarizer()
        self.tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["name"],
                use_fast=True
            )
        self.max_length = self.config["model"]['max_length']

    def run(self):
        df = pd.read_csv(self.config["data"]["source_dir"])
        df['Task'] = df['Task'].apply(lambda x: [x] if isinstance(x, str) else x)
        
        # Multi-label binarization
        y = self.mlb.fit_transform(df["Task"])
        class_weights = self.compute_class_weights(y)
        task_names = self.mlb.classes_

        df["label_vec"] = y.tolist() #type: ignore
        df = df.rename(columns={self.config["data"]["text_col"]: "text"})

        ds = Dataset.from_pandas(df[["text", "label_vec"]])
        ds = ds.train_test_split(test_size=0.15, seed=42)
        ds_tok = ds.map(self.tok_fn, batched=True, remove_columns=["text", "label_vec"])

        return ds_tok, task_names, y

    def tok_fn(self, ex):
        out = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
        )
        # Convert labels to float for binary cross entropy loss
        out["labels"] = [list(map(float, label_vec)) for label_vec in ex["label_vec"]]
        return out
    
    @staticmethod
    def compute_class_weights(labels) -> torch.Tensor:
        num_samples, num_classes = labels.shape
        pos_counts = labels.sum(axis=0)         # number of positives per class
        neg_counts = num_samples - pos_counts   # number of negatives per class

        pos_weight = neg_counts / (pos_counts + 1e-5) # Avoid division by zero
        return torch.tensor(pos_weight, dtype=torch.float)