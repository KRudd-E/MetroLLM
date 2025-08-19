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
        y = self.mlb.fit_transform(df["tasks"])
        #task_names = self.mlb.classes_
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
            padding="max_length",
            max_length=self.max_length,
        )
        out["labels"] = ex["label_vec"]
        return out