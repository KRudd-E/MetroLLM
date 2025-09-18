#preprocess.py

import pandas as pd
from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval
import os


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.mlb = MultiLabelBinarizer()
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
        self.max_length = self.config["model"]['max_length']

    def run(self, run_mode):
        import pickle # unhappy outside of run function!?
        
        df = pd.read_csv(self.config["data"]["source_dir"])
        df.Task = df.Task.apply(literal_eval) # str -> list

        
        if run_mode == 'train':
            #** Fit Binarizer **#
            y = self.mlb.fit_transform(df["Task"])
            
            #** Save mlb **#
            os.makedirs(self.config['output_dir'], exist_ok=True)
            mlb_path = os.path.join(self.config['output_dir'] + '/' + 'mlb.pkl')
            with open(mlb_path, 'wb') as f:
                pickle.dump(self.mlb, f)
        
        
        else:
            #** Load Binarizer **#
            mlb_path = os.path.join(self.config['model']['checkpoint_dir'] + '/' + 'mlb.pkl')
            if os.path.exists(mlb_path):
                with open(mlb_path, 'rb') as f:
                    self.mlb = pickle.load(f)
                print(f"Loaded MultiLabelBinarizer from {mlb_path}")
            else:
                raise FileNotFoundError(f"mlb.pkl not found at {mlb_path}. You must provide the label binarizer from training.")
            
            #** Transform labels **#
            y = self.mlb.transform(df["Task"])


        #** Get task names **#
        task_names = self.mlb.classes_
        print(f'task_names:\n{task_names}')
        
        
        #** Create Dataset **#
        df["label_vec"] = y.tolist() # type: ignore
        ds = Dataset.from_pandas(df[["Text", "label_vec"]])
        
        if run_mode == 'train':
            ds = ds.train_test_split(test_size=0.15, seed=42)
            
        ds_tok = ds.map(self.tok_fn, batched=True, remove_columns=["Text", "label_vec"])

        return ds_tok, task_names, y


    def tok_fn(self, input):
        """Tokenization function"""
        out = self.tokenizer(
            input["Text"],
            truncation=True,
            max_length=self.max_length,
        )
        out["labels"] = [list(map(float, label_vec)) for label_vec in input["label_vec"]]
        return out

