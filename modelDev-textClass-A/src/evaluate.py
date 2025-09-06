import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd 

class Evaluator:
    def __init__(self, config, model_wrapper):
        self.config = config
        self.device = model_wrapper.get_device()
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.data_collator = model_wrapper.get_data_collator()
        self.output_dir = config["evaluation_args"]["output_dir"]

    def run(self, dataset):
        dataloader = DataLoader(
            dataset["test"],  
            batch_size=self.config["evaluation_args"]["batch_size"],
            shuffle=False,
            collate_fn=self.data_collator,
        )

        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits.detach().cpu().numpy()
                preds.append(logits)
                labels.append(batch["labels"].cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds_probs = 1 / (1 + np.exp(-preds))
        preds_classes = (preds_probs >= 0.5).astype(int)

        num_classes = preds_classes.shape[1]
        pred_cols = {f"pred_class_{i}": preds_classes[:, i] for i in range(num_classes)}
        label_cols = {f"label_class_{i}": labels[:, i] for i in range(num_classes)}
        df = pd.DataFrame({**pred_cols, **label_cols})

        csv_path = f"{self.output_dir}/results.csv"
        df.to_csv(csv_path, index=False)

        print(f"Saved predictions and labels to {csv_path}")
        return preds_classes, labels