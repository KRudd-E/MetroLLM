import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

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

        # save for later metrics
        np.save(f"{self.output_dir}/preds.npy", preds)
        np.save(f"{self.output_dir}/labels.npy", labels)

        print(f"Saved predictions and labels to {self.output_dir}")
        return preds, labels