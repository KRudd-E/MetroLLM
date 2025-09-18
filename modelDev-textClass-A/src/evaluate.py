import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.model_wrapper import predict_with_count_limit


class Evaluator:
    def __init__(self, config, model_wrapper, task_names):
        self.config = config

        self.model_wrapper = model_wrapper 
        self.device = model_wrapper.get_device()
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.data_collator = model_wrapper.get_data_collator()

        self.output_dir = config["evaluation_args"]["output_dir"]
        self.task_names = task_names


    def run(self, dataset):

        print("Task names (order):", self.task_names)

        #** Debug: inspect first item in dataset **#
        try:
            first_item = dataset[0]
            if isinstance(first_item, dict) and "labels" in first_item:
                print("[DEBUG] First label vector from dataset:", first_item["labels"])
            else:
                print("[DEBUG] First item from dataset (no 'labels' key):", first_item)
        except Exception as e:
            print(f"[DEBUG] Could not access first item in dataset: {e}")


        dataloader = DataLoader(
            dataset,
            batch_size=self.config["evaluation_args"]["batch_size"],
            shuffle=False,
            collate_fn=self.data_collator,
        )

        self.model.eval()
        preds, labels = [], []

        #** Inference Loop **#
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits.detach().cpu().numpy()
                preds.append(logits)
                labels.append(batch["labels"].cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        print("Task names used for evaluation:", self.task_names)
        print("\nFirst 5 label vectors (from dataloader):\n", labels[:5])
        print("First 5 prediction vectors (from model):\n", preds[:5])

        #** Standard predictions **#
        preds_probs = 1 / (1 + np.exp(-preds))
        preds_classes_standard = (preds_probs >= self.config['model']['standard_threshold']).astype(int)
        
        #** Count-limited predictions **#
        preds_tensor = torch.tensor(preds)
        preds_classes_limited = predict_with_count_limit(preds_tensor, self.config['model']['limited_threshold'], self.config['model']['max_labels'])
        preds_classes_limited = preds_classes_limited.numpy().astype(int)

        print(f"Predicted classes limited example: {preds_classes_limited[:5]}")
        
        
        #** Save to CSV **#
        pred_strs_standard = []
        pred_strs_limited = []
        label_strs = []
        
        for pred_std, pred_lim, label_row in zip(preds_classes_standard, preds_classes_limited, labels):
            pred_labels_std = [self.task_names[i] for i, val in enumerate(pred_std) if val == 1]
            pred_labels_lim = [self.task_names[i] for i, val in enumerate(pred_lim) if val == 1]
            true_labels = [self.task_names[i] for i, val in enumerate(label_row) if val == 1]
            
            pred_strs_standard.append(f"[{','.join(pred_labels_std)}]")
            pred_strs_limited.append(f"[{','.join(pred_labels_lim)}]")
            label_strs.append(f"[{','.join(true_labels)}]")

        df = pd.DataFrame({
            "predictions_standard": pred_strs_standard,
            "predictions_limited": pred_strs_limited,
            "labels": label_strs,
            "pred_count_standard": [len(pred_std.nonzero()[0]) for pred_std in preds_classes_standard],
            "pred_count_limited": [len(pred_lim.nonzero()[0]) for pred_lim in preds_classes_limited],
            "true_count": [len(label.nonzero()[0]) for label in labels]
        })

        csv_path = f"{self.output_dir}/results.csv"
        df.to_csv(csv_path, index=False)

        print(f"Saved predictions and labels to {csv_path}")
        print(f"Average number of labels predicted (standard): {np.mean([len(p.nonzero()[0]) for p in preds_classes_standard]):.2f}")
        print(f"Average number of labels predicted (limited): {np.mean([len(p.nonzero()[0]) for p in preds_classes_limited]):.2f}")
        