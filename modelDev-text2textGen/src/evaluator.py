import os
import json
import torch
import evaluate
from tqdm import tqdm
from datetime import datetime

class Evaluator:
    def __init__(self, model_wrapper, dataset, config):
        self.config = config
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.data_collator = model_wrapper.get_data_collator()
        self.dataset = dataset
        self.metric = evaluate.load(self.config["eval_args"]["metric"])

    def evaluate(self, config):
        self.model.eval()

        # Only keep tensor-friendly fields
        keep = {"input_ids", "labels", "attention_mask"}
        dataset_for_loader = self.dataset.remove_columns(
            [c for c in self.dataset.column_names if c not in keep]
        )

        loader = torch.utils.data.DataLoader(
            dataset_for_loader,
            batch_size=config["eval_args"]["batch_size"],
            collate_fn=self.data_collator,
        )

        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.tokenizer.model_max_length,
                )

            preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            # Convert label ids back to text (strip -100 and pad tokens)
            refs = [
                self.tokenizer.decode(
                    [t for t in seq if t != -100],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for seq in batch["labels"].cpu().numpy()
            ]
            self.metric.add_batch(
                predictions=[p.strip().lower() for p in preds],
                references=[r.strip().lower() for r in refs],
            )

        self.eval_to_jsonl(
            model_dir = self.config['model']['dir'],
            output_dir = config["eval_args"]["output_dir"],
            results = self.metric.compute(),
        )
        print("Evaluation metrics:", self.metric.compute())
        
    @staticmethod
    def eval_to_jsonl(model_dir, output_dir, results):
        """ Save evaluation results to a JSONL file.
        """
          
        os.makedirs(os.path.join(os.getcwd() + output_dir), exist_ok=True)
        with open(os.path.join(os.getcwd() + output_dir), 'r', encoding='utf-8') as f:
            try:
                z = json.load(f)
            except json.JSONDecodeError:
                z = {}

        y = {
            "model_dir": model_dir,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results,
        }
        z.update(y)

        with open(os.path.join(os.getcwd() + output_dir), 'w', encoding='utf-8') as f:
            json.dump(z, f, indent=4)