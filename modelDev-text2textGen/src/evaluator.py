import json
import torch
import evaluate
from tqdm import tqdm
from torch.utils.data import DataLoader
import nltk
import numpy as np

class Evaluator:
    def __init__(self, model_wrapper, dataset, config):
        self.config         = config
        self.model          = model_wrapper.get_model()
        self.tokenizer      = model_wrapper.get_tokenizer()
        self.device         = model_wrapper.get_device()
        self.data_collator  = model_wrapper.get_data_collator()
        self.dataset        = dataset
        #self.metric         = evaluate.load(self.config["eval_args"]["metric"])
        
        self.rouge          = evaluate.load("rouge")
        self.bleu           = evaluate.load("bleu")



    def evaluate(self, config):
        
        #* Dataset column names: ['id', 'task', 'input', 'output', 'input_ids', 'attention_mask', 'labels']
        
        # self.dataset = self.dataset.map(
        #     lambda examples: self.tokenizer(
        #         examples["input"],
        #         padding="max_length",
        #         truncation=True,
        #         max_length=config["model"]["max_length"],
        #     ),
        #     batched=True,
        #     remove_columns=self.dataset.column_names,
        # )
        
        # remove cols ['id', 'task', 'input', 'output'] from dataset
        self.dataset = self.dataset.remove_columns(['id', 'task', 'input', 'output'])

        dataloader = DataLoader(
            dataset     = self.dataset,
            batch_size  = config["eval_args"]["batch_size"],
            collate_fn  = self.data_collator,
        )
        
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        self.model.to(self.device)

        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
        
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )


                # predictions = outputs.cpu().numpy()
                # labels = labels.numpy()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        
        eval_pred = (np.array(all_predictions), np.array(all_labels))
        results = self.compute_metrics3(eval_pred)
        
        print(results)

        # write to json
        with open(config["eval_args"]["output_dir"], "w") as f:
            json.dump(results, f, indent=4)


        

    def compute_metrics3(self, eval_pred):
        predictions, labels = eval_pred

        # Remove -100s and decode
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id) #-100 values are typically used as ignore indices in loss computation during training, but they need to be converted to valid token IDs before decoding.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Normalize text (tokenize into sentences for ROUGE)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        decoded_preds_sent = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        decoded_labels_sent = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

        # Metric accumulators
        exact_matches = []
        bleu_preds = []
        bleu_refs = []
        rouge_preds = []
        rouge_refs = []

        for pred, label, pred_sent, label_sent in zip(decoded_preds, decoded_labels, decoded_preds_sent, decoded_labels_sent):
            pred_len = len(pred.split())

            # Exact match
            exact_matches.append(int(pred == label))

            # BLEU if moderately long
            if pred_len >= 10:
                bleu_preds.append(pred)
                bleu_refs.append([label])

            # ROUGE if longer or paragraph-like
            if pred_len > 30 or '\n' in pred or '\n' in label:
                rouge_preds.append(pred_sent)
                rouge_refs.append(label_sent)

        results = {}

        # Aggregate metrics
        if exact_matches:
            results["exact_match"] = round(np.mean(exact_matches) * 100, 2)
        if bleu_preds:
            bleu_result = self.bleu.compute(predictions=bleu_preds, references=bleu_refs)
            if bleu_result is not None:
                results.update(bleu_result)
                results["bleu"] = round(results["bleu"] * 100, 2)
        if rouge_preds:
            rouge_scores = self.rouge.compute(predictions=rouge_preds, references=rouge_refs, use_stemmer=True)
            if rouge_scores is not None:
                for k, v in rouge_scores.items():
                    results[k] = round(v * 100, 2)

        results["gen_len"] = round(np.mean([len(p.split()) for p in decoded_preds]), 2)

        return results


    # def evaluate(self, config):
    #     self.model.eval()

    #     # Only keep tensor-friendly fields
    #     keep = {"input_ids", "labels", "attention_mask"}
    #     dataset_for_loader = self.dataset.remove_columns(
    #         [c for c in self.dataset.column_names if c not in keep]
    #     )

    #     loader = torch.utils.data.DataLoader(
    #         dataset_for_loader,
    #         batch_size=config["eval_args"]["batch_size"],
    #         collate_fn=self.data_collator,
    #     )

    #     for batch in tqdm(loader, desc="Evaluating", unit="batch"):
    #         input_ids      = batch["input_ids"].to(self.device)
    #         attention_mask = batch["attention_mask"].to(self.device)

    #         with torch.no_grad():
    #             generated = self.model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 max_length=self.tokenizer.model_max_length,
    #             )

    #         preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
    #         # Convert label ids back to text (strip -100 and pad tokens)
    #         refs = [
    #             self.tokenizer.decode(
    #                 [t for t in seq if t != -100],
    #                 skip_special_tokens=True,
    #                 clean_up_tokenization_spaces=True,
    #             )
    #             for seq in batch["labels"].cpu().numpy()
    #         ]
    #         self.metric.add_batch(
    #             predictions=[p.strip().lower() for p in preds],
    #             references=[r.strip().lower() for r in refs],
    #         )

    #     self.eval_to_jsonl(
    #         model_dir = self.config['model']['dir'],
    #         output_dir = config["eval_args"]["output_dir"],
    #         results = self.metric.compute(),
    #     )
    #     print("Evaluation metrics:", self.metric.compute())
        
    # @staticmethod
    # def eval_to_jsonl(model_dir, output_dir, results):
    #     """ Save evaluation results to a JSONL file.
    #     """
          
    #     os.makedirs(os.path.join(os.getcwd() + output_dir), exist_ok=True)
    #     with open(os.path.join(os.getcwd() + output_dir), 'r', encoding='utf-8') as f:
    #         try:
    #             z = json.load(f)
    #         except json.JSONDecodeError:
    #             z = {}

    #     y = {
    #         "model_dir": model_dir,
    #         "timestamp": datetime.datetime.now().isoformat(),
    #         "results": results,
    #     }
    #     z.update(y)

    #     with open(os.path.join(os.getcwd() + output_dir), 'w', encoding='utf-8') as f:
    #         json.dump(z, f, indent=4)