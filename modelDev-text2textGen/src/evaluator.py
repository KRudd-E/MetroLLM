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
        
        print(f"\nConfig:\n{config}\n\n")

        #* Dataset column names: ['id', 'task', 'input', 'output', 'input_ids', 'attention_mask', 'labels']
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

        #** Batch inference loop
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }

                #** Generate **#
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    do_sample=True,
                    top_k=50,
                    top_p=0.98,
                    temperature=0.65,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )

                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        

        results = self.compute_metrics3((all_predictions, all_labels))
    
        #** Save results to json **#
        with open(config["eval_args"]["output_dir"], "w") as f:
            json.dump(results, f, indent=4)
        print(results)


        

    def compute_metrics3(self, eval_pred):
        predictions, labels = eval_pred

        decoded_preds = []
        decoded_labels = []
        
        #** Decode preds and labels **#
        for pred_array in predictions:
            pred_clean = np.where(pred_array != -100, pred_array, self.tokenizer.pad_token_id)
            decoded_pred = self.tokenizer.decode(pred_clean, skip_special_tokens=True)
            decoded_preds.append(decoded_pred)
        
        for label_array in labels:
            label_clean = np.where(label_array != -100, label_array, self.tokenizer.pad_token_id)
            decoded_label = self.tokenizer.decode(label_clean, skip_special_tokens=True)
            decoded_labels.append(decoded_label)


        #** Normalize whitespace and tokenize into sentences **#
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        decoded_preds_sent = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        decoded_labels_sent = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

        #** Metric accumulators **#
        exact_matches = []
        bleu_preds = []
        bleu_refs = []
        rouge_preds = []
        rouge_refs = []

        for pred, label, pred_sent, label_sent in zip(decoded_preds, decoded_labels, decoded_preds_sent, decoded_labels_sent):
            pred_len = len(pred.split())

            #** Exact match **
            exact_matches.append(int(pred == label))

            #** BLEU if 10+ tokens **
            if pred_len >= 10:
                bleu_preds.append(pred)
                bleu_refs.append([label])

            #** ROUGE if 30+ tokens or multi-sentence **
            if pred_len > 30 or '\n' in pred or '\n' in label:
                rouge_preds.append(pred_sent)
                rouge_refs.append(label_sent)

        results = {}

        #** Compute and aggregate metrics **#
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