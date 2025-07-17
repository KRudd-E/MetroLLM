import numpy as np
import nltk
import evaluate
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from src.utils.logging import LoggingCallback, DebugCallback
from collections import defaultdict

class Trainer:
    def __init__(self, model_wrapper, dataset, config):
        self.dataset = dataset
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.data_collator = model_wrapper.get_data_collator()
        
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.exact_match = evaluate.load("exact_match")
        nltk.download("punkt", quiet=True)
        
        self.check_model_config()  # Verify model configuration

    def train(self, config):
        training_args = Seq2SeqTrainingArguments(
            output_dir=config['training_args'].get("output_dir", "modelDev-text2textGen/results"),
            eval_strategy="epoch",
            save_strategy="epoch",  # Save checkpoint at the end of every epoch
            learning_rate=float(config['training_args'].get("lr", 3e-4)),
            per_device_train_batch_size=config['training_args'].get("batch_size", 4),  # Reduced from 8 to 2
            per_device_eval_batch_size=config['training_args'].get("eval_batch_size", 4),  # Reduced from 4 to 1
            weight_decay=float(config['training_args'].get("weight_decay", 0.01)),
            save_total_limit=config['training_args'].get("save_total_limit", 3),
            num_train_epochs=config['training_args'].get("epochs", 6),
            #predict_with_generate=True,
            push_to_hub=False,
            load_best_model_at_end=True,
            metric_for_best_model=config['training_args'].get("metric_for_best_model", "rouge1"),
            greater_is_better=True,
            # Memory optimization settings
            gradient_accumulation_steps=config['training_args'].get("gradient_accumulation_steps", 4),  # Accumulate gradients to simulate larger batch
            #fp16=True,  # Use mixed precision training to reduce memory usage
            bf16=True,  # Use bfloat16 for better performance on TPUs
            dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
            gradient_checkpointing=True,  # Trade compute for memory
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["val"],
            #tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics3, 
            callbacks=[LoggingCallback(config), DebugCallback()]
        )

        trainer.train()


    def compute_metrics3(self, eval_pred):
        predictions, labels = eval_pred

        # Remove -100s and decode
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
                bleu_preds.append(pred.split())
                bleu_refs.append([label.split()])

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


# Debugging and Inspection Functions

    def check_model_config(self):
        """Verify the model configuration is appropriate for the task"""
        config = self.model.config
        # Ensure pad tokens match
        if config.pad_token_id != self.tokenizer.pad_token_id:
            print("WARNING: Model and tokenizer pad tokens don't match!")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print("Vocab size:", config.vocab_size)
        print("Tokenizer vocab size:", len(self.tokenizer))
        
        
        
        

    # def compute_metrics1(self, eval_preds):
    #     preds, labels = eval_preds

    #     if isinstance(preds, tuple):
    #         print("Predictions are a tuple, extracting the first element.")
    #         preds = preds[0]

    #     preds = np.asarray(preds)
    #     preds = np.where(preds >= 0, preds, self.tokenizer.pad_token_id) # Remove negative token ids from preds
    #     decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        
    #     labels = np.asarray(labels)
    #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)         # Replace -100 in labels with pad_token_id
    #     decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    #     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    #     result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #     return result
    
    
    # def compute_metrics2(self, eval_pred):
    #     ### https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887
    #     predictions, labels = eval_pred
    #     decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    #     decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
    #     # Rouge expects a newline after each sentence
    #     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    #     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
    #     # Compute ROUGE scores
    #     result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    #     # Extract ROUGE f1 scores
    #     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
    #     # Add mean generated length to metrics
    #     prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
    #     result["gen_len"] = np.mean(prediction_lens)
        
    #     return {k: round(v, 4) for k, v in result.items()}
