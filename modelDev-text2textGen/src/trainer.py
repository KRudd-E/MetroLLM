# File: src/training/trainer.py
import numpy as np
import nltk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from src.utils.logging import LoggingCallback, DebugCallback
import torch

class Trainer:
    def __init__(self, model_wrapper, dataset, config):
        self.config = config
        self.dataset = dataset
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.data_collator = model_wrapper.get_data_collator()

        self.metric = evaluate.load("rouge")
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)
        
        self.check_model_config()  # Verify model configuration

    def train(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config["train"]['training_args'].get("output_dir", "modelDev-text2textGen/results"),
            eval_strategy="epoch",
            save_strategy="epoch",  # Save checkpoint at the end of every epoch
            learning_rate=float(self.config["train"]['training_args'].get("lr", 3e-4)),
            per_device_train_batch_size=self.config["train"]['training_args'].get("batch_size", 4),  # Reduced from 8 to 2
            per_device_eval_batch_size=self.config["train"]['training_args'].get("eval_batch_size", 4),  # Reduced from 4 to 1
            weight_decay=float(self.config["train"]['training_args'].get("weight_decay", 0.01)),
            save_total_limit=self.config["train"]['training_args'].get("save_total_limit", 3),
            num_train_epochs=self.config["train"]['training_args'].get("epochs", 20),
            predict_with_generate=True,
            push_to_hub=False,
            load_best_model_at_end=True,
            metric_for_best_model=self.config["train"]['training_args'].get("metric_for_best_model", "rouge1"),
            greater_is_better=True,
            # Memory optimization settings
            # gradient_accumulation_steps=self.config["train"]['training_args'].get("gradient_accumulation_steps", 4),  # Accumulate gradients to simulate larger batch
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
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[LoggingCallback(log_path=self.config["train"]['training_args'].get("log_path", "modelDev-text2textGen/results/logs.json"))]
        )

        trainer.train()

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            print("Predictions are a tuple, extracting the first element.")
            preds = preds[0]

        preds = np.asarray(preds)
        preds = np.where(preds >= 0, preds, self.tokenizer.pad_token_id) # Remove negative token ids from preds
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        
        labels = np.asarray(labels)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)         # Replace -100 in labels with pad_token_id
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return result


# Debugging and Inspection Functions

    def check_model_config(self):
        """Verify the model configuration is appropriate for the task"""
        config = self.model.config
        # Ensure pad tokens match
        if config.pad_token_id != self.tokenizer.pad_token_id:
            print("WARNING: Model and tokenizer pad tokens don't match!")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Check other critical settings
        print("Vocab size:", config.vocab_size)
        print("Tokenizer vocab size:", len(self.tokenizer))