# File: src/training/trainer.py
import numpy as np
import nltk
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate

class Trainer:
    def __init__(self, model_wrapper, dataset, config):
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.device = model_wrapper.get_device()
        self.data_collator = model_wrapper.get_data_collator()
        self.dataset = dataset
        self.config = config

        self.metric = evaluate.load("rouge")
        nltk.download("punkt", quiet=True)

    def train(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=self.config["training"].get("lr", 3e-4),
            per_device_train_batch_size=self.config["training"].get("batch_size", 8),
            per_device_eval_batch_size=self.config["training"].get("eval_batch_size", 4),
            weight_decay=self.config["training"].get("weight_decay", 0.01),
            save_total_limit=self.config["training"].get("save_total_limit", 3),
            num_train_epochs=self.config["training"].get("epochs", 3),
            predict_with_generate=True,
            push_to_hub=False
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["val"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: v.mid.fmeasure for k, v in result.items()}