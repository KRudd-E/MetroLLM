import numpy as np
import nltk
import evaluate
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from src.utils.callbacks import LoggingCallback, DebugCallback
from collections import defaultdict

class Trainer:
    def __init__(self, model_wrapper, dataset, config):
       
        self.dataset          = dataset
        self.model            = model_wrapper.get_model()
        self.tokenizer        = model_wrapper.get_tokenizer()
        self.device           = model_wrapper.get_device()
        self.data_collator    = model_wrapper.get_data_collator()
        
        self.rouge        = evaluate.load("rouge")
        self.bleu         = evaluate.load("bleu")
        self.exact_match  = evaluate.load("exact_match")
        
        nltk.download("punkt", quiet=True)
        
        self.check_model_config()


    def train(self, config):
        
        training_args = Seq2SeqTrainingArguments(
            output_dir                    =   str(config['output_dir']),
            eval_strategy                 =   str(config['training_args']['eval_strategy']),
            save_strategy                 =   str(config['training_args']['save_strategy']),
            #eval_steps                    =   int(config['training_args']['eval_steps']),
            #save_steps                    =   int(config['training_args']['save_steps']),
            logging_strategy              =   str(config['training_args']['logging_strategy']),
            logging_steps                 =   int(config['training_args']['logging_steps']),
            learning_rate                 = float(config['training_args']['learning_rate']),
            per_device_train_batch_size   =   int(config['training_args']['per_device_train_batch_size']),
            per_device_eval_batch_size    =   int(config['training_args']['per_device_eval_batch_size']),
            gradient_accumulation_steps   =   int(config['training_args']['gradient_accumulation_steps']),
            weight_decay                  = float(config['training_args']['weight_decay']),
            warmup_ratio                  = float(config['training_args']['warmup_ratio']),
            num_train_epochs              =   int(config['training_args']['num_train_epochs']),
            predict_with_generate         =  bool(config['training_args']['predict_with_generate']),
            generation_max_length         =   int(config['training_args']['generation_max_length']),
            load_best_model_at_end        =  bool(config['training_args']['load_best_model_at_end']),
            metric_for_best_model         =   str(config['training_args']['metric_for_best_model']),
            greater_is_better             =  bool(config['training_args']['greater_is_better']),
            bf16                          =  bool(config['training_args']['bf16']),
            gradient_checkpointing        =  bool(config['training_args']['gradient_checkpointing']),
            dataloader_pin_memory         =  bool(config['training_args']['dataloader_pin_memory']),
            label_smoothing_factor        = float(config['training_args']['label_smoothing_factor']),
            save_total_limit              =   int(config['training_args']['save_total_limit']),
            push_to_hub                   =  bool(config['training_args']['push_to_hub']),
        )
        
        logger = LoggingCallback(config['log_dir'], log_training_steps=config['training_args']['log_training_steps'])
        debugger = DebugCallback()

        trainer = Seq2SeqTrainer(
            model             = self.model,
            args              = training_args,
            train_dataset     = self.dataset["train"],
            eval_dataset      = self.dataset["val"],
            #tokenizer        = self.tokenizer,         #?
            data_collator     = self.data_collator,
            compute_metrics   = self.compute_metrics3, 
            callbacks         = [logger, debugger],
        )

        trainer.train()


    def compute_metrics3(self, eval_pred):
        predictions, labels = eval_pred

        #** Decode preds and labels **#
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        #** Normalize **#
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

            #** BLEU 10+ tokens **
            if pred_len >= 10:
                bleu_preds.append(pred)
                bleu_refs.append([label])

            #** ROUGE 30+ tokens or multi-sentence **
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

        #** Average generated length **#
        results["gen_len"] = round(np.mean([len(p.split()) for p in decoded_preds]), 2)

        return results




    def check_model_config(self):
        """Verify the model configuration is appropriate for the task"""
        config = self.model.config
        # Ensure pad tokens match
        if config.pad_token_id != self.tokenizer.pad_token_id:
            print("WARNING: Model and tokenizer pad tokens don't match!")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print("Vocab size:", config.vocab_size)
        print("Tokenizer vocab size:", len(self.tokenizer))
        
        