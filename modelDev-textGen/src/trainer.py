import numpy as np
import evaluate
import torch
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from src.utils.callbacks import LoggingCallback, DebugCallback, \
    MemoryCleanupCallback

class Trainer:
    def __init__(self, model_wrapper, dataset, config):
       
        self.dataset          = dataset
        self.model            = model_wrapper.get_model()
        self.tokenizer        = model_wrapper.get_tokenizer()
        self.device           = model_wrapper.get_device()
        self.data_collator    = model_wrapper.get_data_collator()
        
        self.f1_metric     = evaluate.load("f1")
        self.em_metric     = evaluate.load("exact_match")
        #self.rouge        = evaluate.load("rouge")
        #self.bleu         = evaluate.load("bleu")

        #nltk.download("punkt", quiet=True)
        
        # Clear cache after loading metrics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.check_model_config()


    def train(self, config):

        #** Training Arguments **#
        training_args = TrainingArguments(
            label_names                     = ["labels"],

            output_dir                      = str(config['output_dir']),
            do_train                        = bool(config['training_args']['do_train']),
            do_eval                         = bool(config['training_args']['do_eval']),
            eval_strategy                   = str(config['training_args']['evaluation_strategy']),
            eval_steps                      = int(config['training_args']['eval_steps']),
            save_strategy                   = str(config['training_args']['save_strategy']),
            save_steps                      = int(config['training_args']['save_steps']),
            save_total_limit                = int(config['training_args']['save_total_limit']),
            logging_strategy                = str(config['training_args']['logging_strategy']),
            logging_steps                   = int(config['training_args']['logging_steps']),
            logging_nan_inf_filter          = bool(config['training_args']['logging_nan_inf_filter']),
            
            learning_rate                   = float(config['training_args']['learning_rate']),
            per_device_train_batch_size     = int(config['training_args']['per_device_train_batch_size']),
            per_device_eval_batch_size      = int(config['training_args']['per_device_eval_batch_size']),
            gradient_accumulation_steps     = int(config['training_args']['gradient_accumulation_steps']),
            eval_accumulation_steps         = int(config['training_args']['eval_accumulation_steps']),
            num_train_epochs                = int(config['training_args']['num_train_epochs']),
            max_steps                       = int(config['training_args']['max_steps']),
            warmup_ratio                    = float(config['training_args']['warmup_ratio']),
            weight_decay                    = float(config['training_args']['weight_decay']),
            max_grad_norm                   = float(config['training_args']['max_grad_norm']),
            lr_scheduler_type               = str(config['training_args']['lr_scheduler_type']),
            
            bf16                            = bool(config['training_args']['bf16']),
            gradient_checkpointing          = bool(config['training_args']['gradient_checkpointing']),
            dataloader_pin_memory           = bool(config['training_args']['dataloader_pin_memory']),
            group_by_length                 = bool(config['training_args']['group_by_length']),

            load_best_model_at_end          = bool(config['training_args']['load_best_model_at_end']),
            metric_for_best_model           = str(config['training_args']['metric_for_best_model']),
            greater_is_better               = bool(config['training_args']['greater_is_better']),
            
            report_to                       = list(config['training_args']['report_to']), 
            remove_unused_columns           = bool(config['training_args']['remove_unused_columns']),
            ddp_find_unused_parameters      = bool(config['training_args']['ddp_find_unused_parameters']),
            run_name                        = str(config['training_args']['run_name']),

        )

        #** Callbacks **#
        logger = LoggingCallback(log_path=config['log_dir'], log_training_steps=config['training_args']['log_training_steps'])
        debugger = DebugCallback()
        memory_cleanup = MemoryCleanupCallback()
        early_stopping = EarlyStoppingCallback(early_stopping_patience=config['training_args']['early_stopping_patience'])


        print(f'\nTrain length: {len(self.dataset["train"])}\n'
              f'Validation length: {len(self.dataset["val"])}\n')
        
        #** Trainer **# 
        trainer = HFTrainer(
            model             = self.model,
            # tokenizer         = self.tokenizer,
            args              = training_args,
            train_dataset     = self.dataset["train"],
            eval_dataset      = self.dataset["val"],
            data_collator     = self.data_collator,
            # compute_metrics   = self.compute_metrics, 
            callbacks         = [logger, debugger, memory_cleanup, early_stopping],
        )


        #** LoRA & DDP Compatibility **#
        if torch.distributed.is_initialized():
            try:
                trainer.model._set_static_graph()  # type: ignore
            except (AttributeError, RuntimeError):
                pass


        trainer.train()
        
        
        #** Save LoRA adapters **#
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(f"{config['output_dir']}/lora_adapters")
        
        #** Cleanup **#
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()



    #** Model Config Check **#
    def check_model_config(self):
        config = self.model.config

        if config.pad_token_id != self.tokenizer.pad_token_id:
            print("Model and tokenizer pad tokens don't match - fixing.")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print("Vocab size:", config.vocab_size)
        print("Tokenizer vocab size:", len(self.tokenizer))
        
        

    # #** Metrics **#
    # def compute_metrics(self, eval_preds,  chunk_size: int = 8):
    #     """Computes perplexity, exact match (EM), and F1. LIGHTER THAN OTHERS
    #     """
    #     preds, labels = eval_preds
        
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     if hasattr(preds, "ndim") and preds.ndim == 3:     # (B, T, V)
    #         preds = preds.argmax(-1)
            
    #     pad_id = getattr(self.tokenizer, "pad_token_id", None)
    #     if pad_id is None:
    #         pad_id = self.tokenizer.eos_token_id  # fallback
    #     labels = np.where(labels != -100, labels, pad_id)
        
    #     preds = np.array(preds)
    #     labels = np.array(labels)
                

    #     pred_texts = []
    #     label_texts = []
    #     n = len(preds)
    #     for i in range(0, n, chunk_size):
    #         chunk_preds = preds[i : i + chunk_size].tolist()
    #         chunk_labels = labels[i : i + chunk_size].tolist()
    #         pred_texts.extend(self.tokenizer.batch_decode(chunk_preds, skip_special_tokens=True))
    #         label_texts.extend(self.tokenizer.batch_decode(chunk_labels, skip_special_tokens=True))


    #     # #** Perplexity **#
    #     # loss = None
    #     # perplexity = None
    #     # if hasattr(eval_preds, "metrics") and "eval_loss" in eval_preds.metrics:
    #     #     loss = eval_preds.metrics["eval_loss"]
    #     #     perplexity = np.exp(loss) if loss < 20 else float("inf")

    #     #** Exact Match & F1 **#
    #     em = float(self.em_metric.compute(predictions=pred_texts, references=label_texts)["exact_match"]) #type: ignore
    #     f1 = float(self.f1_metric.compute(predictions=pred_texts, references=label_texts)["f1"]) #type: ignore

    #     return {
    #         # "perplexity": perplexity,
    #         "exact_match": em,
    #         "f1": f1,
    #     }



    # #** Metrics **#
    # def compute_metrics3(self, eval_pred):
    #     predictions, labels = eval_pred

    #     #** Clear GPU cache **#
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    #     #** Handling Unexpected Inputs & Formatting **#
    #     if isinstance(predictions, tuple):
    #         predictions = predictions[0]

    #     predictions = np.array(predictions)
    #     labels = np.array(labels)
        
    #     if len(predictions.shape) == 3:   # (batch, seq_len, vocab_size)
    #         predictions = np.argmax(predictions, axis=-1)


    #     #** Remove user input tokens and decode **#
    #     predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
    #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
    #     try:
    #         decoded_preds = self.tokenizer.batch_decode(predictions.astype(int), skip_special_tokens=True)
    #         decoded_labels = self.tokenizer.batch_decode(labels.astype(int), skip_special_tokens=True)
    #     except Exception as e:
    #         print(f"Decoding error: {e}")
    #         print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
    #         print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
    #         return {"exact_match": 0.0, "gen_len": 0.0}


    #     #** Normalize text **#
    #     decoded_preds = [pred.strip() for pred in decoded_preds]
    #     decoded_labels = [label.strip() for label in decoded_labels]
    #     decoded_preds_sent = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    #     decoded_labels_sent = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]


    #     #** Metric accumulators **
    #     exact_matches = []
    #     bleu_preds = []
    #     bleu_refs = []
    #     rouge_preds = []
    #     rouge_refs = []

    #     for pred, label, pred_sent, label_sent in zip(decoded_preds, decoded_labels, decoded_preds_sent, decoded_labels_sent):
    #         pred_len = len(pred.split())

    #         #** Exact match **#
    #         exact_matches.append(int(pred == label))

    #         #** BLEU **#
    #         if pred_len >= 10:
    #             bleu_preds.append(pred)
    #             bleu_refs.append([label])

    #         #** ROUGE **#
    #         if pred_len > 30 or '\n' in pred or '\n' in label:
    #             rouge_preds.append(pred_sent)
    #             rouge_refs.append(label_sent)


    #     #** Aggregate metrics **#
    #     results = {}
    #     if exact_matches:
    #         results["exact_match"] = round(np.mean(exact_matches) * 100, 2)
    #     if bleu_preds:
    #         bleu_result = self.bleu.compute(predictions=bleu_preds, references=bleu_refs)
    #         if bleu_result is not None:
    #             results.update(bleu_result)
    #             results["bleu"] = round(results["bleu"] * 100, 2)
    #     if rouge_preds:
    #         rouge_scores = self.rouge.compute(predictions=rouge_preds, references=rouge_refs, use_stemmer=True)
    #         if rouge_scores is not None:
    #             for k, v in rouge_scores.items():
    #                 results[k] = round(v * 100, 2)

    #     results["gen_len"] = round(np.mean([len(p.split()) for p in decoded_preds]), 2)

    #     return results




#! Additional metric functions.


    # # essentially same as compute_metrics3
    # def compute_metrics4(self, eval_pred):
        
    #     predictions, labels = eval_pred
        
    #     # Decode predictions and labels
    #     decoded_preds = []
    #     decoded_labels = []
        
    #     for pred_seq, label_seq in zip(predictions, labels):
    #         valid_label_indices = np.where(label_seq != -100)[0]
    #         if len(valid_label_indices) > 0:
                
    #             # use same positions
    #             pred_tokens = pred_seq[valid_label_indices]
    #             label_tokens = label_seq[valid_label_indices]
                
    #             # Decode
    #             pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
    #             label_text = self.tokenizer.decode(label_tokens, skip_special_tokens=True)
                
    #             decoded_preds.append(pred_text.strip())
    #             decoded_labels.append(label_text.strip())
        
    #     if not decoded_preds:
    #         return {"exact_match": 0.0, "gen_len": 0.0}
        

    #     results = {}
        
    #     # Exact match
    #     exact_matches = [int(pred == label) for pred, label in zip(decoded_preds, decoded_labels)]
    #     results["exact_match"] = round(np.mean(exact_matches) * 100, 2)
        
    #     # ROUGE scores for longer texts
    #     rouge_preds = []
    #     rouge_refs = []
    #     for pred, label in zip(decoded_preds, decoded_labels):
    #         if len(pred.split()) > 5:  # Only compute ROUGE for longer sequences
    #             rouge_preds.append(pred)
    #             rouge_refs.append(label)
        
    #     if rouge_preds:
    #         rouge_scores = self.rouge.compute(
    #             predictions=rouge_preds, 
    #             references=rouge_refs, 
    #             use_stemmer=True
    #         )
    #         if rouge_scores is not None:
    #             for k, v in rouge_scores.items():
    #                 results[k] = round(v * 100, 2)
        
    #     # Average generation length
    #     results["gen_len"] = round(np.mean([len(p.split()) for p in decoded_preds]), 2)
        
    #     return results




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
