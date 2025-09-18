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
        
        


    #? Currently crashes system - memory overload

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


    #     #** Perplexity **#
    #     loss = None
    #     perplexity = None
    #     if hasattr(eval_preds, "metrics") and "eval_loss" in eval_preds.metrics:
    #         loss = eval_preds.metrics["eval_loss"]
    #         perplexity = np.exp(loss) if loss < 20 else float("inf")

    #     #** Exact Match & F1 **#
    #     em = float(self.em_metric.compute(predictions=pred_texts, references=label_texts)["exact_match"]) #type: ignore
    #     f1 = float(self.f1_metric.compute(predictions=pred_texts, references=label_texts)["f1"]) #type: ignore

    #     return {
    #         # "perplexity": perplexity,
    #         "exact_match": em,
    #         "f1": f1,
    #     }

