import numpy as np
import evaluate
import torch
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from src.utils.callbacks import LoggingCallback, DebugCallback

class Trainer_Object:
    def __init__(self, config, model_wrapper):
        self.config = config
        self.model = model_wrapper.get_model()
        self.tokenizer = model_wrapper.get_tokenizer()
        self.data_collator = model_wrapper.get_data_collator()
        self.metric = evaluate.load("f1")
        
        # DEBUG: Print model configuration for debugging
        print(f"DEBUG: Model config num_labels: {self.model.config.num_labels}")
        print(f"DEBUG: Model config problem_type: {self.model.config.problem_type}")
        print(f"DEBUG: Expected class_no from config: {self.config['data']['class_no']}")
        
    def run(self, ds_tok):
        
        # DEBUG: Inspect the dataset structure and label dimensions
        print(f"DEBUG: Dataset structure: {ds_tok}")
        print(f"DEBUG: Train dataset size: {len(ds_tok['train'])}")
        print(f"DEBUG: Test dataset size: {len(ds_tok['test'])}")
        
        # DEBUG: Check sample data to understand label structure
        sample_train = ds_tok["train"][0]
        sample_test = ds_tok["test"][0]
        print(f"DEBUG: Sample train item keys: {sample_train.keys()}")
        print(f"DEBUG: Sample train labels type: {type(sample_train['labels'])}")
        print(f"DEBUG: Sample train labels shape/length: {len(sample_train['labels']) if hasattr(sample_train['labels'], '__len__') else 'scalar'}")
        print(f"DEBUG: Sample train labels: {sample_train['labels']}")
        print(f"DEBUG: Sample test labels shape/length: {len(sample_test['labels']) if hasattr(sample_test['labels'], '__len__') else 'scalar'}")
        
        # DEBUG: Check if all samples have the same label dimensions
        train_label_lengths = [len(item['labels']) for item in ds_tok["train"][:5]]  # Check first 5
        test_label_lengths = [len(item['labels']) for item in ds_tok["test"][:5]]   # Check first 5
        print(f"DEBUG: First 5 train label lengths: {train_label_lengths}")
        print(f"DEBUG: First 5 test label lengths: {test_label_lengths}")
        
        args = TrainingArguments(
            output_dir                    =   str(self.config['training_args']['output_dir']),
            learning_rate                 = float(self.config['training_args']['learning_rate']),
            per_device_train_batch_size   =   int(self.config['training_args']['per_device_train_batch_size']),
            per_device_eval_batch_size    =   int(self.config['training_args']['per_device_eval_batch_size']),
            gradient_accumulation_steps   =   int(self.config['training_args']['gradient_accumulation_steps']),
            num_train_epochs              =   int(self.config['training_args']['num_train_epochs']),
            gradient_checkpointing        =  bool(self.config['training_args']['gradient_checkpointing']),
            warmup_ratio                  = float(self.config['training_args']['warmup_ratio']),
            weight_decay                  = float(self.config['training_args']['weight_decay']),
            eval_strategy                 =   str(self.config['training_args']['eval_strategy']),
            save_strategy                 =   str(self.config['training_args']['save_strategy']),
            save_total_limit              =   int(self.config['training_args']['save_total_limit']),
            metric_for_best_model         =   str(self.config['training_args']['metric_for_best_model']),
            load_best_model_at_end        =  bool(self.config['training_args']['load_best_model_at_end']),
            #logging_dir                  =   str(self.config['training_args']['logging_dir']),
            logging_steps                 =   int(self.config['training_args']['logging_steps']),
            bf16                          =  bool(self.config['training_args']['bf16']),
            push_to_hub                   =  bool(self.config['training_args']['push_to_hub']), # type: ignore
        )
        
        logger = LoggingCallback(self.config['training_args']['logging_dir'], log_training_steps=self.config['training_args']['log_training_steps'])
        debugger = DebugCallback()
        
        trainer = Trainer(
            model             = self.model,
            args              = args,
            train_dataset     = ds_tok["train"],
            eval_dataset      = ds_tok["test"],
            tokenizer         = self.tokenizer,
            data_collator     = self.data_collator,
            compute_metrics   = self.compute_metrics,
            callbacks         = [logger, debugger],
        )
        
        # DEBUG: Test a small batch before training to identify dimension issues
        print("DEBUG: Testing a small batch to identify tensor dimension issues...")
        try:
            # Get a small sample batch using the data collator
            sample_batch = [ds_tok["train"][i] for i in range(min(2, len(ds_tok["train"])))]
            collated_batch = self.data_collator(sample_batch)
            
            print(f"DEBUG: Collated batch keys: {collated_batch.keys()}")
            if 'labels' in collated_batch:
                print(f"DEBUG: Collated labels shape: {collated_batch['labels'].shape}")
                print(f"DEBUG: Collated labels dtype: {collated_batch['labels'].dtype}")
                print(f"DEBUG: Sample collated labels: {collated_batch['labels']}")
            
            # Test model forward pass
            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in collated_batch.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']})
                print(f"DEBUG: Model output logits shape: {outputs.logits.shape}")
                print(f"DEBUG: Expected labels shape should match: {outputs.logits.shape}")
                
        except Exception as e:
            print(f"DEBUG: Error during batch testing: {e}")
            print("DEBUG: This helps identify the exact issue before training starts")
        
        trainer.train()


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        
        # DEBUG: Print tensor shapes during evaluation
        print(f"DEBUG: compute_metrics - logits shape: {logits.shape}")
        print(f"DEBUG: compute_metrics - labels shape: {labels.shape}")
        print(f"DEBUG: compute_metrics - logits type: {type(logits)}")
        print(f"DEBUG: compute_metrics - labels type: {type(labels)}")
        
        assert type(logits) == np.ndarray, "Logits should be a numpy array" # debugging
        probs = 1 / (1 + np.exp(-logits))           # sigmoid
        preds = (probs >= 0.5).astype(int)          # threshold; tune if needed
        
        # DEBUG: Print prediction shapes
        print(f"DEBUG: compute_metrics - predictions shape: {preds.shape}")
        
        result = self.metric.compute(
            predictions=preds, references=labels, average="micro"
        )
        return {"f1_micro": result["f1"] if result and "f1" in result else 0.0}