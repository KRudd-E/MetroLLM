# File: src/models/model_wrapper.py
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch


class DeepSeekWrapper:
    def __init__(self, run, config):
       
        #***** Train *****#
        if run == 'train':
            
            #** 4-Bit Quantization **#
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            #** Model Loader **#
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model']['name'],
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True, 
                trust_remote_code=True,
                quantization_config=bnb_config
            )
            
            #** LoRA Setup **#
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config['lora']['rank'],
                lora_alpha=config['lora']['alpha'],
                lora_dropout=config['lora']['dropout'],
                target_modules=config['lora']['target_modules'],
            )
            self.model = get_peft_model(self.model, lora_config)
            
            # Enable gradient checkpointing compatibility with PEFT
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads() # type: ignore
            
            # Ensure model is in training mode
            self.model.train()
            
            # Debug: Print parameter status
            trainable_params = 0
            all_params = 0
            for name, param in self.model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    
            print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}")
            
            self.model.print_trainable_parameters()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['model']['name'], 
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id # type: ignore
            
            #** Data Collator **#
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False,  # Causal LM (chat), not masked LM (e.g., direct classification)
                pad_to_multiple_of=8
            )
        
        #***** Eval *****#
        elif run == 'eval': 
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model']['dir'], 
                low_cpu_mem_usage=True, 
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['model']['dir'], 
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
                
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False,
                pad_to_multiple_of=8
            )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device

    def get_data_collator(self):
        return self.data_collator
