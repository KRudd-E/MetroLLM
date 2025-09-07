# File: src/models/model_wrapper.py
from transformers import AutoTokenizer, AutoModelForCausalLM, \
    DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import os


class DeepSeekWrapper:
    def __init__(self, run, config):
       
        #***** Train *****#
        if run == 'train':
            
            #** Map device(s) **#
            if torch.cuda.is_available():
                is_distributed = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ) or \
                               (torch.distributed.is_available() and torch.distributed.is_initialized())
                
                if is_distributed:
                    local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    device_map = {"": local_rank}
                    self.device = torch.device(f"cuda:{local_rank}")
                    print(f"Distributed training detected. Using device {local_rank} for local rank {local_rank}\n")
                    print(f"Global rank: {os.environ.get('RANK', 'unknown')}, World size: {os.environ.get('WORLD_SIZE', 'unknown')}\n")
                else:
                    device_map = {"": torch.cuda.current_device()}
                    self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
                    print(f"Single GPU training. Using device {torch.cuda.current_device()}")
            else:
                raise EnvironmentError("No CUDA available.")
            
            #** 4-Bit Quantization **#
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            #** Model Loader **#
            self.model = AutoModelForCausalLM.from_pretrained(
                config['model']['name'],
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True, 
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map=device_map,
                # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            )
            
            # Clear cache after model loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Model loaded. Memory usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else "Model loaded on CPU")

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
            
            #** Enable gradient checkpointing compatibility with PEFT **#
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads() # type: ignore
            
            #** Ensure training mode & print parameters **#
            self.model.train()
            self.model.print_trainable_parameters()
            
            #** Tokenizer **#
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['model']['name'], 
                trust_remote_code=True
            )
            
            #** Padding token handling **#
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id # type: ignore
            
            #** Data Collator **#
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False,  # Causal LM (chat), not masked LM (e.g., direct classification)
                pad_to_multiple_of=8,
                return_tensors="pt"
            )
            
        #***** Eval *****#
        elif run == 'evaluate':

            # ** Device map **
            if torch.cuda.is_available():
                device_map = "auto"
                print(f"Evaluation using available GPU(s) with device_map={device_map}")
            else:
                device_map = {"": "cpu"}
                print("\nUsing CPU for evaluation!\n")

            # ** Base Model **
            base_model = AutoModelForCausalLM.from_pretrained(
                config['model']['name'],
                device_map=device_map,
                dtype=torch.float16,   # faster on A40
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # ** Load Peft Model (LoRA) **
            peft_model = PeftModel.from_pretrained(
                base_model,
                model_id=config['model']['dir'],
                device_map=device_map,
            )

            # ** Merge LoRA into base weights for faster inference **
            self.model = peft_model.merge_and_unload() # type: ignore

            # ** Tokenizer **
            self.tokenizer = AutoTokenizer.from_pretrained(
                config['model']['name'],
                trust_remote_code=True
            )

            # ** Ensure Padding Token **
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id  # type: ignore

            # ** Data Collator **
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
