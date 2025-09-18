from datasets import load_dataset, DatasetDict, Dataset
 
class DatasetLoader:
    def __init__(self, config, run, model_wrapper):
        self.config = config
        self.tokenizer = model_wrapper.get_tokenizer()
    
    
    #*** Load Data ***#
    def load_training_data(self, config):
        """Load and preprocess training data."""
        
        #** Load data **#
        dataset = load_dataset("csv", data_files=config['data']['dir'], split="train")
        try: dataset = dataset.remove_columns(['id'])
        except: pass
        
        #** Split data **#
        train_dataset, val_dataset = dataset.train_test_split(test_size=config['data']['test_size']).values() # type: ignore
        print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
        
        
        #** Tokenize data **#
        tokenized_train = train_dataset.map(
            lambda x: self.preprocess_mapping(x, config), 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        tokenized_val = val_dataset.map(
            lambda x: self.preprocess_mapping(x, config), 
            batched=True, 
            remove_columns=val_dataset.column_names
        )
        
        return DatasetDict({"train": tokenized_train, "val": tokenized_val})


    def load_evaluation_data(self, config):
        """Load and preprocess evaluation data."""
        
        #** Load data **#
        raw_test = load_dataset("csv", data_files=config['data']['dir'], split="train")
        try: raw_test = raw_test.remove_columns(['id'])
        except: pass
        
        #** Tokenize data **#
        tokenized_test = raw_test.map(
            lambda x: self.preprocess_mapping(x, config), 
            batched=True,
            remove_columns=raw_test.column_names #type: ignore
        )
        
        return tokenized_test



    #*** Preprocess Function ***#
    def preprocess_mapping(self, examples, config):
        """
        Process a batch of examples for causal language modeling.
        examples: dict with keys like 'input', 'output' containing lists of strings
        """        

        formatted_texts = []
        for input_text, output_text in zip(examples[config['data']['input_col']], examples[config['data']['output_col']]):

            #** Ensure strings **#
            if not isinstance(input_text, str):
                input_text = str(input_text) if input_text is not None else ""
            if not isinstance(output_text, str):
                output_text = str(output_text) if output_text is not None else ""
                
            #** Apply chat template **#
            messages = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
            
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            #** Append EOS token if needed **#
            eos_tok = getattr(self.tokenizer, "eos_token", None)
            if eos_tok is not None and not formatted_text.endswith(eos_tok):
                print(f"\nAppending EOS token to formatted text.\n{eos_tok}\n")
                formatted_text = formatted_text + eos_tok

            formatted_texts.append(formatted_text)

        #** Tokenize formatted texts **#
        model_inputs = self.tokenizer(
            formatted_texts,
            max_length=config['data']['max_seq_length'],
            padding=False, 
            truncation=True,
            return_tensors=None
        )

        #** Create label **#
        labels = []
        for input_ids in model_inputs["input_ids"]:
            labels.append(input_ids.copy())

        #** Mask user input in labels **#
        for i, input_text in enumerate(examples[config['data']['input_col']]):
            
            if not isinstance(input_text, str):
                input_text = str(input_text) if input_text is not None else ""
                
            #** Recreate user part with chat template **#
            user_msg = [{"role": "user", "content": input_text}]
            user_formatted = self.tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=False
            )
            
            #** Tokenize user part to get its length **#
            user_tokens = self.tokenizer(user_formatted, add_special_tokens=False)["input_ids"]
            user_length = len(user_tokens)
            
            if user_length >= len(labels[i]):
                print(f"WARNING: computed user_length ({user_length}) >= tokenized example length ({len(labels[i])}).")

            
            #** Mask user part in labels **#
            for j in range(min(user_length, len(labels[i]))):
                labels[i][j] = -100

        
        return {
            "input_ids": model_inputs["input_ids"],
            "labels": labels,
            "attention_mask": model_inputs["attention_mask"]
        }