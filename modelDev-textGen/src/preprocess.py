from datasets import load_dataset, DatasetDict, Dataset
 
class DatasetLoader:
    def __init__(self, config, run, model_wrapper):
        self.config = config
        self.tokenizer = model_wrapper.get_tokenizer()
    
    
    #*** Load Data ***#
    def load_training_data(self, config):
        """Load and preprocess training data."""
        
        dataset = load_dataset("csv", data_files=config['data']['dir'], split="train")
        # Remove 'id' column if it exists
        try:
            dataset = dataset.remove_columns(['id'])
        except:
            pass
        
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.15).values()  # type: ignore
        
        # Use remove_columns to ensure only tokenized features remain
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
        
        raw_test = load_dataset("csv", data_files=config['data']['dir'], split="train")
        # Remove 'id' column if it exists
        try:
            raw_test = raw_test.remove_columns(['id'])
        except:
            pass
            
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
            # Ensure strings
            if not isinstance(input_text, str):
                input_text = str(input_text) if input_text is not None else ""
            if not isinstance(output_text, str):
                output_text = str(output_text) if output_text is not None else ""
                
            # Create chat format
            messages = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
            
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)

        # Tokenize all formatted texts
        model_inputs = self.tokenizer(
            formatted_texts,
            max_length=config['data']['max_seq_length'],
            padding=False, 
            truncation=True,
            return_tensors=None
        )

        # Create labels (copy of input_ids)
        labels = []
        for input_ids in model_inputs["input_ids"]:
            labels.append(input_ids.copy())

        # Mask the user input part in labels
        for i, input_text in enumerate(examples[config['data']['input_col']]):
            if not isinstance(input_text, str):
                input_text = str(input_text) if input_text is not None else ""
                
            # Create user-only message to find where to start loss computation
            user_msg = [{"role": "user", "content": input_text}]
            user_formatted = self.tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize user part to find length
            user_tokens = self.tokenizer(user_formatted, add_special_tokens=False)["input_ids"]
            user_length = len(user_tokens)
            
            # Mask user input tokens in labels (set to -100)
            for j in range(min(user_length, len(labels[i]))):
                labels[i][j] = -100

        # Return only the required tokenized features
        return {
            "input_ids": model_inputs["input_ids"],
            "labels": labels,
            "attention_mask": model_inputs["attention_mask"]
        }