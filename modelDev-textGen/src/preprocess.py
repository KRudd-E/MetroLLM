from datasets import load_dataset, DatasetDict, Dataset
 
class DatasetLoader:
    def __init__(self, config, run, model_wrapper):
        self.config = config
        self.tokenizer = model_wrapper.get_tokenizer()
    
    
    #*** Load Data ***#
    def load_training_data(self, config):
        """Load and preprocess training data."""
        
        dataset = load_dataset("csv", data_files=config['data']['dir'], split="train")
        dataset.column_names
        # Remove 'id' column if it exists
        try:
            dataset = dataset.remove_columns(['id'])
        except:
            pass
        
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.1875).values()  # type: ignore
        
        tokenized_train = train_dataset.map(lambda x: self.preprocess_mapping(x, config), batched=True)
        tokenized_val = val_dataset.map(lambda x: self.preprocess_mapping(x, config), batched=True)
        
        return DatasetDict({"train": tokenized_train, "val": tokenized_val})

    def load_evaluation_data(self, config):
        """Load and preprocess evaluation data."""
        
        raw_test = load_dataset("csv", data_files=config['data']['dir'], split="train")
        tokenized_test = raw_test.map(lambda x: self.preprocess_mapping(x, config), batched=True)
        
        return tokenized_test


    #*** Preprocess Function ***#
    def preprocess_mapping(self, examples, config):
        
        # https://huggingface.co/docs/transformers/main/chat_templating
        
        # Format text using chat template @ hugging face
        formatted_texts = []
        for input_text, output_text in zip(examples[config['data']['input_col']], examples[config['data']['output_col']]):
            
            messages = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text}
            ]
            
            # ERROR: TypeError: can only concatenate str (not "NoneType") to str
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)
        
        # Tokenize formatted text
        model_inputs = self.tokenizer(
            formatted_texts,
            max_length=config.get('data', {}).get('max_seq_length', 2048),
            padding="max_length",
            truncation=True,
            return_tensors=None
        )
        
        
        # Label masking: masks the input in labels so the model only learns from the output
        model_inputs["labels"] = model_inputs["input_ids"].copy()
    
        for i, (input_text, formatted_text) in enumerate(zip(examples[config['data']['input_col']], formatted_texts)):
            
            # Find where assistant response starts
            user_msg = [{"role": "user", "content": input_text}]
            user_formatted = self.tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize just the user part to find where to start computing loss
            user_tokens = self.tokenizer(user_formatted, add_special_tokens=False)["input_ids"]
            user_length = len(user_tokens)
            
            # Mask the user input part in labels by setting to -100 
            for j in range(min(user_length, len(model_inputs["labels"][i]))):
                model_inputs["labels"][i][j] = -100

        return model_inputs



    # # https://www.digitalocean.com/community/tutorials/fine-tuning-deepseek-medical-cot
    # def formatting_prompts_func(examples):
    #     inputs = examples["Question"]
    #     cots = examples["Complex_CoT"]
    #     outputs = examples["Response"]
    #     texts = []
    #     for input, cot, output in zip(inputs, cots, outputs):
    #         text = prompt_template.format(input, cot, output) + tokenizer.eos_token
    #         texts.append(text)
    #     return {
    #         "text": texts,
    #     }