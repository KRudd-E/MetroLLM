from datasets import load_dataset, DatasetDict
 
class DatasetLoader:
    def __init__(self, config, run, model_wrapper):
        self.config = config
        self.tokenizer = model_wrapper.get_tokenizer()

    
    def load_training_data(self, config):
        """Load and preprocess training data."""
        dataset = load_dataset("csv", data_files=config['data']['dir'], split="train")
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.1875).values()  # type: ignore
        tokenized_train = train_dataset.map(lambda x: self.preprocess_mapping(x, config), batched=True)
        tokenized_val = val_dataset.map(lambda x: self.preprocess_mapping(x, config), batched=True)
        return DatasetDict({"train": tokenized_train, "val": tokenized_val})


    def load_evaluation_data(self, config):
        """Load and preprocess evaluation data."""
        raw_test = load_dataset("csv", data_files=config['data']['dir'], split="train")
        tokenized_test = raw_test.map(lambda x: self.preprocess_mapping(x, config), batched=True)
        return tokenized_test


    def preprocess_mapping(self, examples, config):
        """Preprocess input examples for training or evaluation."""
        
        model_inputs = self.tokenizer(
            examples[config['data']['input_col']],
            max_length=config.get('data', {}).get('max_seq_length', self.tokenizer.model_max_length),
            padding="max_length",
            truncation=True,
        )

        #** Tokenize targets **#
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples[config['data']['output_col']],
                max_length=config.get('model', {}).get('max_target_length', self.tokenizer.model_max_length),
                padding="max_length",
                truncation=True,
            )["input_ids"]

        #** Replace padding token ids in labels by -100 to ignore in loss **#
        model_inputs["labels"] = [
            [(tok if tok != self.tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels
        ]

        return model_inputs
