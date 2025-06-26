from datasets import load_dataset, DatasetDict
from transformers import T5Tokenizer

class DatasetLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(self.config['train']['model']['name'])
    
    
    def load_training_data(self):
        """Load and preprocess training data.

        Returns:
            DatasetDict: The tokenized training and validation datasets.
        """
        assert self.config['train']['data']['path']
        dataset = load_dataset("csv", data_files=self.config['train']['data']['path'])["train"]
        train_dataset, val_dataset = dataset.train_test_split(test_size=0.1875).values()
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_val = val_dataset.map(self.preprocess_function, batched=True)
        return DatasetDict({"train": tokenized_train, "val": tokenized_val})

    def load_evaluation_data(self):
        """Load and preprocess evaluation data.

        Returns:
            Dataset: The tokenized evaluation dataset.
        """
        assert self.config['eval']['data']['path']
        raw_test = load_dataset("csv", data_files=self.config['eval']['data']['path'])["train"]
        tokenised_test = raw_test.map(self.preprocess, batched=True)
        return tokenised_test

    def preprocess(self, examples):
        model_inputs = self.tokenizer(
            examples["input"],
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenise targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["output"],
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                padding="max_length",
            )["input_ids"]

        # Loss masking (-100 replaces pad_token_id)  ⇢ list[list[int]]
        labels = [
            [(tok if tok != self.tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs
            
    def preprocess_function(self, examples):
        """Preprocess the input examples for the model.

        Args:
            examples (dict): The input examples to preprocess.

        Returns:
            dict: The preprocessed input examples.
        """
        inputs = [self.config['train']['data']['prefix'] + doc for doc in examples[self.config['train']['data']['input_col']]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config['train']['data']['max_seq_length'],
            padding="max_length",
            truncation=True,
        )

        labels = self.tokenizer(
            examples[self.config['train']['data']['target_col']],
            max_length=self.config['train']['model']['max_target_length'],
            padding="max_length",
            truncation=True,
        )
        label_ids = labels["input_ids"]
        label_ids = [
            [(lid if lid != self.tokenizer.pad_token_id else -100) for lid in label]
            for label in label_ids
        ]
        model_inputs["labels"] = label_ids

        return model_inputs