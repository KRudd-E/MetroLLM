from datasets import load_dataset, DatasetDict
from transformers import T5Tokenizer

class DatasetLoader:
    def __init__(self, config):
        self.train_path = config["data"]["train_path"]
        #self.val_path = config["data"].get("val_path")
        self.input_col = config["data"].get("input_col", "input")
        self.target_col = config["data"].get("target_col", "output")
        self.prefix = config["data"].get("prefix", "")

        self.model_name = config["model"]["name"]
        self.max_input_length = config["model"].get("max_input_length", 128)
        self.max_target_length = config["model"].get("max_target_length", 64)

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    def load(self):
        dataset = load_dataset("csv", data_files=self.train_path)["train"]

        train_dataset, val_dataset = dataset.train_test_split(test_size=0.3).values()

        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_val = val_dataset.map(self.preprocess_function, batched=True)

        return DatasetDict({"train": tokenized_train, "val": tokenized_val})

    def preprocess_function(self, examples):
        inputs = [self.prefix + doc for doc in examples[self.input_col]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
        )

        labels = self.tokenizer(
            examples[self.target_col],
            max_length=self.max_target_length,
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
