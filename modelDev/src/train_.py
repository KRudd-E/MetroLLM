""" 
For training (fine-tuning) the model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

class TrainerWrapper:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True)

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["instruction"] + " " + examples["input"],
            text_target=examples["output"],
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length
        )

    def train_model(self, dataset):
        tokenized = dataset.map(self.tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            logging_steps=10,
            save_total_limit=2,
            save_strategy="epoch",
            evaluation_strategy="no"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.model.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")