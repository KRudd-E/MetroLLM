"""

This file holds configuration settings - like hyperparamters - 
so they can be easily managed and updated.

"""

class Config:
    def __init__(self):
        
        self.data_path: str = '../data/train.jsonl'
        self.out_path: str = './outputs'
        self.model_name: str = 'deepseek-ai/deepseek-llm-r-1.5b'  # Huggingface name
        
        self.learning_rate = 2e-5
        self.batch_size = 2
        self.num_epochs = 3
        self.max_seq_length = 2048
        self.device = 'cuda'




