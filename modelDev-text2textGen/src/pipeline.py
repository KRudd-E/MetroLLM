from src.model_wrapper import FlanT5Wrapper
from src.preprocess import DatasetLoader
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils.misc import training_query, evaluation_query

class FinetunePipeline:
    def __init__(self, config):
        self.config = config
        if config['train_or_eval'] == 'train': training_query(config)
        elif config['train_or_eval'] == 'eval': evaluation_query(config)
        else: raise ValueError("Invalid value for 'train_or_eval' in config. Must be 'train' or 'eval'.")
        
    def run(self):
        
        if self.config['train_or_eval'] == 'train':
            dataset = DatasetLoader(self.config).load_training_data()
            model_wrapper = FlanT5Wrapper(self.config)
            trainer = Trainer(model_wrapper=model_wrapper, dataset=dataset, config=self.config)
            trainer.train()

        elif self.config['train_or_eval'] == 'eval':
            dataset = DatasetLoader(self.config).load_evaluation_data()
            model_wrapper = FlanT5Wrapper(self.config)
            evaluator = Evaluator(model_wrapper=model_wrapper, dataset=dataset, config=self.config)
            evaluator.evaluate()