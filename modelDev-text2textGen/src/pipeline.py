from src.model_wrapper import FlanT5Wrapper
from src.preprocess import DatasetLoader
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils.misc import initialisation_query

class FinetunePipeline:
    def __init__(self, config):
        self.config = config
        initialisation_query(config)
        
    def run(self):
        dataset = DatasetLoader(self.config).load()
        model_wrapper = FlanT5Wrapper(self.config)

        trainer = Trainer(model_wrapper=model_wrapper, dataset=dataset, config=self.config)
        trainer.train()

        # evaluator = Evaluator(model_wrapper=model_wrapper, dataset=dataset, config=self.config)
        # evaluator.evaluate()