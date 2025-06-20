from src.model_wrapper import FlanT5Wrapper
from src.preprocess import DatasetLoader
from src.trainer import Trainer
from src.evaluator import Evaluator

class FinetunePipeline:
    def __init__(self, config):
        self.config = config
        user_input = input("This will fine-tune the FLAN-T5 model. Do you want to proceed? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y']:
            print("Exiting the pipeline.")
            exit(0)
        

    def run(self):
        dataset = DatasetLoader(self.config).load()
        model_wrapper = FlanT5Wrapper(self.config)

        trainer = Trainer(model_wrapper=model_wrapper, dataset=dataset, config=self.config)
        trainer.train()

        # evaluator = Evaluator(model_wrapper=model_wrapper, dataset=dataset, config=self.config)
        # evaluator.evaluate()