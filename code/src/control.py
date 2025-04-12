"""

To tie the pipeline together in order.
Running this code gathers data, preprocesses, trains, tests, 
evaluates, and provides various utilites.

"""

from config import Config
from preprocess import DataPreprocessor
from train import Trainer
from test_ import Tester
from evaluate import Evaluator

class Controller:
    def __init__(self):
        self.config = Config()
        self.preprocessor = DataPreprocessor(self.config)
        self.trainer = Trainer(self.config)
        self.tester = Tester(self.config)
        self.evaluator = Evaluator(self.config)

    def run(self):
        self.preprocessor.prepare_data()
        self.trainer.train_model()
        self.tester.run_tests()
        self.evaluator.evaluate_model()

if __name__ == "__main__":
    Controller().run()