"""

To tie the pipeline together in order.
Running this code gathers data, preprocesses, trains, tests, 
evaluates, and provides various utilites.

"""

from preprocess import DataPreprocessor
from train_ import TrainerWrapper
from test_ import Tester
from evaluate import Evaluator
import time

class Controller:
    def __init__(self, config):
        
        self.config = config
        if self.config['train_text_classification'] is True and self.config['train_text_generation'] is True:
            raise ValueError("Cannot train both text classification and text generation at the same time. Please set one to False.")
        
        self.preprocessor = DataPreprocessor(self.config)
        self.trainer = TrainerWrapper(self.config)
        self.tester = Tester(self.config)
        self.evaluator = Evaluator(self.config)

    def run(self):
        
        #************ Grab Data ************#
        print(">> Gathering and Tokenizing Data")
        dataset = self.preprocessor.run()
        print(">> Gathered and Tokenized Data")
        time.sleep(self.config['sleep'])
        
        #************ Fine Tune ************#
        print(f">> Fine Tuning {self.config['model_name']}")
        self.trainer.train_model(dataset)
        
        
        
        self.tester.run_tests()
        self.evaluator.evaluate_model()