from src.utils.misc import training_query, evaluation_query, get_config, parser, modelDev_text2text_query

class FinetunePipeline:
    def __init__(self):
        self.config = get_config()
        
    def run(self):
        run = parser()
        modelDev_text2text_query(run, self.config)
        
        #************ Train ************#
        if run == 'train':
            training_query()
            
            from src.preprocess import DatasetLoader
            dataset = DatasetLoader(self.config, run).load_training_data(self.config['train'])
            
            from src.model_wrapper import FlanT5Wrapper
            model_wrapper = FlanT5Wrapper(run, self.config['train'])
            
            from src.trainer import Trainer
            trainer = Trainer(model_wrapper, dataset, self.config)
            trainer.train(self.config['train'])

        #************ Evaluate ************#
        elif run == 'evaluate':
            evaluation_query()
            
            from src.preprocess import DatasetLoader
            dataset = DatasetLoader(self.config, run).load_evaluation_data(self.config['eval']) #config=self.config['eval']
            
            from src.model_wrapper import FlanT5Wrapper
            model_wrapper = FlanT5Wrapper(run, self.config['eval'])
            
            from src.evaluator import Evaluator
            evaluator = Evaluator(model_wrapper, dataset, self.config)
            evaluator.evaluate(self.config['eval'])