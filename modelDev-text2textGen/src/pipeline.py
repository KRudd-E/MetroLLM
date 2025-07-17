from src.utils.utils import training_query, \
    evaluation_query, get_config, parser, modelDev_text2text_query, \
        setup_training_output_dir

class FinetunePipeline:
    def __init__(self):
        self.config = get_config()
        
    def run(self):
        run = parser()
        #modelDev_text2text_query(run, self.config)
        
        #************ Train ************#
        if run == 'train':
            
            from src.preprocess import DatasetLoader
            from src.model_wrapper import FlanT5Wrapper
            from src.trainer import Trainer
            
            #training_query(self)
            setup_training_output_dir(self)
            
            dataset = DatasetLoader(self.config, run)
            dataset.load_training_data(self.config['train'])
            
            model_wrapper = FlanT5Wrapper(run, self.config['train'])
            
            trainer = Trainer(model_wrapper, dataset, self.config)
            trainer.train(self.config['train'])

        #************ Evaluate ************#
        elif run == 'evaluate':
            
            from src.preprocess import DatasetLoader
            from src.model_wrapper import FlanT5Wrapper
            from src.evaluator import Evaluator
            
            #evaluation_query()
            
            dataset = DatasetLoader(self.config, run).load_evaluation_data(self.config['eval'])
            
            model_wrapper = FlanT5Wrapper(run, self.config['eval'])
            
            evaluator = Evaluator(model_wrapper, dataset, self.config)
            evaluator.evaluate(self.config['eval'])