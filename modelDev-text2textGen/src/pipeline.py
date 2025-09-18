from src.utils.utils import training_query, \
    evaluation_query, get_config, parser, modelDev_text2text_query, \
        setup_training_output_dir

class FinetunePipeline:
    def __init__(self):
        self.config = get_config()
        print(f'\nConfig:\n{self.config}\n')
        
    def run(self):
        run = parser()
        #modelDev_text2text_query(run, self.config)
        print(f"\nRunning {run}\n")
        
        from src.preprocess import DatasetLoader
        from src.model_wrapper import FlanT5Wrapper
        
        #** Train **#
        if run == 'train':
            from src.trainer import Trainer
            
            self = setup_training_output_dir(self)
            model_wrapper = FlanT5Wrapper(run, self.config['train'])
            
            ds_loader = DatasetLoader(self.config, run, model_wrapper)
            dataset = ds_loader.load_training_data(self.config['train'])
            
            trainer = Trainer(model_wrapper, dataset, self.config)
            trainer.train(self.config['train'])

        #** Evaluate **#
        elif run == 'evaluate':
            from src.evaluator import Evaluator
            
            model_wrapper = FlanT5Wrapper(run, self.config['eval'])
            
            ds_loader = DatasetLoader(self.config, run, model_wrapper)
            dataset = ds_loader.load_evaluation_data(self.config['eval'])


            evaluator = Evaluator(model_wrapper, dataset, self.config)
            evaluator.evaluate(self.config['eval'])