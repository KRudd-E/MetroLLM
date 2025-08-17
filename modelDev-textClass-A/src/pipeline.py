from src.preprocess import Preprocessor
from src.train import Trainer_Object
from src.utils.misc import get_config, parser, modelDev_textclass_query

class FineTunePipeline:
    def __init__(self):
        self.config = get_config()
        
    def run(self):
        run = parser()
        #modelDev_textclass_query(run, self.config)
        
        from src.preprocess import Preprocessor
        from src.model_wrapper import ClassificationWrapper
        
        #! INVESTIGATE LIMIT ON LENGTH OF TEXT
        #************ Train ************#
        if run == 'train':
            from src.train import Trainer_Object
            
            model_wrapper = ClassificationWrapper(run, self.config['train'])
            
            preprocessor = Preprocessor(self.config['train'], model_wrapper)
            ds_tok, task_names = preprocessor.run()
            
            trainer = Trainer_Object(self.config['train'], model_wrapper)
            trainer.run(ds_tok)
    
        #************ Evaluate ************#
        if run == 'evaluate':
            from src.evaluate import Evaluator
            
            model_wrapper = ClassificationWrapper(run, self.config['eval'])
            

            # ...
            evaluator = Evaluator(self.config['eval'])
            evaluator.run()