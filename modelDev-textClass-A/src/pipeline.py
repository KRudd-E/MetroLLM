from src.utils.misc import get_config, parser, \
            modelDev_textclass_query, setup_training_output_dir
from src.preprocess import Preprocessor
from src.model_wrapper import ClassificationWrapper


class FineTunePipeline:
    def __init__(self):
        self.config = get_config()
        
    def run(self):
        run_mode = parser()
    
        #! INVESTIGATE LIMIT ON LENGTH OF TEXT
        #************ Train ************#
        if run_mode == 'train':
            from src.train import Trainer_Object
            self = setup_training_output_dir(self)
            
            preprocessor = Preprocessor(self.config['train'])
            ds_tok, task_names, class_weights = preprocessor.run()
            
            model_wrapper = ClassificationWrapper(run_mode, self.config['train'], class_weights)
            
            trainer = Trainer_Object(self.config['train'], model_wrapper)
            trainer.run(ds_tok)

        #************ Evaluate ************#
        elif run_mode == 'evaluate':
            from src.evaluate import Evaluator
            
            #model_wrapper = ClassificationWrapper(run_mode, self.config['eval'], class_weights)
            

            # ...
            evaluator = Evaluator(self.config['eval'])
            evaluator.run()