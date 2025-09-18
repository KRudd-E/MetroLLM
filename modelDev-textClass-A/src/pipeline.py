# pipeline.py
# This script contains the FineTunePipeline class which employs the various TC-A modules to run either training or evaluation.

from src.utils.misc import get_config, parser, setup_training_output_dir
from src.utils.disable_compilation import disable_compilation
from src.preprocess import Preprocessor
from src.model_wrapper import WeightedBCEModelWrapper, compute_pos_weights

class FineTunePipeline:
    def __init__(self):
        disable_compilation()
        self.config = get_config()
        
    def run(self):
        run_mode = parser()
    
        #** Train **#
        if run_mode == 'train':
            from src.train import Trainer_Object

            setup_training_output_dir(self)
            
            preprocessor = Preprocessor(self.config['train'])
            ds_tok, task_names, y_labels = preprocessor.run(run_mode)
            
            pos_weights = compute_pos_weights(y_labels)
            model_wrapper = WeightedBCEModelWrapper(self.config['train'], pos_weights)

            trainer = Trainer_Object(self.config['train'], model_wrapper, pos_weights)
            trainer.run(ds_tok)


        #** Evaluate **#
        elif run_mode == 'evaluate':
            from src.evaluate import Evaluator

            preprocessor = Preprocessor(self.config['eval'])
            ds_tok, task_names, y_labels = preprocessor.run(run_mode)

            pos_weights = compute_pos_weights(y_labels)
            model_wrapper = WeightedBCEModelWrapper(self.config['eval'], pos_weights, checkpoint_dir=self.config['eval']['model']['checkpoint_dir'])

            evaluator = Evaluator(self.config['eval'], model_wrapper, task_names)
            evaluator.run(ds_tok)
            