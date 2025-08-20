from src.utils.misc import get_config, parser, \
            modelDev_textclass_query, setup_training_output_dir
from src.utils.disable_compilation import disable_compilation
from src.preprocess import Preprocessor
from src.model_wrapper import ClassificationWrapper


class FineTunePipeline:
    def __init__(self):
        disable_compilation() 
        self.config = get_config()
        
    def run(self):
        run_mode = parser()
    
        #! INVESTIGATE LIMIT ON LENGTH OF TEXT
        #************ Train ************#
        if run_mode == 'train':
            from src.train import Trainer_Object
            from src.model_wrapper import compute_pos_weights
            
            self = setup_training_output_dir(self)
            
            preprocessor = Preprocessor(self.config['train'])
            ds_tok, task_names, y_labels = preprocessor.run()
            
            # Compute positive weights for weighted BCE
            pos_weights = compute_pos_weights(y_labels)
            print(f"Computed positive weights for {len(task_names)} classes:")
            for i, (name, weight) in enumerate(zip(task_names, pos_weights)):
                print(f"  {name}: {weight:.3f}")
            
            model_wrapper = ClassificationWrapper(run_mode, self.config['train'], pos_weights)
            
            trainer = Trainer_Object(self.config['train'], model_wrapper)
            trainer.run(ds_tok)

        #************ Evaluate ************#
        elif run_mode == 'evaluate':
            from src.evaluate import Evaluator
            
            evaluator = Evaluator(self.config['eval'])
            results = evaluator.run()
            print(f"\nEvaluation completed. Results saved to evaluation output.")