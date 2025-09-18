from src.utils.utils import get_config, parser, setup_training_output_dir, setup_distributed

class FinetunePipeline:
    def __init__(self):
        self.config = get_config()
        setup_distributed()
        
    def run(self):
        run = parser()
        print(f"\nRunning {run}\n")
        
        from src.preprocess import DatasetLoader
        from src.model_wrapper import DeepSeekWrapper
        
        #** Train **#
        if run == 'train':
            from src.trainer import Trainer
            
            setup_training_output_dir(self)
            model_wrapper = DeepSeekWrapper(run, self.config['train'])
            
            ds_loader = DatasetLoader(self.config, run, model_wrapper)
            dataset = ds_loader.load_training_data(self.config['train'])
            
            trainer = Trainer(model_wrapper, dataset, self.config)
            trainer.train(self.config['train'])
            
            model_wrapper.destroy_process_group()

        #** Evaluate **#
        elif run == 'evaluate':
            from src.evaluator import MMLU_Evaluator, Test_Set_Evaluator, Task_Evaluator
            
            model_wrapper = DeepSeekWrapper(run, self.config['eval'])
            
            ds_loader = DatasetLoader(self.config, run, model_wrapper)
            dataset = ds_loader.load_evaluation_data(self.config['eval'])

            mmlu_evaluator = MMLU_Evaluator(model_wrapper, self.config['eval']['mmlu_args'])
            mmlu_evaluator.evaluate()
            
            test_set_evaluator = Test_Set_Evaluator(model_wrapper, self.config['eval']['test_set_args'], dataset)
            test_set_evaluator.evaluate()
            
            task_evaluator = Task_Evaluator(model_wrapper, self.config['eval']['task_args'])
            task_evaluator.evaluate()
            
            model_wrapper.destroy_process_group()