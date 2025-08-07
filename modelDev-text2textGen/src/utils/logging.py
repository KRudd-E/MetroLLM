from transformers.trainer_callback import TrainerCallback
import os
import json
import datetime

class LoggingCallback(TrainerCallback):
    def __init__(self, config):
        self.log_dir = config['training_args']['log_dir']

    def on_epoch_end(self, args, state, control, metrics=None, **kwargs):
        
        if not os.path.exists(os.path.join(os.getcwd() + self.log_dir)):
            with open(os.path.join(os.getcwd() + self.log_dir), 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
        
        with open(os.path.join(os.getcwd() + self.log_dir), 'r', encoding='utf-8') as f:
            try:
                z = json.load(f)
            except json.JSONDecodeError:
                z = {}

        y = {state.epoch: {"timestamp": datetime.datetime.now().isoformat(),
                "step": state.global_step,
                "metrics": metrics,}}
        z.update(y) 
        
        with open(os.path.join(os.getcwd() + self.log_dir), 'w', encoding='utf-8') as f:
            json.dump(z, f, indent=4)



class DebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, metrics=None, **kwargs):
        if logs and "loss" in logs:
            if logs["loss"] == 0:
                print("WARNING: Zero loss detected!")
