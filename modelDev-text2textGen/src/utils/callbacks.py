from transformers.trainer_callback import TrainerCallback
import os
import json
import datetime


class LoggingCallback(TrainerCallback):
    def __init__(self, log_dir, log_training_steps=True):
        self.log_dir = log_dir
        self.log_training_steps = log_training_steps
        
        #** Ensure log file **#
        if not os.path.exists(log_dir):
            with open(log_dir, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=4)



    #*** Load and Save Logs ***#
    def load_logs(self) -> list:
        try:
            with open(self.log_dir, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                if isinstance(logs, list):
                    return logs
                else:
                    return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def save_logs(self, logs):
        with open(self.log_dir, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4)
    


    #*** Callbacks ***#
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        logs = self.load_logs()
        
        y = {
            "type": "evaluation",
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": state.epoch,
            "step": state.global_step,
            "metrics": metrics
            }
        
        logs.append(y)
        self.save_logs(logs)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.log_training_steps and logs is not None:
            
            existing_logs = self.load_logs()
            
            y = {
                "type": "train_log",
                "timestamp": datetime.datetime.now().isoformat(),
                "epoch": state.epoch,
                "step": state.global_step,
                "logs": logs
            }
            
            existing_logs.append(y)
            self.save_logs(existing_logs)
        else:
            return 
        
        
#*********-------------- Debug Callback -----------*********#

class DebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, metrics=None, **kwargs):
        if logs and "loss" in logs:
            if logs["loss"] == 0:
                print("WARNING: Zero loss detected!")