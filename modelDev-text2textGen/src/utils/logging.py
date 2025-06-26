from transformers import TrainerCallback
import os

import datetime

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path="training_logs.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": state.epoch,
            "step": state.global_step,
            "metrics": metrics,
        }
        # Ensure the log file exists and write headers if it's empty
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("timestamp,epoch,step,metrics\n")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{log_entry['timestamp']},{log_entry['epoch']},\"{log_entry['step']},{log_entry['metrics']}\"\n")

class DebugCallback(TrainerCallback):
    def on_step_end(self, state, logs=None):
        if logs and "loss" in logs:
            if logs["loss"] == 0:
                print("WARNING: Zero loss detected!")

