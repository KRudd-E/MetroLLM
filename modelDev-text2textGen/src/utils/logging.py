from transformers import TrainerCallback
import os
import json
import datetime

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path="training_logs.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def on_evaluate(self, state, metrics=None):
        if metrics is None:
            return

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": state.epoch,
            "step": state.global_step,
            "metrics": metrics,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

class DebugCallback(TrainerCallback):
    def on_step_end(self, state, logs=None):
        if logs and "loss" in logs:
            if logs["loss"] == 0:
                print("WARNING: Zero loss detected!")

