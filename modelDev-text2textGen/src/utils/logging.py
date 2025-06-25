from transformers import TrainerCallback
import os
import json

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path="training_logs.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        log_entry = {
            "epoch": state.epoch,
            "step": state.global_step,
            "metrics": metrics,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")