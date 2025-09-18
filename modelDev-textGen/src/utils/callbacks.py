from transformers.trainer_callback import TrainerCallback
import os
import json
import datetime
import torch


def is_main_process():
    """Return True only on process rank 0."""
    rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or os.environ.get("LOCAL_RANK")
    try:
        return int(rank) == 0 # type: ignore
    except Exception:
        return True


class LoggingCallback(TrainerCallback):
    """ Append-only JSON logging. Only rank-0 writes to avoid copies."""
    def __init__(self, log_path, log_training_steps=True):
        self.log_path = log_path
        self.log_training_steps = log_training_steps

        #** Setup log file **#
        if is_main_process() and not os.path.exists(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                pass

    #** Append log entry **#
    def _append(self, record: dict):
        if not is_main_process():
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"LoggingCallback append error: {e}")


    #** Callback methods **#
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        y = {
            "type": "evaluation",
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": state.epoch,
            "step": state.global_step,
            "metrics": metrics
        }
        self._append(y)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.log_training_steps or logs is None:
            return
        y = {
            "type": "train_log",
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": state.epoch,
            "step": state.global_step,
            "logs": logs
        }
        self._append(y)



#******* Debug ******#

class DebugCallback(TrainerCallback):
    def on_step_end(self, args, state, control, logs=None, metrics=None, **kwargs):
        if logs and "loss" in logs:
            if logs["loss"] is not None and logs["loss"] == 0:
                print("WARNING: Zero loss detected!")



#****** Memory Cleanup ******#

class MemoryCleanupCallback(TrainerCallback):
    """ Keep GPU memory low and only run expensive cleanup occasionally. """
    def __init__(self, aggressive_every=3, periodic_step=500):
        self.eval_count = 0
        self.aggressive_every = aggressive_every
        self.periodic_step = periodic_step

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_count += 1
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            #** Aggressive cleanup every 3 evals **#
            if self.eval_count % self.aggressive_every == 0:
                try:
                    torch.cuda.synchronize()
                    torch.cuda.ipc_collect()
                    print(f"Aggressive memory cleanup performed after eval #{self.eval_count}")
                except Exception:
                    pass

    def on_save(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.periodic_step == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_log(self, args, state, control, logs=None, **kwargs):
        return