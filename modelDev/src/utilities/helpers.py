def save_checkpoint(model, path):
    print(f"Saving model to {path}")
    # Actual saving logic here

def load_checkpoint(path):
    print(f"Loading model from {path}")
    # Actual loading logic here

def format_prompt(instruction, input_text=""):
    return f"{instruction}\n{input_text}"