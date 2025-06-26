def training_query(config):
    while True:
        user_input = input("This will fine-tune the FLAN-T5 model. Do you want to proceed? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y', 'no', 'n']:
            print("Invalid input. Please enter 'yes' or 'no'.")
        elif user_input in ['no', 'n']:
            print("Exiting the pipeline.")
            exit(0)
        else:
            print("Proceeding with the fine-tuning process.")
            break

def evaluation_query(config):
    while True:
        user_input = input("This will evaluate the fine-tuned FLAN-T5 model. Do you want to proceed? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y', 'no', 'n']:
            print("Invalid input. Please enter 'yes' or 'no'.")
        elif user_input in ['no', 'n']:
            print("Exiting the pipeline.")
            exit(0)
        else:
            print("Proceeding with the evaluation process.")
            break

def get_config(config_path):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def eval_to_jsonl(eval_results, output_path):
    """
    Save evaluation results to a JSONL file.

    Args:
        eval_results (dict): The evaluation results to save.
        output_path (str): The path to the output JSONL file.
    """
    import os
    import json
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(eval_results) + "\n")