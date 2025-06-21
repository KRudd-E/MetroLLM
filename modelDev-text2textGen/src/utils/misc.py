def initialisation_query(config):
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

def get_config(config_path):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config