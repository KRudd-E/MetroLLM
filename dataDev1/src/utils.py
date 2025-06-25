def get_config(config_path="dataDev1/config.yaml"):
    """Load configuration settings from a YAML file.

    Args:
        config_path (str, optional): Path to the YAML config file. Defaults to "dataDev1/config.yaml".

    Returns:
        dict: Configuration settings.
    """
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def applicationsDB_initialisation_query():
    """Prompt user for applications DB processing confirmation.
    """
    while True:
        user_input = input("Do you want to process the applications database? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y', 'no', 'n']:
            print("Invalid input. Please enter 'yes' or 'no'.")
        elif user_input in ['no', 'n']:
            print("Exiting the pipeline.")
            exit(0)
        else:
            break

def companiesDB_initialisation_query():
    """Prompt user for companies DB processing confirmation.
    """
    while True:
        user_input = input("This will process the companies database. Do you want to proceed? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y', 'no', 'n']:
            print("Invalid input. Please enter 'yes' or 'no'.")
        elif user_input in ['no', 'n']:
            print("Exiting the pipeline.")
            exit(0)
        else:
            break

def definitionsDB_initialisation_query():
    """Prompt user for definitions DB processing confirmation.
    """
    while True:
        user_input = input("This will process the definitions database. Do you want to proceed? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y', 'no', 'n']:
            print("Invalid input. Please enter 'yes' or 'no'.")
        elif user_input in ['no', 'n']:
            print("Exiting the pipeline.")
            exit(0)
        else:
            break

