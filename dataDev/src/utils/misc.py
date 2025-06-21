def get_config(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config