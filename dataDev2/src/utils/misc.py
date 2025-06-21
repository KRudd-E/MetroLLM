def get_config(config_path='dataDev2/config.yaml'):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config