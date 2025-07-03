def get_config(config_path='dataDev2/config.yaml'):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def t2t_app_query(beginning_subfolder, output_dir, method):
    while True:
        if not beginning_subfolder:
            user_input = input(f"You are about to {method.upper()} '{output_dir}'. Do you wish to continue? (y/n): ")
        else:
            user_input = input(f"You are about to {method.upper()} '{output_dir}', beginning at subfolder '{beginning_subfolder}'. Do you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")


def t2t_def_query(self, method, output):
    while True:
        user_input = input(f"You are about to {method} '{output}'. Do you wish to continue? (y/n): ")
        if user_input.lower is 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")