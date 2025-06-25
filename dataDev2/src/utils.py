def get_config(config_path='dataDev2/config.yaml'):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def t2t_app_query(self):
    while True:
        if not self.config['applicationsDB']['beginning_subfolder']:
            user_input = input(f"You are about to {self.config['applicationsDB']['method'].upper()} '{self.config['applicationsDB']['output']}'. Do you wish to continue? (y/n): ")
        else:
            user_input = input(f"You are about to {self.config['applicationsDB']['method'].upper()} '{self.config['applicationsDB']['output']}', beginning at subfolder '{self.config['applicationsDB']['beginning_subfolder']}'. Do you wish to continue? (y/n): ")
        if user_input in ['Y', 'y', 'N', 'n']:
            if user_input in ['N', 'n']:
                print("Exiting the script.")
                exit()
            break
        else:
            print("Invalid input. Please enter Y or N.")



def t2t_def_query(self):
    while True:
        user_input = input(f"You are about to {self.config['definitionsDB']['method'].upper()} '{self.config['definitionsDB']['output']}'. Do you wish to continue? (y/n): ")
        if user_input in ['Y', 'y', 'N', 'n']:
            if user_input in ['N', 'n']:
                print("Exiting the script.")
                exit()
            break
        else:
            print("Invalid input. Please enter Y or N.")