def get_config(config_path:str='dataDev2/config.yaml') -> dict:
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Script that runs dataDev2")
    parser.add_argument('--model_type', '-mt', type=str, required=True, help='Type of data to create: one of Text2Text, TextGen, or TextClass')
    parser.add_argument('--data_source', '-ds', type=str, required=True, help='Source of the data: one of Applications, Companies, or Definitions')
    inputs = parser.parse_args()
    if inputs.model_type.lower() not in ['text2text', 't2t', 'textgen', 'tg', 'textclass', 'tc']:
        raise ValueError("Invalid model type (mt). Choose from: Text2Text, TextGen, or TextClass.")
    if inputs.data_source.lower() not in ['applications', 'a', 'companies', 'c', 'definitions', 'd']:
        raise ValueError("Invalid data source (ds). Choose from: Applications, Companies, or Definitions.")
    return parser

def dataDev2_query(p1: str, p2: str) -> None:
    while True:
        if p2 == 'a': p2 = 'Applications'
        elif p2 == 'c': p2 = 'Companies'
        elif p2 == 'd': p2 = 'Definitions'
        if p1 == 't2t': p1 = 'Text2Text'
        elif p1 == 'tg': p1 = 'TextGen'
        elif p1 == 'tc': p1 = 'TextClass'
        user_input = input(f"\n         ********************* >>>>> -- DataDev2 -- <<<<< *********************          \n\nThis script will transform decomposed data into .csv files suitable for model development.\nYou have opted to create **{p1}** formatted data, sourced from the **{p2}** database.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': print('\n'), exit()
        else: print("\nInvalid input. Please enter Y or N.")

def t2t_app_query(starting_subfolder: str, output_dir: str, method: str) -> None:
    while True:
        if not starting_subfolder and method.lower() == 'overwrite':
            user_input = input(f"\nYou have chosen to {method.upper()} '{output_dir[1:]}'.\n Do you wish to continue? (y/n): ")
        elif not starting_subfolder and method.lower() == 'append':
            user_input = input(f"\nYou have chosen to {method.upper()} to '{output_dir[1:]}'.\n Do you wish to continue? (y/n): ")
        elif starting_subfolder and method.lower() == 'overwrite':
            user_input = input(f"\nYou have chosen to {method.upper()} '{output_dir[1:]}', starting at subfolder '{starting_subfolder}'.\nDo you wish to continue? (y/n): ")
        elif starting_subfolder and method.lower() == 'append':
            user_input = input(f"\nYou have chosen to {method.upper()} to '{output_dir[1:]}', starting at subfolder '{starting_subfolder}'.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': print('\n'), exit()
        else: print("Invalid input. Please enter Y or N.")


def t2t_def_query(method: str, output: str) -> None:
    while True:
        user_input = input(f"You are about to {method} '{output}'. Do you wish to continue? (y/n): ")
        if user_input.lower == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")