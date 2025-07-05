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
    return parser.parse_args()

def dataDev2_query(args) -> None:
    while True:
        if args.model_type == 't2t': args.model_type = 'Text2Text'
        elif args.model_type == 'tg': args.model_type = 'TextGen'
        elif args.model_type == 'tc': args.model_type = 'TextClass'
        if args.data_source == 'a': args.data_source = 'Applications'
        elif args.data_source == 'c': args.data_source = 'Companies'
        elif args.data_source == 'd': args.data_source = 'Definitions'
        user_input = input(f"\n         ********************* >>>>> -- DataDev2 -- <<<<< *********************          \n\nThis script will transform decomposed data into .csv files suitable for model development.\nYou have opted to create **{args.model_type}** formatted data, sourced from the **{args.data_source}** database.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("\nInvalid input. Please enter Y or N.")

def t2t_app_query(starting_subfolder: str, output_dir: str, append_or_overwrite: str) -> None:
    while True:
        if not starting_subfolder and append_or_overwrite.lower() == 'overwrite':
            user_input = input(f"\nYou have chosen to {append_or_overwrite.upper()} '{output_dir[1:]}'.\n Do you wish to continue? (y/n): ")
        elif not starting_subfolder and append_or_overwrite.lower() == 'append':
            user_input = input(f"\nYou have chosen to {append_or_overwrite.upper()} to '{output_dir[1:]}'.\n Do you wish to continue? (y/n): ")
        elif starting_subfolder and append_or_overwrite.lower() == 'overwrite':
            user_input = input(f"\nYou have chosen to {append_or_overwrite.upper()} '{output_dir[1:]}', starting at subfolder '{starting_subfolder}'.\nDo you wish to continue? (y/n): ")
        elif starting_subfolder and append_or_overwrite.lower() == 'append':
            user_input = input(f"\nYou have chosen to {append_or_overwrite.upper()} to '{output_dir[1:]}', starting at subfolder '{starting_subfolder}'.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")


def t2t_def_query(append_or_overwrite: str, output: str, starting_definition: str) -> None:
    while True:
        if starting_definition:
            user_input = input(f"\nYou have chosen to {append_or_overwrite} '{output}, starting at {starting_definition}'.\nDo you wish to continue? (y/n): ")
        else: 
            user_input = input(f"\nYou have chosen to {append_or_overwrite} '{output}'.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")