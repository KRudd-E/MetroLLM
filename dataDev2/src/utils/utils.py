def get_config(config_path:str='dataDev2/config.yaml') -> dict:
    """Load configuration from a YAML file."""
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    
def parser():
    """Parse command line arguments for the script."""
    
    import argparse
    parser = argparse.ArgumentParser(description="Script that runs dataDev2")
    parser.add_argument('--model_type', '-mt', type=str, required=True, help='Type of data to create: one of Text2Text, TextGen, or TextClass')
    parser.add_argument('--data_source', '-ds', type=str, required=True, help='Source of the data: one of Applications, Companies, or Definitions')
    
    inputs = parser.parse_args()

    #** Normalize inputs **#
    if      inputs.model_type.lower() in ['t2t','text2text']    : inputs.model_type = 'text2textgen'
    elif    inputs.model_type.lower() in ['tg','textgen']       : inputs.model_type = 'textgen'
    elif    inputs.model_type.lower() in ['tc','textclass']     : inputs.model_type = 'textclass'
    else: raise ValueError("Invalid model type (mt). Choose from: Text2Text, TextGen, or TextClass.")
    
    if      inputs.data_source.lower() in ['a','applications']  : inputs.data_source = 'applications'
    elif    inputs.data_source.lower() in ['c','companies']     : inputs.data_source = 'companies'
    elif    inputs.data_source.lower() in ['d','definitions']   : inputs.data_source = 'definitions'
    else:   raise ValueError("Invalid data source (ds). Choose from: Applications, Companies, or Definitions.")
    
    return inputs

def dataDev2_query(args) -> None:
    """Query for confirmation before proceeding with dataDev 2."""
    while True:
        user_input = input(f"\n         ********************* >>>>> -- DataDev2 -- <<<<< *********************          \n\nThis script will transform decomposed data into .csv files suitable for model development.\nYou have opted to create **{args.model_type}** formatted data, sourced from the **{args.data_source}** database.\nDo you wish to continue? (y/n): ")
        
        if user_input.lower() == 'y': 
            break # Proceed with the script
        elif user_input.lower() == 'n': 
            exit() # Exit the script
        else: 
            print("\nInvalid input. Please enter Y or N.")



#***** Text2Text Generation Queries *****
def t2t_app_query(starting_subfolder: str, output_dir: str, append_or_overwrite: str) -> None:
    """Query for confirmation before proceeding with Text2Text - applications database processing."""
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
    """Query for confirmation before proceeding with Text2Text - definitions database processing."""
    while True:
        if starting_definition:
            user_input = input(f"\nYou have chosen to {append_or_overwrite} '{output}, starting at {starting_definition}'.\nDo you wish to continue? (y/n): ")
        else: 
            user_input = input(f"\nYou have chosen to {append_or_overwrite} '{output}'.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")
    
def t2t_comp_query():
    """Query for confirmation before proceeding with Text2Text - companies database processing."""
    # Placeholder
    pass




#***** Text Classification Queries *****
def tc_app_query():
    """Query for confirmation before proceeding with Text Classification - applications database processing."""
    while True:
        user_input = input("\nYou have chosen to format the Applications database in preparation for Text Classification.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")

def tc_def_query():
    """"Query for confirmation before proceeding with Text Classification - definitions database processing."""
    while True:
        user_input = input("\nYou have chosen to format the Definitions database in preparation for Text Classification.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")

def tc_comp_query():
    """Query for confirmation before proceeding with Text Classification - companies database processing."""
    while True:
        user_input = input("\nYou have chosen to format the Companies database in preparation for Text Classification.\nDo you wish to continue? (y/n): ")
        if user_input.lower() == 'y': break
        elif user_input.lower() == 'n': exit()
        else: print("Invalid input. Please enter Y or N.")
    
    

    
#***** Text Generation Queries *****
def tg_app_query():
    """Query for confirmation before proceeding with Text Generation - applications database processing."""
    # Placeholder
    pass

def tg_def_query():
    """Query for confirmation before proceeding with Text Generation - definitions database processing."""
    # Placeholder
    pass

def tg_comp_query():
    """Query for confirmation before proceeding with Text Generation - companies database processing."""
    # Placeholder
    pass