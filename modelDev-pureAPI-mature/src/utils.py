def pureAPI_query(config):
    user_input = input(f"\n         ********************* >>>>> -- ModelDev PureAPI -- <<<<< *********************          \n\nThis script uses **{config['model_source']}** to classify selected case studies.\nYou have selected the data source **{config['txt_grandparent_dir']}**\nDo you wish to continue? (y/n): ")

    if user_input.lower() in ['yes', 'y']: pass
    elif user_input.lower() in ['no', 'n']: exit()
    else: 
        while True:
            user_input = input("Invalid input. Please enter Y or N: ")
            if user_input.lower() in ['yes', 'y']:
                break
            elif user_input.lower() in ['no', 'n']:
                exit()

def get_config(config_path='modelDev-pureAPI/config.yaml'):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def check_config(config):
    pass

